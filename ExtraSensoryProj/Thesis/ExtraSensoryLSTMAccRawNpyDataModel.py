# import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, f1_score, balanced_accuracy_score, \
    roc_curve, auc, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout

import ExtraSensoryFeaturesLabels, ExtraSensoryHelperFunctions

## Read all csv files inside a folder
## Change the folder path here
rawDataFolder = '/home/rana/Software&Data/Data/Upal/no_timestamp/watch_acc_norm_dropna_5sec/'
# rawDataFolder = 'C:/Users/zc01698/Desktop/Dataset/_ExtraSensory/ExtraSensory.raw_measurements.watch_acc/No timestamp npy/watch_acc_norm_natozero/'

resultPath = '/home/rana/Thesis/DrQA/upal/_Results/LSTM/Acc/'

LOOP_COUNT = 3
SAVE_MODEL_NAME = 'LSTMAccRawNpy_L200_D200_SeqLen125'
SEQ_LEN = 125
user_threshold =60
csv_df_panda = pd.DataFrame()

# Hyperparameters
BATCH_SIZE = 10000
EPOCH_COUNT = 100

user_count = 0
sequences = []
sequences_labels = []

# Go through each of the preprocessed folders and npy files
for path in Path(rawDataFolder).glob('**/*.npy'):
    print('preprocessed: ',path)

    # find the user_id for current folder
    splittedPath = str(path).split('\\')
    user_id = splittedPath[len(splittedPath)-1].split('.')[0]
    print('userid: ',user_id)

    # load the preprocessed csv data in dataframe
    # skip if it's empty
    with open(path, 'rb') as f:
        np_feature_lable = np.load(f,allow_pickle=True)

    if np_feature_lable.shape[0] <= 1:
        continue
    # each row contains the feautures and the last column is the label
    # we have to create sequences from it and label those
    np_features = np.array(np_feature_lable[:,:-1])
    np_labels_int = np.array(np_feature_lable[:,3:])

    np_labels_bin = np.array(list(map(lambda x:ExtraSensoryHelperFunctions.IntToBinArr(x),np_labels_int)))
    # print('np_labels_bin shape: ',np_labels_bin.shape)
    # print('np_labels_int: ', np_labels_int[0])
    # print('np_labels_bin[0]: ', np_labels_bin[0])


    # create sequences with seq length
    cur_file_sequences = np.split(np_features,(np_features.shape[0]/SEQ_LEN))
    cur_file_sequences_labels = np.array(np.split(np_labels_bin,(np_labels_bin.shape[0]/SEQ_LEN)))
    # take only 1 label for 1 sequence
    cur_file_sequences_labels = np.array(cur_file_sequences_labels[:,0])

    if len(sequences) == 0:
        sequences = cur_file_sequences
        sequences_labels = cur_file_sequences_labels
    else:
        sequences = np.concatenate([sequences,cur_file_sequences])
        sequences_labels = np.concatenate([sequences_labels, cur_file_sequences_labels])

    user_count+=1
    print('user_count: ', user_count)
    if user_count >= user_threshold:
        break
print('file_count: ', user_count)
sequences = np.array(sequences)
# sequences= sequences.reshape(sequences.shape[0],sequences.shape[1],3)
sequences_labels = np.array(sequences_labels)
# sequences_labels= sequences_labels.transpose(axis=1)
print('n_sequence,seq_len,n_features: ', sequences.shape)
print('sequences_labels shape: ', sequences_labels.shape)

# print(sequences)

n_features = sequences.shape[2]
n_classes = len(ExtraSensoryFeaturesLabels.labels)

print('n_features: ',n_features)
print('n_classes: ',n_classes)

# divide train and test set
X_train, X_test, Y_train, Y_test = train_test_split(sequences, sequences_labels,random_state=0, test_size=0.3)
# print(Y_train)
positive_counts = dict()
class_weights = ExtraSensoryHelperFunctions.calculating_class_weights(Y_train)
# build the model
# model 1
model = Sequential()
model.add(LSTM(200,input_shape=(SEQ_LEN,n_features)))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(n_classes, activation='sigmoid'))
## model 2
# model = Sequential()
# model.add(LSTM(200,input_shape=(SEQ_LEN,n_features),return_sequences=True))
# model.add(LSTM(100,return_sequences=True))
# model.add(LSTM(50,return_sequences=True))
# model.add(LSTM(25))
# model.add(Dense(n_classes, activation='sigmoid'))

# model.save(ExtraSensoryHelperFunctions.MODEL_PATH+'LSTM2')
METRICS = [
      # tensorflow.keras.metrics.TruePositives(name='tp'),
      # tensorflow.keras.metrics.FalsePositives(name='fp'),
      # tensorflow.keras.metrics.TrueNegatives(name='tn'),
      # tensorflow.keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)
model.compile(loss=ExtraSensoryHelperFunctions.get_weighted_loss(class_weights), optimizer='adam', metrics=METRICS)

# #Checkpoint
# checkpoint_path = "D:/Upal/Repositories/ComplexActivityRecognition/ExtraSensoryProj/Checkpoints/cp2.ckpt"
#
# # Create a callback that saves the model's weights
# cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

for lc in range(0,LOOP_COUNT):
    cur_result_path = resultPath+'Run'+str(lc)+'/'
    ExtraSensoryHelperFunctions.create_folder(cur_result_path)
    ## Fit model for multiple labels and print accuracy
    ## 2296*1130 = 2594485-5
    # history= model.fit(X_train, Y_train, validation_split=0.3,batch_size=10000, epochs=50,callbacks=[cp_callback])
    history= model.fit(
        X_train, Y_train, validation_split=0.3,
        batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,verbose=2,shuffle = False,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

    # save model to file
    ExtraSensoryHelperFunctions.save_model_keras(model,SAVE_MODEL_NAME,ExtraSensoryHelperFunctions.MODEL_PATH)

    # predict
    pred = model.predict(X_test,verbose=1)
    pred_proba = model.predict_proba(X_test)

    pred[pred>=0.5]=1
    pred[pred<0.5]=0
    # print('pred: ', pred)
    # print('Y_test: ', Y_test)

    conf_mat = multilabel_confusion_matrix(Y_test,pred)
    # print('conf mat: ')
    # print(conf_mat)

    # summarize history for accuracy
    plt = ExtraSensoryHelperFunctions.PlotEpochVsAcc(plt,history)
    plt.savefig(cur_result_path+SAVE_MODEL_NAME+'_Acc.png')

    # summarize history for loss
    plt = ExtraSensoryHelperFunctions.PlotEpochVsLoss(plt,history)
    plt.savefig(cur_result_path+SAVE_MODEL_NAME+'_Loss.png')

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    conf_mat = dict()
    metrics_texts = dict()

    acc = dict()
    ba = dict()
    f1 = dict()
    precision = dict()
    recall = dict()

    for i in range(0,n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(Y_test[:, i], pred_proba[:, i])

        conf_mat[i] = confusion_matrix(Y_test[:, i], pred[:, i])
        acc[i] = accuracy_score(Y_test[:, i], pred[:, i])
        ba[i] = balanced_accuracy_score(Y_test[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        f1[i] = f1_score(Y_test[:, i], pred[:, i])
        precision[i] = precision_score(Y_test[:, i], pred[:, i])
        recall[i] = recall_score(Y_test[:, i], pred[:, i])

        metrics_text = f'category: {i} \n'
        metrics_text += f'Test accuracy: {acc[i]} \n'
        metrics_text += f'balanced accuracy: {ba[i]} \n'
        metrics_text += f'AUC: {roc_auc[i]} \n'
        metrics_text += f'f1 score: {f1[i]} \n'
        metrics_text += f'precision: {precision[i]} \n'
        metrics_text += f'recall: {recall[i]} \n'
        metrics_text += f'confusion matrix: \n {conf_mat[i]}'
        metrics_text += '\n'
        # print(metrics_text)
        metrics_texts[i] = metrics_text

    metrics_text = f'AVERAGE:\n'
    metrics_text += f'Test accuracy: {np.array([acc[k] for k in acc]).mean()} \n'
    metrics_text += f'balanced accuracy: {np.array([ba[k] for k in ba]).mean()} \n'
    metrics_text += f'AUC: {np.array([roc_auc[k] for k in roc_auc]).mean()} \n'
    metrics_text += f'f1 score: {np.array([f1[k] for k in f1]).mean()} \n'
    metrics_text += f'precision: {np.array([precision[k] for k in precision]).mean()} \n'
    metrics_text += f'recall: {np.array([recall[k] for k in recall]).mean()} \n'
    metrics_text += '\n'
    metrics_texts['avg'] = metrics_text
    # write result to file
    ExtraSensoryHelperFunctions.save_dict_file(cur_result_path+SAVE_MODEL_NAME+'_results.txt',metrics_texts)

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(0,n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(1,n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(12,9))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = ['red','green','blue','orange','olive','purple','cyan']
    for i in range(0,n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f}, conf_mat = \n{2})'
                       ''.format(np.array(ExtraSensoryFeaturesLabels.labels)[i], roc_auc[i], conf_mat[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(cur_result_path+SAVE_MODEL_NAME+'_ROC.png')