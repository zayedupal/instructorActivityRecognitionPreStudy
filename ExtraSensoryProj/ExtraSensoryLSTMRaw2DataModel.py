# import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, f1_score, balanced_accuracy_score, \
    roc_curve, auc, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from instructorActivityRecognitionPreStudy.ExtraSensoryProj import ExtraSensoryFeaturesLabels, ExtraSensoryHelperFunctions

## Read all csv files inside a folder
## Change the folder path here
rawDataFolder = 'C:/Users/zc01698/Desktop/Dataset/_ExtraSensory/ExtraSensory.raw_measurements.watch_acc/watch_acc_normalized/'

SEQ_LEN = 25
user_threshold = 5
csv_df_panda = pd.DataFrame()

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
    print('np_labels_bin shape: ',np_labels_bin.shape)
    print('np_labels_int: ', np_labels_int[0])
    print('np_labels_bin[0]: ', np_labels_bin[0])


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
positive_counts = dict()
class_weights = ExtraSensoryHelperFunctions.calculating_class_weights(Y_train)
# build the model
# # model 1
# model = Sequential()
# model.add(LSTM(200,input_shape=(SEQ_LEN,n_features)))
# model.add(Dense(200, activation='relu',kernel_initializer='he_uniform'))
# model.add(Dense(n_classes, activation='sigmoid'))
## model 2
model = Sequential()
model.add(LSTM(500,input_shape=(SEQ_LEN,n_features),return_sequences=True))
model.add(LSTM(200,return_sequences=True))
model.add(LSTM(100,return_sequences=True))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(25))
model.add(Dense(n_classes, activation='sigmoid'))

# model.save(ExtraSensoryHelperFunctions.MODEL_PATH+'LSTM2')
METRICS = [
      # keras.metrics.TruePositives(name='tp'),
      # keras.metrics.FalsePositives(name='fp'),
      # keras.metrics.TrueNegatives(name='tn'),
      # keras.metrics.FalseNegatives(name='fn'),
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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)
# model.compile(loss=ExtraSensoryHelperFunctions.get_weighted_loss(class_weights), optimizer='adam', metrics=METRICS)

# #Checkpoint
# checkpoint_path = "D:/Upal/Repositories/ComplexActivityRecognition/ExtraSensoryProj/Checkpoints/cp2.ckpt"
#
# # Create a callback that saves the model's weights
# cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

## Fit model for multiple labels and print accuracy
## 2296*1130 = 2594485-5
# history= model.fit(X_train, Y_train, validation_split=0.3,batch_size=10000, epochs=50,callbacks=[cp_callback])
history= model.fit(X_train, Y_train, validation_split=0.3,batch_size=10000, epochs=50,verbose=2)

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
ExtraSensoryHelperFunctions.PlotEpochVsAcc(plt,history)

# summarize history for loss
ExtraSensoryHelperFunctions.PlotEpochVsLoss(plt,history)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
thresholds = dict()
conf_mat = dict()
for i in range(0,n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(Y_test[:, i], pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    conf_mat[i] = confusion_matrix(Y_test[:, i], pred[:, i])
    # print(conf_mat[i])
    metrics_text = f'category: {i} \n'
    metrics_text += f'Test accuracy: {accuracy_score(Y_test[:, i], pred[:, i])} \n'
    metrics_text += f'balanced accuracy: {balanced_accuracy_score(Y_test[:, i], pred[:, i])} \n'
    metrics_text += f'AUC: {roc_auc[i]} \n'
    metrics_text += f'f1 score: {f1_score(Y_test[:, i], pred[:, i])} \n'
    metrics_text += f'precision: {precision_score(Y_test[:, i], pred[:, i])} \n'
    metrics_text += f'recall: {recall_score(Y_test[:, i], pred[:, i])} \n'
    metrics_text += f'confusion matrix: \n {conf_mat[i]}'
    metrics_text += '\n'
    print(metrics_text)

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
plt.figure()
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
plt.show()