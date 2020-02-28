import numpy as np
import pandas as pd
from keras import Sequential, Model, optimizers
from keras.layers import LSTM, Dropout, Dense,Flatten
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, f1_score, balanced_accuracy_score, \
    roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras

from instructorActivityRecognitionPreStudy.ExtraSensoryProj import ExtraSensoryFeaturesLabels, ExtraSensoryHelperFunctions
# import glob
from pathlib import Path
import matplotlib.pyplot as plt

## Read all csv files inside a folder
## Change the folder path here
rawDataFolder = 'D:/Upal/Dataset/ExtraSensory.raw_measurements.watch_acc/watch_acc/'
# processedFolder='D:/Upal/Dataset/ExtraSensory.per_uuid_features_labels/'
processedFolder='ExtraSensoryProj/ExtraSensoryData/'
sequences = []
sequences_labels = []
SEQ_LEN = 25
user_threshold = 50
csv_df_panda = pd.DataFrame()

user_count = 0
# Go through each of the preprocessed folders and csv files
for path in Path(processedFolder).glob('**/*.csv'):
    print('preprocessed: ',path)

    # find the user_id for current folder
    splittedPath = str(path).split('\\')
    user_id = splittedPath[len(splittedPath)-1].split('.')[0]
    print('userid: ',user_id)

    # load the preprocessed csv data in dataframe
    # skip if it's empty
    pre_df_panda = pd.read_csv(path)
    if pre_df_panda.shape[0] <= 1:
        continue
    
    # get the timestamps from the preprocessed data
    # then find out the raw data file for each timestamp
    # then load the raw data to np.array and
    # sort it according to the precise timestamp
    # sequences = []
    # sequences_labels = []
    for index, pre_df_row in pre_df_panda.iterrows():
        # get the file path
        curDatFilePath = ExtraSensoryHelperFunctions.findRawDataFile(rawDataFolder,user_id,int(pre_df_row['timestamp']))
        # replace unknown labels or NaNs by 0
        labels_df = pre_df_row[ExtraSensoryFeaturesLabels.labels].fillna(0)

        label = np.array(labels_df)
        if curDatFilePath is not None:
            curDatFile = open(curDatFilePath)
            lines = []
            # read the file and save it to lines
            for line in curDatFile:
                lines.append(line.strip().split(' '))
            curDatFile.close()
            # convert it ot np.array and sort with timestamp
            acc_feature_timestamp = np.array(lines,dtype=float)
            acc_feature_timestamp = acc_feature_timestamp[np.argsort(acc_feature_timestamp[:,0])]
            acc_features = acc_feature_timestamp[:,1:]
            # drop extra rows not divisible by sequence length
            if acc_features.shape[0]%SEQ_LEN > 0:
                acc_features = acc_features[:-(acc_features.shape[0]%SEQ_LEN),:]

            # create sequences with seq length
            cur_file_sequences = np.split(acc_features,(acc_features.shape[0]/SEQ_LEN))
            for cur_seq in cur_file_sequences:
                sequences.append(cur_seq)
                sequences_labels.append(label)

    user_count+=1
    print('user_count: ', user_count)
    if user_count >= user_threshold:
        break

sequences = np.array(sequences)
sequences_labels = np.array(sequences_labels)
print('n_sequence,seq_len,n_features: ', sequences.shape)
print('sequences_labels shape: ', sequences_labels.shape)
# print(sequences)

n_features = sequences.shape[2]
n_classes = len(ExtraSensoryFeaturesLabels.labels)

print('n_features: ',n_features)
print('n_classes: ',n_classes)

# divide train and test set
X_train, X_test, Y_train, Y_test = train_test_split(sequences, sequences_labels,random_state=0, test_size=0.33)

# build the model
model = Sequential()
model.add(LSTM(200,input_shape=(SEQ_LEN,n_features)))
model.add(Dense(200, activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(n_classes, activation='sigmoid'))


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

## Fit model for multiple labels and print accuracy
history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test),batch_size=64, epochs=200)

pred = model.predict(X_test,verbose=2)
pred_proba = model.predict_proba(X_test)
pred[pred>=0.5]=1
pred[pred<0.5]=0
print('pred: ', pred)
print('Y_test: ', Y_test)

conf_mat = multilabel_confusion_matrix(Y_test,pred)
print('conf mat: ')
print(conf_mat)

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