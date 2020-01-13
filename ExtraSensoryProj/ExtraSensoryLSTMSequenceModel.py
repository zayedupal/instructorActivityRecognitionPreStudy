import numpy as np
import pandas as pd
from keras import Sequential, Model, optimizers
from keras.layers import LSTM, Dropout, Dense,Flatten
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score, roc_auc_score,roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras

from ComplexActivityRecognition.ExtraSensoryProj import ExtraSensoryFeaturesLabels, ExtraSensoryHelperFunctions
# import glob
from pathlib import Path
import matplotlib.pyplot as plt
from keras.utils import Sequence

class MyBatchGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=1, shuffle=True):
        'Initialization'
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *self.X[index].shape))
        yb = np.empty((self.batch_size, *self.y[index].shape))
        # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
            yb[s] = self.y[index]
        return Xb, yb


## Read all csv files inside a folder
## Change the folder path here
folderPath='C:/Users/zc01698/Desktop/Dataset/_ExtraSensory/ExtraSensory.per_uuid_features_labels/'
# folderPath='ComplexActivityRecognition/ExtraSensoryProj/SmallestDataFile/'
# folderPath='ComplexActivityRecognition/ExtraSensoryProj/ExtraSensoryData/'

n_classes = len(ExtraSensoryFeaturesLabels.labels)
n_features = len(ExtraSensoryFeaturesLabels.features)

sequences = []
sequences_labels = []
csv_df_panda = pd.DataFrame()
for path in Path(folderPath).glob('**/*.csv'):
    cur_df_panda = pd.read_csv(path)
    cur_df_panda = ExtraSensoryHelperFunctions.dropAnyNANFeatures(cur_df_panda)
    if cur_df_panda.shape[0]<=1:
        continue
    print(path,cur_df_panda.shape)
    cur_file_features = cur_df_panda[ExtraSensoryFeaturesLabels.features]
    cur_file_labels = cur_df_panda[ExtraSensoryFeaturesLabels.labels]
    # normalize features
    normalize_scaler = MinMaxScaler(feature_range=(0, 1))
    nor_file_features = normalize_scaler.fit_transform(cur_file_features)
    cur_file_features = pd.DataFrame(nor_file_features,columns=cur_file_features.columns,index=cur_file_features.index)

    print(cur_file_labels.shape)
    # print(cur_file_features)

    # get consecutive rows with same label combination as a sequence
    # what should we do with the rows which have All zeroes for the labels?
    same_count = 1
    cur_file_row_label = []
    cur_file_row_label = np.array(cur_file_row_label)

    seq_count = 0

    sequence = []

    for index,labels in cur_file_labels.iterrows():
        # print('featu: ', cur_file_features[cur_file_features.index==index])
        if np.array_equal(np.array(labels),cur_file_row_label):
            # sequence is continuing. add current values to the sequence
            # print('heh: ',np.array(cur_file_features[cur_file_features.index == index]).reshape(n_features))
            sequence.append(np.array(cur_file_features[cur_file_features.index == index]).reshape(n_features))
            same_count += 1
        else:
            if len(cur_file_row_label)>0:
                #not the first time
                # print('labels: ',cur_file_row_label,same_count)
                # end of previous sequence.
                # add it in the sequencelist
                sequences.append(np.array(sequence))

            # beginning of new sequence, label it
            sequence = []
            same_count = 1
            cur_file_row_label = np.array(labels)
            sequence.append(np.array(cur_file_features[cur_file_features.index == index]).reshape(n_features))
            sequences_labels.append(np.array(labels))

    # add the last sequence
    sequences.append(np.array(sequence))



    # print('sequencelist: ', np.array(sequences).shape)
    # print('sequencelabels: ', np.array(sequences_labels).shape)
    # print('sequencelabels: ', np.array(sequences_labels))

sequences = np.array(sequences)
sequences_labels = np.array(sequences_labels)
print('sequencelist: ', sequences.shape)
print('sequencelabels: ', sequences_labels.shape)
print(sequences)
print(sequences_labels)

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]



# divide train and test set
X_train, X_test, Y_train, Y_test = train_test_split(sequences, sequences_labels,random_state=0, test_size=0.33)


colors = ['red','green','blue','orange','olive','purple','cyan']
model = Sequential()
model.add(LSTM(200,input_shape=(None,n_features)))
model.add(Dense(200, activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(n_classes, activation='sigmoid'))
#
# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)

## Fit model for multiple labels and print accuracy
history = model.fit_generator(generator=MyBatchGenerator(X_train, Y_train, batch_size=1), epochs=5)
pred = model.predict_generator(generator=MyBatchGenerator(X_test,Y_test,batch_size=1),verbose=2)
print(pred)
# pred_proba = model.predict_proba(X_test)
# preds = model.predict(X_test)
# preds[preds>=0.5] = 1
# preds[preds<0.5] = 0
# # acc = history.history['accuracy']
# print(Y_test)
# print(preds)
#
#
# n_classes = len(ExtraSensoryFeaturesLabels.labels)
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# thresholds = dict()
# conf_mat = dict()
# for i in range(0,n_classes):
#     fpr[i], tpr[i], thresholds[i] = roc_curve(Y_test[:, i], pred_proba[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#     conf_mat[i] = confusion_matrix(Y_test[:, i], preds[:, i])
#     print(conf_mat[i])
#
#
# # print(thresholds.shape)
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), pred_proba.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# # Compute macro-average ROC curve and ROC area
#
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(0,n_classes)]))
#
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(1,n_classes):
#     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#
# # Finally average it and compute AUC
# mean_tpr /= n_classes
#
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# # Plot all ROC curves
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)
#
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)
#
# for i in range(0,n_classes):
#     plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
#              label='ROC curve of class {0} (area = {1:0.2f}, conf_mat = \n{2})'
#                    ''.format(np.array(ExtraSensoryFeaturesLabels.labels)[i], roc_auc[i], conf_mat[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()
