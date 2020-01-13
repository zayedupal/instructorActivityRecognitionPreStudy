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

## Read all csv files inside a folder
## Change the folder path here
folderPath='C:/Users/zc01698/Desktop/Dataset/_ExtraSensory/ExtraSensory.per_uuid_features_labels/'
# folderPath='ComplexActivityRecognition/ExtraSensoryProj/ExtraSensoryData/'

csv_df_panda = pd.DataFrame()
for path in Path(folderPath).glob('**/*.csv'):
    print(path)
    cur_df_panda = pd.read_csv(path)
    if len(csv_df_panda)==0:
        csv_df_panda = cur_df_panda
    else:
        csv_df_panda = csv_df_panda.append(cur_df_panda)

print('shape of whole dataset: ')
print(csv_df_panda.shape)

# take only needed columns
csv_df_panda = ExtraSensoryHelperFunctions.dropUnimportantColumns(csv_df_panda)

# For all the below missing value handling methods,
# First rows which has NaN in all labels, are dropped

# rows with any missing values are dropped
input_df_panda = ExtraSensoryHelperFunctions.dropAnyNANFeatures(csv_df_panda)

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

n_classes = len(ExtraSensoryFeaturesLabels.labels)
n_features = len(ExtraSensoryFeaturesLabels.features)

categories = ExtraSensoryFeaturesLabels.labels
print(categories)
print('shape of dataset after dropping missing labels: ')
print(input_df_panda.shape)


# divide train and test set
train, test = train_test_split(input_df_panda, random_state=0, test_size=0.33, shuffle=True)

for c in categories:
    unique, counts = np.unique(train[c], return_counts=True)
    print('unique')
    print(unique)
    print('counts')
    print(counts)

X_train = np.array(train[ExtraSensoryFeaturesLabels.features])
X_test = np.array(test[ExtraSensoryFeaturesLabels.features])

Y_train = np.array(train[ExtraSensoryFeaturesLabels.labels])
Y_test = np.array(test[ExtraSensoryFeaturesLabels.labels])

# normalize features
scaler_train = MinMaxScaler(feature_range=(0, 1))
X_train = scaler_train.fit_transform(X_train)
scaler_test = MinMaxScaler(feature_range=(0, 1))
X_test = scaler_test.fit_transform(X_test)

# reshape data for feeding into LSTM
# samples, timestep, features
n_train_samples = X_train.shape[0]
n_test_samples = X_test.shape[0]
n_timestep = 1

X_train = X_train.reshape(n_train_samples,n_timestep,n_features)
X_test = X_test.reshape(n_test_samples,n_timestep,n_features)
# Y_train = Y_train.reshape(1,Y_train.shape[0],n_classes)
# Y_test = Y_test.reshape(1,Y_test.shape[0],n_classes)
print('shape of X_train, Y_train, X_test, Y_test dataset: ')
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

colors = ['red','green','blue','orange','olive','purple','cyan']
# print(X_train.shape[0])
# print(X_train.shape[1])
# print(X_train.shape[2])
model = Sequential()
model.add(LSTM(100,input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dense(100, activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(n_classes, activation='sigmoid'))

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)

## Fit model for multiple labels and print accuracy
history = model.fit(X_train,Y_train,epochs=20,batch_size=32,verbose=2,class_weight='auto')

pred_proba = model.predict_proba(X_test)
preds = model.predict(X_test)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
# acc = history.history['accuracy']
print(Y_test)
print(preds)


n_classes = len(ExtraSensoryFeaturesLabels.labels)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
thresholds = dict()
conf_mat = dict()
for i in range(0,n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(Y_test[:, i], pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    conf_mat[i] = confusion_matrix(Y_test[:, i], preds[:, i])
    print(conf_mat[i])


# print(thresholds.shape)
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), pred_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area

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
