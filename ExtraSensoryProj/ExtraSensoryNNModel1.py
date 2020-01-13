import numpy as np
import pandas as pd
from keras import Sequential, Model, optimizers
from keras.layers import LSTM, Dropout, Dense,Input
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

X_train = np.array(train[ExtraSensoryFeaturesLabels.features])
X_test = np.array(test[ExtraSensoryFeaturesLabels.features])

Y_train = np.array(train[ExtraSensoryFeaturesLabels.labels])
Y_test = np.array(test[ExtraSensoryFeaturesLabels.labels])

# It is recommended to standardize the features (subtract mean and divide by standard deviation),
# so that all their values will be roughly in the same range:
model_load_params = dict()
model_load_params['mean_vec'],model_load_params['std_vec'] = ExtraSensoryHelperFunctions.estimate_standardization_params(X_train);
X_train = ExtraSensoryHelperFunctions.standardize_features(X_train,model_load_params['mean_vec'],model_load_params['std_vec']);
X_test = ExtraSensoryHelperFunctions.standardize_features(X_test,model_load_params['mean_vec'],model_load_params['std_vec']);

# # normalize features
# scaler_train = MinMaxScaler(feature_range=(0, 1))
# X_train = scaler_train.fit_transform(X_train)
# scaler_test = MinMaxScaler(feature_range=(0, 1))
# X_test = scaler_test.fit_transform(X_test)

print('shape of X_train, Y_train, X_test, Y_test dataset: ')
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

colors = ['red','green','blue','orange','olive','purple','cyan']

# Model
model = Sequential()
model.add(Dense(n_features, activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))

# With binary cross entropy, you can only classify two classes. With categorical cross entropy, you're not limited to how many classes your model can classify.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)

# different metrices
fpr = dict()
tpr = dict()
roc_auc = dict()
thresholds = dict()
conf_mat = dict()
Y_test_ravel = []
preds_prob_ravel = []

# fit model for each of the labels independently
for i in range(0,n_classes):
    # Fit model for multiple labels and print accuracy
    history = model.fit(X_train,Y_train[:,i],epochs=20,batch_size=200,verbose=2,class_weight='balanced')
    pred_proba = model.predict_proba(X_test)
    preds = model.predict(X_test)
    preds[preds>=0.5] = 1
    preds[preds < 0.5] = 0

    # acc = history.history['accuracy']
    ExtraSensoryHelperFunctions.printUniqueCount(Y_train[:, i],'train set count: ')
    ExtraSensoryHelperFunctions.printUniqueCount(Y_test[:, i], 'test set count: ')
    ExtraSensoryHelperFunctions.printUniqueCount(preds, 'prediction set count: ')

    fpr[i], tpr[i], thresholds[i] = roc_curve(Y_test[:, i], pred_proba)
    roc_auc[i] = auc(fpr[i], tpr[i])
    conf_mat[i] = confusion_matrix(Y_test[:, i], preds)
    print('confusion matrix: ')
    print(conf_mat[i])

    # for later use in graph
    Y_test_ravel.append(Y_test[:, i].ravel())
    preds_prob_ravel.append(pred_proba.ravel())

Y_test_ravel = np.array(Y_test_ravel)
preds_prob_ravel = np.array(preds_prob_ravel)

# graphs
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test_ravel.ravel(), preds_prob_ravel.ravel())
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
