import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, balanced_accuracy_score,roc_auc_score,roc_curve,auc,precision_score,recall_score

from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import utils

# from ComplexActivityRecognition.ExtraSensoryProj import ExtraSensoryFeaturesLabels, ExtraSensoryHelperFunctions
from sklearn.preprocessing import MinMaxScaler

from ExtraSensoryProj import ExtraSensoryFeaturesLabels, ExtraSensoryHelperFunctions

# import glob
from pathlib import Path
import matplotlib.pyplot as plt

## Read all csv files inside a folder
## Change the folder path here
# folderPath='C:/Users/zc01698/Desktop/Dataset/_ExtraSensory/ExtraSensory.per_uuid_features_labels/'
# folderPath='ComplexActivityRecognition/ExtraSensoryProj/ExtraSensoryData/'

folderPath = 'D:/Upal/Dataset/ExtraSensory.per_uuid_features_labels/'
# folderPath='D:/Upal/Repositories/ComplexActivityRecognition/ExtraSensoryProj/ExtraSensoryData/'

csv_df_panda = pd.DataFrame()
for path in Path(folderPath).glob('**/*.csv'):
    print('hola: ',path)
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

# rows with missing values in all features are dropped others are filled with 0's
# input_df_panda = ExtraSensoryHelperFunctions.dropAllNANFeaturesOthersZero(csv_df_panda)

# filling missing values in features with 0's
# input_df_panda = ExtraSensoryHelperFunctions.fillNANFeaturesByZero(csv_df_panda)

# # merging has no better effect
# # merge walking and strolling
# input_df_panda['label:FIX_walking'] = input_df_panda['label:FIX_walking'].replace(0,input_df_panda['label:STROLLING'])
# input_df_panda.drop(['label:STROLLING'],axis=1)
# ExtraSensoryFeaturesLabels.labels.remove('label:STROLLING')

categories = ExtraSensoryFeaturesLabels.labels
print(categories)
print('shape of dataset after dropping missing labels: ')
print(input_df_panda.shape)

# divide train and test set
train, test = train_test_split(input_df_panda, random_state=0, test_size=0.3, shuffle=True)

X_train = train[ExtraSensoryFeaturesLabels.features]
X_test = test[ExtraSensoryFeaturesLabels.features]

# normalize features
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaler_train.fit(X_train)
X_train = scaler_train.transform(X_train)
X_test = scaler_train.transform(X_test)

# # It is recommended to standardize the features (subtract mean and divide by standard deviation),
# # so that all their values will be roughly in the same range:
# model_load_params = dict()
# model_load_params['mean_vec'],model_load_params['std_vec'] = ExtraSensoryHelperFunctions.estimate_standardization_params(X_train);
# X_train = ExtraSensoryHelperFunctions.standardize_features(X_train,model_load_params['mean_vec'],model_load_params['std_vec']);
# X_test = ExtraSensoryHelperFunctions.standardize_features(X_test,model_load_params['mean_vec'],model_load_params['std_vec']);
#
#
# # save load params to a file
# ExtraSensoryHelperFunctions.WriteArrayToCSV(model_load_params['mean_vec'],ExtraSensoryHelperFunctions.MODEL_PATH+'mean_vec.csv')
# ExtraSensoryHelperFunctions.WriteArrayToCSV(model_load_params['std_vec'],ExtraSensoryHelperFunctions.MODEL_PATH+'std_vec.csv')

print('shape of training dataset: ')
print(X_train.shape)
print('shape of test dataset: ')
print(X_test.shape)

## Fit model for multiple labels and print accuracy
from sklearn import tree
# model = tree.DecisionTreeClassifier(criterion='gini',class_weight='balanced')
SVC_pipeline = Pipeline([
                # ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear'), n_jobs=1))
                # ('clf', OneVsRestClassifier(tree.DecisionTreeClassifier(criterion='gini'), n_jobs=1))
                ('clf', OneVsRestClassifier(RandomForestClassifier(criterion='gini',class_weight='balanced'), n_jobs=1))
            ])


# different metrices
fpr = dict()
tpr = dict()
roc_auc = dict()
thresholds = dict()
conf_mat = dict()
graph_text = dict()

i=0
test_merged = []
pred_merged = []
pred_merged=np.array(pred_merged)

for category in categories:
    print('... Processing {}'.format(category))
    unique, counts = np.unique(test[category], return_counts=True)
    countDict = dict(zip(unique, counts))
    print(countDict)

    clf = SVC_pipeline.fit(X_train, train[category])

    # ExtraSensoryHelperFunctions.SaveModelSklearn(clf,category.split(':')[1],ExtraSensoryHelperFunctions.MODEL_PATH)

    pred_proba = SVC_pipeline.predict_proba(X_test)
    prediction = SVC_pipeline.predict(X_test)
    if i==0:
        test_merged = test[category]
        pred_merged = pred_proba[:, 1]
    else:
        test_merged = test_merged.append(test[category])

        pred_merged = np.append(pred_merged,pred_proba[:,1])

    fpr[i], tpr[i], thresholds[i] = roc_curve(test[category], pred_proba[:,1], pos_label=1)
    # roc_auc[i] = auc(test[category],pred_proba[:,1])
    conf_mat[i] = confusion_matrix(test[category], prediction)

    roc_auc[i] = roc_auc_score(test[category],pred_proba[:,1])

    metrics_text = f'{category}\n'
    metrics_text += f'Test accuracy: {accuracy_score(test[category], prediction)} \n'
    metrics_text += f'balanced accuracy: {balanced_accuracy_score(test[category], prediction)} \n'
    metrics_text += f'AUC: {roc_auc[i]} \n'
    metrics_text += f'f1 score: {f1_score(test[category], prediction)} \n'
    metrics_text += f'precision: {precision_score(test[category], prediction)} \n'
    metrics_text += f'recall: {recall_score(test[category], prediction)} \n'
    metrics_text += f'confusion matrix: \n {conf_mat[i]}'
    metrics_text += '\n'
    graph_text[i] = metrics_text
    print(metrics_text)
    i+=1


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_merged.ravel(), pred_merged.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(0,len(categories))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(0,len(categories)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(categories)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
# fig, (plt, ax2) = plt.subplots(1, 2,figsize=(16,8))
plt.figure(figsize=(16,8))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = ['red','green','blue','orange','olive','purple','cyan']
for i in range(0,len(categories)):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label='ROC curve of class {0} (area = {1:0.2f}, conf_mat = \n{2})'
             ''.format(np.array(ExtraSensoryFeaturesLabels.labels)[i], roc_auc[i],conf_mat[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()



# from imblearn.over_sampling import RandomOverSampler

# ros = RandomOverSampler()
# X_ros, y_ros = ros.fit_sample(X_train, train[category])
# SVC_pipeline.fit(X_ros, y_ros)

# from imblearn.over_sampling import SMOTE
#
# smote = SMOTE('minority')
# X_sm, y_sm = smote.fit_resample(X_train,train[category])
# SVC_pipeline.fit(X_sm,y_sm)

# from sklearn.utils import class_weight
# class_weight = class_weight.compute_class_weight('balanced',n)

### LogisticRegression gave better accuracy_score
### So far we have seen "Sitting accuracy is around 68%
### So far we have seen "Talking accuracy is around 89%, which is suspicious
### Sleeping accuracy is around 75%, this is not so important label
### others are around 90%