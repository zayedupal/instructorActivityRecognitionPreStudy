import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, f1_score, balanced_accuracy_score, \
    roc_curve, auc, confusion_matrix, precision_score, recall_score

import ExtraSensoryHelperFunctions, ExtraSensoryFeaturesLabels

#timstamped user data folders
timeStampedDataFolderAcc = '/home/rana/Software&Data/Data/Upal/timestamped/watch_acc_walk_stand_sit_cw_npy_5secperseq/'
timeStampedDataFolderGyro = '/home/rana/Software&Data/Data/Upal/timestamped/gyro_acc_walk_stand_sit_cw_npy_5secperseq/'

resultPath = '/home/rana/Thesis/DrQA/upal/_Results/'

USER_THRESH = 15
SEQ_LEN_ACC = 125
SEQ_LEN_GYRO = 200
ACC_MODEL_PATH = 'C:/Users/zc01698/Desktop/_ExtrasensoryOutput/LSTM_RAW/Watch_Acc_125seq/Models/'
ACC_MODEL_NAME = 'LSTMAccRawNpy_L200_D200_SeqLen125_100epoch'
GYRO_MODEL_PATH = 'C:/Users/zc01698/Desktop/_ExtrasensoryOutput/LSTM_RAW/Phn_Gyro_200seq/Models/'
GYRO_MODEL_NAME = 'LSTMGyroRawNpy_L200_D200_SeqLen200_100epoch'

# WEIGHTS ARE CURRENTLY GIVEN BY PRECISION
ACC_FUSION_WEIGHTS = np.array([0.71,0.20,0.27,0.25])
GYRO_FUSION_WEIGHTS = np.array([0.49,0.25,0.18,0.28])
fused_weight_sum = ACC_FUSION_WEIGHTS+GYRO_FUSION_WEIGHTS
print("fused_weight_sum: ", fused_weight_sum)


def npyFileToTimeStampFeatureLabelDict(path,seq_len):
    with open(path, 'rb') as f:
        np_feature_lable = np.load(f, allow_pickle=True)

    if np_feature_lable.shape[0] <= 1:
        return np.empty()

    # each row contains the feautures and the last column is the label
    # we have to create sequences from it and label those
    np_features = np.array(np_feature_lable[:, :-1])
    np_labels_int = np.array(np_feature_lable[:, 3:])
    np_labels_bin = np.array(list(map(lambda x: ExtraSensoryHelperFunctions.IntToBinArr(x), np_labels_int)))
    # print('np_labels_bin shape: ', np_labels_bin.shape)
    # print('np_labels_int: ', np_labels_int[0])
    # print('np_labels_bin[0]: ', np_labels_bin[0])

    # create sequences with seq length
    cur_file_sequences = np.split(np_features, (np_features.shape[0] / seq_len))
    cur_file_sequences_labels = np.array(np.split(np_labels_bin, (np_labels_bin.shape[0] / seq_len)))
    # take only 1 label for 1 sequence
    cur_file_sequences_labels = np.array(cur_file_sequences_labels[:, 0])
    cur_file_sequences = np.array(cur_file_sequences)
    return cur_file_sequences, cur_file_sequences_labels


##########################################################################################
acc_model = ExtraSensoryHelperFunctions.load_model_keras(ACC_MODEL_PATH, ACC_MODEL_NAME)
gyro_model = ExtraSensoryHelperFunctions.load_model_keras(GYRO_MODEL_PATH, GYRO_MODEL_NAME)

users_acc = [x[1] for x in os.walk(timeStampedDataFolderAcc)][0]
# print(len(users_acc))
# print(users_acc)

users_gyro = [x[1] for x in os.walk(timeStampedDataFolderGyro)][0]
# print("gyro:",len(users_gyro))
# print(users_gyro)
common_users = set(users_acc).intersection(users_gyro)
# print("common_users:",len(common_users))
# print(common_users)

user_count = 0

prediction_probas = []
predictions = []
ground_truths = []

# go through each user id
for user_id in common_users:
    print("user_id: ", user_id)
    # find common timestamps
    user_path_acc = timeStampedDataFolderAcc + user_id + '/';
    user_path_gyro = timeStampedDataFolderGyro + user_id + '/';
    users_timestamps_acc = [x[2] for x in os.walk(user_path_acc)][0]
    users_timestamps_gyro = [x[2] for x in os.walk(user_path_acc)][0]
    common_users_timestamps = set(users_timestamps_acc).intersection(users_timestamps_gyro)

    inner_count = 0

    for common_user_t in common_users_timestamps:
        if not (os.path.exists(user_path_acc+common_user_t) and os.path.exists(user_path_gyro + common_user_t)):
            continue
        # get sequences and labels for common user timestamps
        acc_sequences, acc_labels = npyFileToTimeStampFeatureLabelDict(user_path_acc+common_user_t,SEQ_LEN_ACC)
        gyro_sequences, gyro_labels = npyFileToTimeStampFeatureLabelDict(user_path_gyro + common_user_t,SEQ_LEN_GYRO)

        # take same number of sequences from both acc and gyro
        if acc_sequences.shape[0]>gyro_sequences.shape[0]:
            acc_sequences = acc_sequences[0:gyro_sequences.shape[0],:,:]
            acc_labels = acc_labels[0:gyro_sequences.shape[0],:]
        elif acc_sequences.shape[0]<gyro_sequences.shape[0]:
            gyro_sequences = gyro_sequences[0:acc_sequences.shape[0],:,:]

        # both ground truth labels of acc and gyro
        # are same at the same timestamp for the same user
        labels = acc_labels

        # predict both sequences using corresponding model
        preds_acc = acc_model.predict(acc_sequences)
        weighted_preds_acc = preds_acc * ACC_FUSION_WEIGHTS
        # print("weighted_preds_acc: ", weighted_preds_acc)

        preds_gyro = gyro_model.predict(gyro_sequences)
        weighted_preds_gyro = preds_gyro * GYRO_FUSION_WEIGHTS
        # print("weighted_preds_gyro: ", weighted_preds_gyro)

        weight_preds_fused = (weighted_preds_acc+weighted_preds_gyro)/fused_weight_sum
        # print("weight_preds_fused: ", weight_preds_fused)

        # binarize the predictions
        rounded_preds = np.array(weight_preds_fused)
        rounded_preds[rounded_preds>=0.5] = 1
        rounded_preds[rounded_preds < 0.5] = 0

        # add preds and ground truths to corresponding arrays for metrics calculation later
        if len(ground_truths) == 0:
            ground_truths = labels
            predictions = rounded_preds
            prediction_probas =  np.array(weight_preds_fused,dtype=float)
        else:
            ground_truths = np.concatenate([ground_truths, labels])
            predictions = np.concatenate([predictions, rounded_preds])
            prediction_probas = np.concatenate([prediction_probas, weight_preds_fused])

        inner_count+=1
        # if inner_count >= 50:
        #     break
    user_count += 1;
    print("user_count: ", user_count)
    if user_count >=USER_THRESH:
        break

print("prediction_probas: ", prediction_probas)
print("predictions: ", predictions)
print("ground_truths: ", ground_truths)

############################################################################
# metrics calculations
############################################################################
n_classes = ground_truths.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
thresholds = dict()
conf_mat = dict()
metrics_texts = dict()
for i in range(0,n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(ground_truths[:, i], prediction_probas[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    conf_mat[i] = confusion_matrix(ground_truths[:, i], predictions[:, i])
    # print(conf_mat[i])
    metrics_text = f'category: {i} \n'
    metrics_text += f'Test accuracy: {accuracy_score(ground_truths[:, i], predictions[:, i])} \n'
    metrics_text += f'balanced accuracy: {balanced_accuracy_score(ground_truths[:, i], predictions[:, i])} \n'
    metrics_text += f'AUC: {roc_auc[i]} \n'
    metrics_text += f'f1 score: {f1_score(ground_truths[:, i], predictions[:, i])} \n'
    metrics_text += f'precision: {precision_score(ground_truths[:, i], predictions[:, i])} \n'
    metrics_text += f'recall: {recall_score(ground_truths[:, i], predictions[:, i])} \n'
    metrics_text += f'confusion matrix: \n {conf_mat[i]}'
    metrics_text += '\n'
    print(metrics_text)
    metrics_texts[i] = metrics_text

# write result to file
ExtraSensoryHelperFunctions.save_dict_file(resultPath+'LSTM_Acc_Gyro_Fusion_ROC_results.txt',metrics_texts)

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(ground_truths.ravel(), prediction_probas.ravel())
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
plt.savefig(resultPath+'LSTM_Acc_Gyro_Fusion_ROC.png')