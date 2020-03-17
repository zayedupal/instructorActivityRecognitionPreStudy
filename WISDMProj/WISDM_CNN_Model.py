import WISDM_Helper
import numpy as np
import datetime
import sklearn.preprocessing
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, f1_score, balanced_accuracy_score, \
    roc_curve, auc, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential

#################################################################################################################
# Constants and variables
#################################################################################################################
ACTIVITIES = ['A','D','E','F','Q']

# actual
ACCEL_DATA_PATH = '/home/rana/Software&Data/Data/Upal/wisdm-dataset/raw/watch/accel/'
GYRO_DATA_PATH = '/home/rana/Software&Data/Data/Upal/wisdm-dataset/raw/watch/gyro/'

# # test
# ACCEL_DATA_PATH = '/home/rana/Software&Data/Data/Upal/wisdm-dataset/raw/watch/mixed_test/acc/'
# GYRO_DATA_PATH = '/home/rana/Software&Data/Data/Upal/wisdm-dataset/raw/watch/mixed_test/gyro/'

resultPath = '/home/rana/Thesis/DrQA/upal/_Results/WISDM/CNN/'
profilePath = '/home/rana/Thesis/DrQA/upal/_Profiling/WISDM/CNN/'

LOOP_COUNT = 1
SAVE_MODEL_NAME = 'WISDM_CNN_Conv2_10_5_L200_D200'
SEQ_LEN = 100

# Hyperparameters
BATCH_SIZE = 10000
EPOCH_COUNT = 200

def main():

    #################################################################################################################
    # Preprocessing
    #################################################################################################################

    one_hot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)

    Xs,ys,_ = WISDM_Helper.handle_raw_files(acc_folder_path=ACCEL_DATA_PATH,gyro_folder_path=GYRO_DATA_PATH,
                                            ACTIVITIES=ACTIVITIES, one_hot_encoder=one_hot_encoder,seq_len=100)

    # print(ys)

    #################################################################################################################
    # Train and validation set division
    #################################################################################################################
    # divide train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(Xs, ys,random_state=0, test_size=0.3)


    #################################################################################################################
    # Build and Compile model
    #################################################################################################################
    n_features = len(X_train[0][0])
    n_classes = len(ACTIVITIES)

    for lc in range(0,LOOP_COUNT):
        # model 1
        model = Sequential()
        model.add(Conv1D(filters=10, kernel_size=25, activation='relu',input_shape=(SEQ_LEN, n_features)))
        model.add(MaxPooling1D())
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=10, kernel_size=25, activation='relu'))
        model.add(MaxPooling1D())

        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        METRICS = [
              'acc',
              keras.metrics.Precision(name='precision'),
              keras.metrics.Recall(name='recall'),
              keras.metrics.AUC(name='auc'),
        ]

        # compile the model using categorical_crossentropy for multi class classification
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)

    #################################################################################################################
    # profile
    #################################################################################################################
        log_dir = profilePath +'Run'+str(lc)+'/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, profile_batch=3)
        callbacks_list = [tensorboard_callback]

    #################################################################################################################
    # Fit and predict
    #################################################################################################################

        cur_result_path = resultPath+'Run'+str(lc)+'/'
        WISDM_Helper.create_folder(cur_result_path)

        callbacks_list.append(EarlyStopping(monitor='val_loss', patience=3))

        history= model.fit(
            X_train, Y_train, validation_data=(X_test,Y_test),
            batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,verbose=2,shuffle = False,
            callbacks=callbacks_list
        )

        # save model to file
        WISDM_Helper.save_model_keras(model,SAVE_MODEL_NAME+'Run'+str(lc),WISDM_Helper.MODEL_PATH)

        # predict
        pred_proba = model.predict(X_test,verbose=1)
        pred = pred_proba
        print('pred_proba: ', pred_proba)
        pred_max = np.argmax(np.array(pred_proba),axis=1)

        unique,count = np.unique(pred_max, return_counts=True)
        print(unique,count)
        y_max = np.argmax(np.array(Y_test),axis=1)
        unique,count = np.unique(y_max, return_counts=True)
        print(unique,count)

        pred = (pred_proba == pred_proba.max(axis=1)[:,None]).astype(int)
        print('pred: ', pred)
        print('Y_test: ', Y_test)

        conf_mat = multilabel_confusion_matrix(Y_test,pred)
        # conf_mat = confusion_matrix(Y_test,pred)
        print('conf mat: ')
        print(conf_mat)

        import matplotlib.pyplot as plt
        # summarize history for accuracy
        plt = WISDM_Helper.PlotEpochVsAcc(plt,history)
        plt.savefig(cur_result_path+SAVE_MODEL_NAME+'_Acc.png')

        # summarize history for loss
        plt = WISDM_Helper.PlotEpochVsLoss(plt,history)
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

        for i in range(0, n_classes):
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
        WISDM_Helper.save_dict_file(cur_result_path + SAVE_MODEL_NAME + '_results.txt', metrics_texts)

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(0, n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(1, n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=(12, 9))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = ['red', 'green', 'blue', 'orange', 'olive', 'purple', 'cyan']
        for i in range(0, n_classes):
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f}, conf_mat = \n{2})'
                           ''.format(np.array(ACTIVITIES)[i], roc_auc[i], conf_mat[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(cur_result_path + SAVE_MODEL_NAME + '_ROC.png')
