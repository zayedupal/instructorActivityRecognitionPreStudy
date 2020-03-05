import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd
import os
import pickle

#################################################################################################################
# Other helpers
#################################################################################################################
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_dict_file(path,texts):
    result_file = open(path, "wb")
    pickle.dump(texts, result_file)
    result_file.close()

#################################################################################################################
# Data wrangling
#################################################################################################################
def dateparse (time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))

def concat_np_array(parent,child):
    if len(parent) == 0:
        parent = child
    else:
        parent = np.concatenate((parent, child))
    return parent

def create_sequence(df,label,time_steps):
    Xs, ys = [], []
    for i in range(df.shape[0] - time_steps):
        v = df.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(label)
    # print('Xs length: ',len(Xs),'Ys length: ',len(ys))
    return np.array(Xs),np.array(ys)

def handle_raw_files(acc_folder_path,gyro_folder_path,ACTIVITIES, one_hot_encoder,seq_len=100):
    sequences = np.empty(0)
    sequences_labels = np.empty(0)

    start_time = time.time()

    # sorted file list, so we can get same user's file for both acc and gyro
    acc_file_array = sorted(Path(acc_folder_path).glob('**/*.txt'))
    gyro_file_array = sorted(Path(gyro_folder_path).glob('**/*.txt'))


    files = []
    activity_data_dict = dict()

    for acc_f,gyro_f in zip(acc_file_array,gyro_file_array):
        acc_df = pd.read_csv(acc_f,index_col=2,header=None)
        acc_df.columns=['user','activity','acc_x','acc_y','acc_z']
        acc_df['acc_z']=acc_df['acc_z'].astype(str).str[:-1].astype(np.float)
        gyro_df = pd.read_csv(gyro_f, index_col=2,header=None)
        gyro_df.columns = ['user', 'activity', 'gyro_x', 'gyro_y', 'gyro_z']
        gyro_df['gyro_z'] = gyro_df['gyro_z'].astype(str).str[:-1].astype(np.float)

        # take rows with selected activities
        acc_df = acc_df[acc_df['activity'].isin(ACTIVITIES)]
        gyro_df = gyro_df[gyro_df['activity'].isin(ACTIVITIES)]

        # take common timestamps and join acc and gyro data
        merged_df = acc_df.merge(gyro_df,left_index=True, right_index=True,suffixes=['_acc','_gyro'])

        # ignore rows which have same timestamp but different label
        merged_df = merged_df.loc[merged_df['activity_acc']==merged_df['activity_gyro']]
        merged_df.drop(['user_gyro','activity_gyro'],axis=1,inplace=True)
        merged_df.rename(columns={'activity_acc':'activity','user_acc':'user'},inplace=True)

        one_hot_encoder.fit(np.array(ACTIVITIES).reshape(-1,1))

        print('Creaating sequence for user ',merged_df.head(1)['user'].values[0])

        # for each activity of current user create data list and sequence
        for act in ACTIVITIES:
            # print('Creaating sequence for user ',merged_df.head(1)['user'].values[0], 'activity ',act)
            activity_data_dict[act] = merged_df[merged_df['activity']==act].drop(merged_df.columns[[1, 2]], axis=1)
            # print('data of cur act: ',activity_data_dict[act])
            # one hot encode
            one_hot_encoded = one_hot_encoder.transform(np.array(act).reshape(-1,1))
            Xs, ys = create_sequence(activity_data_dict[act],one_hot_encoded.flatten(),seq_len)
            sequences = concat_np_array(sequences,Xs)
            sequences_labels = concat_np_array(sequences_labels, ys)

    print('total len of seq: ',len(sequences))
    print('total len of seq labels: ',len(sequences_labels))

    return sequences, sequences_labels


#################################################################################################################
# Plots
#################################################################################################################
def PlotEpochVsAcc(plt,history):
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    return plt

def PlotEpochVsLoss(plt,history):
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    return plt