import WISDM_Helper
import sklearn.preprocessing

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

resultPath = '/home/rana/Thesis/DrQA/upal/_Results/WISDM/LSTM/'

LOOP_COUNT = 3
SAVE_MODEL_NAME = 'WISDM_LSTM_L200_D200'
SEQ_LEN = 100

# Hyperparameters
BATCH_SIZE = 10000
EPOCH_COUNT = 200

#################################################################################################################
# Preprocessing
#################################################################################################################

one_hot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)

Xs,ys,total_rows = WISDM_Helper.handle_raw_files(acc_folder_path=ACCEL_DATA_PATH,gyro_folder_path=GYRO_DATA_PATH,
                                        ACTIVITIES=ACTIVITIES, one_hot_encoder=one_hot_encoder,seq_len=100)

print("total sensor data count: ",total_rows)