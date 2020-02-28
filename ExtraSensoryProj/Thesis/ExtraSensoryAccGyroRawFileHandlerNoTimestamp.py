import pickle
import mmap
import time
from pathlib import Path

import numpy as np
import pandas as pd

# from ComplexActivityRecognition.ExtraSensoryProj import ExtraSensoryFeaturesLabels, ExtraSensoryHelperFunctions
from instructorActivityRecognitionPreStudy.ExtraSensoryProj import ExtraSensoryFeaturesLabels, ExtraSensoryHelperFunctions

## Read all csv files inside a folder
## Change the folder path here

rawDataFolder = 'C:/Users/zc01698/Desktop/Dataset/_ExtraSensory/ExtraSensory.raw_measurements.proc_gyro/proc_gyro/'
processedFolder='C:/Users/zc01698/Desktop/Dataset/_ExtraSensory/ExtraSensory.per_uuid_features_labels/'
normalizedDataFolder = 'C:/Users/zc01698/Desktop/Dataset/_ExtraSensory/ExtraSensory.raw_measurements.proc_gyro/no_timestamp/phn_gyro_norm_dropna_1sec/'

# rawDataFolder = 'D:/Upal/Dataset/ExtraSensory.raw_measurements.watch_acc/watch_acc/'
# normalizedDataFolder = 'D:/Upal/Dataset/ExtraSensory.raw_measurements.watch_acc/watch_acc_norm_dropna/'
# processedFolder = 'D:/Upal/Dataset/ExtraSensory.per_uuid_features_labels/'

# processedFolder='D:/Upal/Repositories/ComplexActivityRecognition/ExtraSensoryProj/ExtraSensoryData/'

SEQ_LEN = 40
user_threshold = 2
# user_threshold = 2
csv_df_panda = pd.DataFrame()

user_count = 0

def FeatureFileHandlingFunc(path):
    print('preprocessed: ', path)
    # find the user_id for current folder
    splittedPath = str(path).split('\\')
    user_id = splittedPath[len(splittedPath) - 1].split('.')[0]
    print('userid: ', user_id)

    if Path(normalizedDataFolder + user_id + ".npy").is_file():
        return
    # load the preprocessed csv data in dataframe
    # skip if it's empty
    pre_df_panda = pd.read_csv(path)
    if pre_df_panda.shape[0] <= 1:
        return

    # get the timestamps from the preprocessed data
    # then find out the raw data file for each timestamp
    # then load the raw data to np.array and
    # sort it according to the precise timestamp
    # sequences = []
    # sequences_labels = []
    label_count = len(ExtraSensoryFeaturesLabels.labels)
    all_acc_features = np.empty((0, 4), dtype=float)
    count = 0
    for index, pre_df_row in pre_df_panda.iterrows():
        # # replace unknown labels or NaNs by 0
        # labels_df = pre_df_row[ExtraSensoryFeaturesLabels.labels].fillna(0)

        labels_df = pre_df_row[ExtraSensoryFeaturesLabels.labels]
        # print('labels: ', labels_df)
        # labels_df = labels_df.drop([0])
        # print('labels2: ', labels_df.columns)
        # remove NaNs
        if labels_df.isnull().any():
            # print('hehe')
            continue

        # labels_df = pre_df_row[ExtraSensoryFeaturesLabels.labels].dropna('any')

        label = np.array(labels_df)
        label_int = ExtraSensoryHelperFunctions.BitArrayToInt(label)
        # print('label arr: ',label)
        # print('label int: ',label_int)
        # get the file path
        curDatFilePath = ExtraSensoryHelperFunctions.findRawDataFile(rawDataFolder, user_id,
                                                                     int(pre_df_row['timestamp']))
        if curDatFilePath is not None:
            lines = []
            with open(curDatFilePath,'r+b') as curDatFile:
                map_file = mmap.mmap(curDatFile.fileno(),0,access=mmap.ACCESS_READ)
                # read the file and save it to lines
                while True:
                    line = map_file.readline()
                    if line == b'':break
                    # line =
                    # print('line: ',line.decode("utf-8"))
                    lines.append(line.decode("utf-8").strip().split(' '))
                map_file.close()
            curDatFile.close()
            # convert it ot np.array and sort with timestamp
            acc_feature_timestamp = np.array(lines, dtype=float)
            acc_feature_timestamp = acc_feature_timestamp[np.argsort(acc_feature_timestamp[:, 0])]
            acc_features = acc_feature_timestamp[:, 1:]
            # drop extra rows not divisible by sequence length
            if acc_features.shape[0] % SEQ_LEN > 0:
                acc_features = acc_features[:-(acc_features.shape[0] % SEQ_LEN), :]

            # count += acc_features.shape[0]

            label_rows = np.full((acc_features.shape[0], 1), label_int, dtype=float)
            acc_features = np.append(acc_features, label_rows, axis=1)
            # print(all_acc_features.shape)
            all_acc_features = np.append(all_acc_features,acc_features,axis=0)

    # user_count += 1
    # print('user_count: ', user_count)

    if all_acc_features.shape[0] > 0:
        # print(all_acc_features.shape)
        # all_acc_features, scaler = ExtraSensoryHelperFunctions.NormalizeFeatures(all_acc_features)
        all_acc_features = np.asarray(all_acc_features)
        np.save(normalizedDataFolder + user_id + ".npy", all_acc_features)
        # with open(normalizedDataFolder + user_id + ".npy",'wb') as f:
        #     pickle.dump(all_acc_features,f)


# Go through each of the preprocessed folders and csv files
start_time = time.time()
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(4)
file_array = Path(processedFolder).glob('**/*.csv')
pool.map(FeatureFileHandlingFunc, file_array)
pool.close()
pool.join()

elapsed_time = time.time()-start_time
print("runtime: ",elapsed_time)
# with open(normalizedDataFolder + '0A986513-7828-4D53-AA1F-E02D6DF9561B' + ".npy", 'rb') as f:
#     ulala = np.load(f,allow_pickle=True)
#     print('ulala: ',ulala.shape)
#     print(ulala[0])