import os
import csv
# from ComplexActivityRecognition.ExtraSensoryProj import ExtraSensoryFeaturesLabels

# import tensorflow
from tensorflow.keras.models import model_from_json
# from tensorflow import keras
from upal import ExtraSensoryFeaturesLabels
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump, load
import keras
import pickle
# constants
# MODEL_PATH = 'ComplexActivityRecognition/ExtraSensoryProj/SavedModels/'
MODEL_PATH = '/home/rana/Thesis/DrQA/upal/_Models/'

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_dict_file(path,texts):
    result_file = open(path, "wb")
    pickle.dump(texts, result_file)
    result_file.close()

def save_model_keras(model,name,path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(path+name+'.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path+name+'.h5')
    print("Saved model to: ", path)

def load_model_keras(path,name):
    # load json and create model
    json_file = open(path+name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+name+'.h5')
    print("Loaded model from disk")
    return loaded_model

def get_actual_date_labels(tick_seconds):
    import datetime;
    import pytz;

    time_zone = pytz.timezone('US/Pacific');  # Assuming the data comes from PST time zone
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    datetime_labels = [];

    tick_datetime = datetime.datetime.fromtimestamp(tick_seconds, tz=time_zone);
    weekday_str = weekday_names[tick_datetime.weekday()];
    time_of_day = tick_datetime.strftime('%I:%M%p');
    datetime_labels.append('%s\n%s' % (weekday_str, time_of_day));


    return datetime_labels;

def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return keras.backend.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*keras.backend.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss

def get_weighted_loss_singleclass(weights):
    def weighted_loss(y_true, y_pred):
        return keras.backend.mean((weights[0]**(1-y_true))*(weights[1]**(y_true))*keras.backend.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss

def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights

def IntToBinArr(x):
    bit = [0, 0, 0, 0]
    x = int(x)
    if x == 0: return bit

    count = 0
    while x:
        # bit.append(x % 2)
        bit[count] = (x % 2)
        x >>= 1
        count = count+1
    return np.array(bit[::-1])

def BitArrayToInt(arr):
    out = 0
    for bit in arr:

        out = (out<<1)|int(bit)
    return out

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

def ReadCSVToArray(filepath):
    with open(filepath, 'r') as outfile:
        csv_reader = csv.reader(outfile, delimiter=',')
        arr = []
        for row in csv_reader:
            arr.append(row)
    outfile.close()
    return arr

def WriteArrayToCSV(arr,filepath):
    with open(filepath, 'w+') as outfile:
        csv_writer = csv.writer(outfile, delimiter=',')
        csv_writer.writerow(arr)
    outfile.close()

from sklearn.preprocessing import MinMaxScaler
def NormalizeFeatures(arr):
    scaler = MinMaxScaler()
    scaler.fit(arr)
    return scaler.transform(arr),scaler

def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train,axis=0);
    std_vec = np.nanstd(X_train,axis=0);
    return (mean_vec,std_vec);

def standardize_features(X,mean_vec,std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1,-1));
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1,-1));
    X_standard = X_centralized / normalizers;
    return X_standard;


def SaveModelSklearn(model,name,path):
    dump(model, path+name+'.joblib')
    return path+name+'.joblib'

def LoadModelSklearn(name,path):
    return load(path+name+'.joblib')

def printUniqueCount(arr,str):
    number, count = np.unique(arr, return_counts=True)
    print(str)
    print(number)
    print(count)

def findRawDataFile(folderPath, userID,timestamp):
    if os.path.isdir(folderPath+userID):
        # print('findRawDataFile isdir: ', folderPath + userID,timestamp)
        for file in Path(folderPath+userID).glob(str(timestamp)+'*'):
            # print('raw file: ',folderPath+userID+'/'+file.name)
            return folderPath+userID+'/'+file.name
    return None

def dropUnimportantColumns(df):
    features_df = df[ExtraSensoryFeaturesLabels.features]
    labels_df = df[ExtraSensoryFeaturesLabels.labels]
    result_df = pd.concat([features_df,labels_df],axis=1)
    return result_df

def dropAnyNANFeatures(df):
    # drop rows which has no labels
    df = pd.DataFrame(df,columns=df.columns,dtype=np.float32)
    df = df.reset_index()
    # Take the index only as key of the rows and concat it with labels columns
    label_df = df[df.columns[0:1]]
    label_df = pd.concat(
        [
            label_df,
            df[ExtraSensoryFeaturesLabels.labels]
        ], axis=1
    )
    # drop rows which has no labels
    # dropping these will cause the model to not understand unidentified labels
    label_df = label_df.dropna(how='all')
    # fill the NaNs as 0s
    label_df = label_df.fillna(0)
    # print('label_df: ',label_df)

    # Take the index only as key of the rows and concat it with features columns
    # After that we convert it to float
    features_df = df[df.columns[0:1]]
    features_df = pd.concat(
        [
            features_df,
            df[ExtraSensoryFeaturesLabels.features],
        ], axis=1
    )
    # drop rows which has nans in features
    features_df = features_df.dropna()
    # print('features_df: ', features_df)
    label_df.drop(columns=['index'],inplace=True)
    # Now inner join the label columns
    # so that we get the whole rows,
    # but the rows with missing values in features are dropped
    result = pd.concat([features_df, label_df], axis=1, join='inner')
    result.apply(pd.to_numeric, errors='coerce')

    return result

def dropAnyNANFeaturesTimestampIncluded(df):
    # drop rows which has no labels
    df = df.reset_index()
    # Take the index only as key of the rows and concat it with labels columns
    label_df = df[df.columns[0:1]]
    label_df = pd.concat(
        [
            label_df,
            df[ExtraSensoryFeaturesLabels.labels]
        ], axis=1
    )
    # drop rows which has no labels
    label_df = label_df.dropna(how='all')
    # fill the NaNs as 0s
    label_df = label_df.fillna(0)

    # Take the index only as key of the rows and concat it with features columns
    # After that we convert it to float
    features_df = df[df.columns[0:2]]
    features_df = pd.concat(
        [
            features_df,
            df[ExtraSensoryFeaturesLabels.features],
        ], axis=1
    )
    # timestamp_df = df['timestamp']
    # print(timestamp_df)
    # drop rows which has no features
    features_df = features_df.dropna()
    label_df.drop(columns=['index'],inplace=True)
    # Now inner join the label columns
    # so that we get the whole rows,
    # but the rows with missing values in features are dropped
    result = pd.concat([features_df, label_df], axis=1, join='inner')
    result.apply(pd.to_numeric, errors='coerce')

    return result

def dropAllNANFeaturesOthersZero(df):
    # drop rows which has no labels
    df = df.reset_index()
    # Take the index only as key of the rows and concat it with labels columns
    label_df = df[df.columns[0:1]]
    label_df = pd.concat(
        [
            label_df,
            df[ExtraSensoryFeaturesLabels.labels]
        ], axis=1
    )
    # drop rows which has no labels
    label_df = label_df.dropna(how='all')
    # fill the NaNs as 0s
    label_df = label_df.fillna(0)

    # Take the index only as key of the rows and concat it with features columns
    # After that we convert it to float
    features_df = df[df.columns[0:1]]
    features_df = pd.concat(
        [
            features_df,
            df[ExtraSensoryFeaturesLabels.features],
        ], axis=1
    )

    # drop rows which has no features
    features_df = features_df.dropna(how='all')
    # fill the NaNs as 0s
    features_df.fillna(0, inplace=True)

    # Now inner join the label columns
    # so that we get the whole rows,
    # but the rows with missing values in features are dropped
    result = pd.concat([features_df, label_df], axis=1, join='inner')
    result.apply(pd.to_numeric, errors='coerce')

    return result


def fillNANFeaturesByZero(df):
    # drop rows which has no labels
    df = df.reset_index()
    # Take the index only as key of the rows and concat it with labels columns
    label_df = df[df.columns[0:1]]
    label_df = pd.concat(
        [
            label_df,
            df[ExtraSensoryFeaturesLabels.labels]
        ], axis=1
    )
    # drop rows which has no labels
    label_df = label_df.dropna(how='all')
    # fill the NaNs as 0s
    label_df = label_df.fillna(0)

    # Take the index only as key of the rows and concat it with features columns
    # After that we convert it to float
    features_df = df[df.columns[0:1]]
    features_df = pd.concat(
        [
            features_df,
            df[ExtraSensoryFeaturesLabels.features],
        ], axis=1
    )

    # fill the NaNs as 0s
    features_df.fillna(0, inplace=True)

    # Now inner join the label columns
    # so that we get the whole rows,
    # but the rows with missing values in features are dropped
    result = pd.concat([features_df, label_df], axis=1, join='inner')
    result.apply(pd.to_numeric, errors='coerce')

    return result


def nanToZeros(df):
    result = pd.concat(
        [
            df['timestamp'],
            df[ExtraSensoryFeaturesLabels.features],
            df[ExtraSensoryFeaturesLabels.labels]
        ], axis=1
    )
    result.fillna(0, inplace=True)
    return result