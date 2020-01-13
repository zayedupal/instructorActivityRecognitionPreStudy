import json
import pandas as pd
from pandas.io.json import json_normalize
from kafka import KafkaConsumer
from ComplexActivityRecognition.ExtraSensoryProj import ExtraSensoryFeaturesLabels, ExtraSensoryHelperFunctions
import numpy as np

models = dict()
models['COMPUTER_WORK'] = ExtraSensoryHelperFunctions.LoadModelSklearn('COMPUTER_WORK',ExtraSensoryHelperFunctions.MODEL_PATH)
models['FIX_walking'] = ExtraSensoryHelperFunctions.LoadModelSklearn('FIX_walking',ExtraSensoryHelperFunctions.MODEL_PATH)
models['OR_standing'] = ExtraSensoryHelperFunctions.LoadModelSklearn('OR_standing',ExtraSensoryHelperFunctions.MODEL_PATH)
models['SITTING'] = ExtraSensoryHelperFunctions.LoadModelSklearn('SITTING',ExtraSensoryHelperFunctions.MODEL_PATH)
models['TALKING'] = ExtraSensoryHelperFunctions.LoadModelSklearn('TALKING',ExtraSensoryHelperFunctions.MODEL_PATH)

# load model presets
mean_vec = ExtraSensoryHelperFunctions.ReadCSVToArray(ExtraSensoryHelperFunctions.MODEL_PATH + 'mean_vec.csv')
mean_vec = mean_vec[0]
std_vec = ExtraSensoryHelperFunctions.ReadCSVToArray(ExtraSensoryHelperFunctions.MODEL_PATH + 'std_vec.csv')
std_vec = std_vec[0]

while True:
    try:
        consumer = KafkaConsumer('test',auto_offset_reset='latest', bootstrap_servers=['localhost:9092'])
        # consumer.subscribe([rubis])
        for msg in consumer:
            response = json.loads(msg.value.decode('utf-8'))
            # print(response)
            json_data=json.loads(response)
            df = json_normalize(json_data)

            input_df_panda = ExtraSensoryHelperFunctions.dropUnimportantColumns(df)
            # print(input_df_panda)
            input_df_panda = ExtraSensoryHelperFunctions.dropAnyNANFeatures(df)
            input_df_panda = pd.DataFrame(input_df_panda, columns=input_df_panda.columns, dtype=np.float32)
            features = input_df_panda[ExtraSensoryFeaturesLabels.features]
            labels = input_df_panda[ExtraSensoryFeaturesLabels.labels]

            mean_vec = np.array(mean_vec,dtype=np.float32)
            std_vec = np.array(std_vec,dtype=np.float32)

            # the model has standardized features
            ExtraSensoryHelperFunctions.standardize_features(features, mean_vec,std_vec)

            if features.empty:
                print('Ignoring empty row')
                continue
            print('features:', features)
            results = dict()
            for modelname in models:
                result = models[modelname].predict(features)
                results[modelname] = result[0]
            print(results)
    except Exception as e:
        print("Error: ", (e))