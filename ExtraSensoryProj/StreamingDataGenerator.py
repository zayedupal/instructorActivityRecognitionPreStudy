from time import sleep
from json import dumps
from kafka import KafkaProducer
import pandas as pd
import json
import csv

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],value_serializer=lambda x:dumps(x).encode('utf-8'))
with open("ComplexActivityRecognition/ExtraSensoryProj/ExtraSensoryData/0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv") as file:
# with open("ComplexActivityRecognition/ExtraSensoryProj/ExtraSensoryData/test.csv") as file:
    reader = csv.DictReader(file, delimiter=",")
    for row in reader:
        data = json.dumps(row)
        # data = row
        producer.send('test',value=data)
        sleep(1)
        print("Successfully sent data to kafka topic")
        print(data)