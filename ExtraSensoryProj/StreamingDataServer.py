import sys
import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.4.4 pyspark-shell'
# os.environ['PYSPARK_SUBMIT_ARGS'] = ' — packages org.apache.spark:spark-streaming-kafka-0–8_2.11:2.1.0,org.apache.spark:spark-sql-kafka-0–10_2.11:2.1.0 pyspark-shell'
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from uuid import uuid1
import json
from pyspark.mllib.clustering import StreamingKMeans
from pyspark.mllib.linalg import SparseVector, DenseVector
from pyspark.sql import SQLContext, SparkSession
sc = SparkContext(appName="Sparkstreaming")
spark = SparkSession.builder.appName("Spark-Kafka-Integration").master("local").getOrCreate()
ssc = StreamingContext(sc,1)
kafka_stream = KafkaUtils.createStream(ssc,"localhost:2181","raw-event-streaming-consumer",{"test":1})
# raw = kafka_stream.flatMap(lambda kafkaS: [kafkaS])
d_stream = kafka_stream.map(lambda v: json.loads(v[1]))
d_stream.count().map(lambda x:'datapoints in this batch: %s' % x).pprint()
# lines = raw.map(lambda xs: xs[1].split("|"))
d_stream.pprint()
ssc.start()
ssc.awaitTermination()


