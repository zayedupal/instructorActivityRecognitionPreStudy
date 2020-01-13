from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyspark.ml.feature import PCA
import numpy as np


# Read csv file
path = "sensorData.csv"

sparkSession = SparkSession.builder.appName('sensorClustering').getOrCreate()
input_df_spark = sparkSession.read.csv(path, header = True, inferSchema = True)

# add index column for easier identification
input_df_panda = input_df_spark.toPandas().reset_index()
input_df_spark = sparkSession.createDataFrame(input_df_panda)

from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(
    inputCols = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ', 'rotX', 'rotY', 'rotZ', 'rotS'],
    outputCol = 'features')
vector_df = vectorAssembler.transform(input_df_spark)

##PCA to reduce dimension of the feature vector, to plot
# vectorAssemblerRot = VectorAssembler(inputCols = ['rotX', 'rotY', 'rotZ', 'rotS'], outputCol = 'rotFeatures')
# vector_df_rot = vectorAssemblerRot.transform(input_df_spark)
# pca = PCA(k=3, inputCol="rotFeatures", outputCol="rotPCA")
# pca_model_rot = pca.fit(vector_df_rot)
# pca_rot_result = pca_model_rot.transform(vector_df_rot)
# pca_rot_result = pca_rot_result.toPandas()
# pca_rot_result.rotPCA = [float(item[0]) for item in pca_rot_result.rotPCA]

# Use kmeans
from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=2, seed=1)  # 2 clusters here
model = kmeans.fit(vector_df.select('features'))
# centers = model.clusterCenters()
kmeans_prediction = model.transform(vector_df)
kmeans_prediction.show()

#PCA to reduce dimension of the feature vector, to plot
pca = PCA(k=1, inputCol="features", outputCol="pcaFeatures")
pca_model = pca.fit(kmeans_prediction)
pca_result = pca_model.transform(kmeans_prediction).select("pcaFeatures")
pca_result = pca_result.toPandas().reset_index()
pca_result.pcaFeatures = [float(item[0]) for item in pca_result.pcaFeatures]

transformed = kmeans_prediction.select('index','prediction')
transformed = transformed.toPandas().merge(pca_result)
df_pred = sparkSession.createDataFrame(transformed)
df_pred.show()
pddf_pred = df_pred.toPandas().set_index('index')

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

#The Silhouette is a measure for the validation of the consistency within clusters.
# It ranges between 1 and -1, where a value close to 1 means that the points in a cluster
# are close to the other points in the same cluster and far from the points of the other clusters.
# In short closer value towards 1 means good clustering
silhouette = evaluator.evaluate(kmeans_prediction)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# # save result to csv
# df = transformed.toPandas()
#
# columns=['features','prediction']
# export_csv = df.to_csv('output.csv')
# print(export_csv)

# # Draw graphs
# fig = plt.figure(figsize=(10,10))
#
# ax = fig.add_subplot(2,2,1, projection='3d',title='accelerometer')
# ax.scatter(input_df_panda.accelX, input_df_panda.accelY,input_df_panda.accelZ, c=input_df_panda.index,cmap=plt.hot())
# ax.set_xlabel('accelX')
# ax.set_ylabel('accelY')
# ax.set_zlabel('accelZ')
#
# ax = fig.add_subplot(2,2,2, projection='3d',title='gyrosensor')
# ax.scatter(input_df_panda.gyroX, input_df_panda.gyroY,input_df_panda.gyroZ, c=input_df_panda.index,cmap=plt.hot())
# ax.set_xlabel('gyroX')
# ax.set_ylabel('gyroY')
# ax.set_zlabel('gyroZ')
#
# ax = fig.add_subplot(2,2,4, projection='3d', title='result')
#
# ax.scatter(pddf_pred.index, pddf_pred.pcaFeatures,pddf_pred.prediction, c=pddf_pred.prediction,cmap=plt.cm.coolwarm)
# ax.set_xlabel('index')
# ax.set_ylabel('pcaFeatures')
# ax.set_zlabel('prediction')
#
# plt.show()

# ax = fig.add_subplot(2,2,3, projection='3d',title='gyrosensor')
# ax.scatter(pca_rot_result.rotPCA[0], pca_rot_result.rotPCA[1],pca_rot_result.rotPCA[2], c=pca_rot_result.index,cmap=plt.hot())
# ax.set_xlabel('rotPCA_X')
# ax.set_ylabel('rotPCA_Y')
# ax.set_zlabel('rotPCA_Z')