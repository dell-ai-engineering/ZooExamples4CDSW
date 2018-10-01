    image-similarityimage-similarity.py

    Download
    Open In Workbench

import re

from bigdl.nn.criterion import CrossEntropyCriterion
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType

from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.api.keras.layers import Dense, Input, Flatten
from zoo.pipeline.api.net import *
from zoo.pipeline.nnframes import *

sc = init_nncontext("ImageSimilarityExample")

image_path = "hdfs:///user/leelau/zoo/images/img-classify/*/*"
imageDF = NNImageReader.readImages(image_path, sc, 24).sample(False, 0.4)

def getLabel(path):
    if "bathroom" in path: label = 1.0
    elif "bedroom" in path: label = 2.0
    elif "house" in path: label = 3.0
    else: label = 4.0
    return label

getLabelUDF = udf(lambda row: getLabel(row[0]), DoubleType())
labelDF = imageDF.withColumn("label", getLabelUDF(col('image')))
(trainingDF, validationDF) = labelDF.randomSplit([0.8, 0.2])
trainingDF.cache()
print("training data distribution:")
trainingDF.groupBy("label").count().show()

model_def_path = "hdfs:///user/leelau/zoo/images/googlenet_places365/deploy.prototxt"
model_weight_path = "hdfs:///user/leelau/zoo/images/googlenet_places365/googlenet_places365.caffemodel"

# load pretrained caffe model
preTrained_model = Net.load_caffe(model_def_path, model_weight_path)

# create a new model by remove layers after pool5/drop_7x7_s1
part_model = preTrained_model.new_graph(["pool5/drop_7x7_s1"])

# optionally freeze layers from input to pool4/3x3_s2 inclusive
# model.freeze_up_to(["pool4/3x3_s2"])


from bigdl.nn.layer import Sequential, Linear, View, LogSoftMax
from bigdl.nn.criterion import CrossEntropyCriterion

# add a new linear layer with numClass outputs, in our example, it's 6.
scene_network = Sequential().add(part_model).add(View([1024])).add(Linear(1024, 6)).add(LogSoftMax())

transformer = ChainedPreprocessing(
    [RowToImageFeature(), ImageResize(256, 256), ImageCenterCrop(224, 224),
     ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(), ImageFeatureToTensor()])

classifier = NNClassifier(scene_network, CrossEntropyCriterion(), transformer).setLearningRate(0.001).setLearningRateDecay(1e-3).setBatchSize(20).setMaxEpoch(2).setFeaturesCol("image").setCachingSample(False)

# train the model
scene_classification_model = classifier.fit(trainingDF)
print("Finished training")


# evaluate the model
print("Start evaluation:")
predictionDF = scene_classification_model.transform(validationDF).cache()
predictionDF.sample(False, 0.01).show()

correct = predictionDF.filter("label=prediction").count()
overall = predictionDF.count()
accuracy = correct * 1.0 / overall
predictionDF.unpersist()
trainingDF.unpersist()
print("Accuracy = %g " % accuracy)




# Visual Similarity Model
# Triplet (query image, positive image, negative image)
vgg_def_path = "hdfs:///user/leelau/zoo/images/vgg_16_places365/deploy.prototxt"
vgg_weight_path = "hdfs:///user/leelau/zoo/images/vgg_16_places365/vgg16_places365.caffemodel"

transformer = ChainedPreprocessing(
    [RowToImageFeature(), ImageResize(256, 256), ImageCenterCrop(224, 224),
     ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(), ImageFeatureToTensor()])

full_model = Net.load_caffe(vgg_def_path, vgg_weight_path)

from bigdl.nn.layer import Sequential, View, Normalize

# create a new model by removing layers after pool5
model = full_model.new_graph(["pool5"])
# generates a vector of dimension 25088 for each image
vggModel = Sequential().add(model).add(View([25088])).add(Normalize(2.0))

embeddingModel = NNModel(vggModel, transformer).setFeaturesCol("image")
embeddingModel.transform(imageDF.limit(10)).show()




# Qury for similar images
# we selected a combination of sample images from places dataset in the samples folder
image_path = "hdfs:///user/leelau/zoo/images/samples/*"
imageDF = NNImageReader.readImages(image_path, sc)

pathDF = with_origin_column(imageDF)
sceneDF = scene_classification_model.setBatchSize(4).setPredictionCol("sceneClassification").transform(pathDF)
embeddingDF = embeddingModel.setBatchSize(4).setPredictionCol("embedding").transform(sceneDF)

# save data in a dict, where key is the path of the image, value are category and embedding
imageDict = embeddingDF.select("origin", "sceneClassification", "embedding").rdd.map(lambda r: (r[0], (r[1], r[2]))).collectAsMap()
print(str(len(imageDict)) + " images cached in imageDict")


from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, struct, col
import numpy as np
from heapq import nlargest

query_origin = "hdfs://pNameNode.vcse.lab:8020/user/leelau/zoo/images/samples/00000099.jpg"
query_record = imageDict[query_origin]
query_classification = query_record[0]
query_embedding = query_record[1]

def get_score(x):
    candidate_classification = x[0]
    candidate_embedding = x[1]
    classScore = 1.0 if query_classification == candidate_classification else 0.0
    visualScore = np.dot(query_embedding, candidate_embedding)
    return classScore + visualScore

score_dict = {k: get_score(v) for k, v in imageDict.items()}

# select the top 3 most similar images
top_3 = nlargest(3, score_dict, key=score_dict.get)
print(top_3)
