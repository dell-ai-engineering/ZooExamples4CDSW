import re

from bigdl.nn.criterion import CrossEntropyCriterion
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType

from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.api.keras.layers import Dense, Input, Flatten
from zoo.pipeline.api.keras.models import *
from zoo.pipeline.api.net import *
from zoo.pipeline.nnframes import *

sc = init_nncontext("ImageTransferLearningExample")

model_path = "hdfs:///user/leelau/zoo/demo/bigdl_inception-v1_imagenet_0.4.0.model"
image_path = "hdfs:///user/leelau/zoo/demo/*/*"
imageDF = NNImageReader.readImages(image_path, sc)

getName = udf(lambda row:
                  re.search(r'(cat|dog)\.([\d]*)\.jpg', row[0], re.IGNORECASE).group(0),
                  StringType())
getLabel = udf(lambda name: 1.0 if name.startswith('cat') else 2.0, DoubleType())

labelDF = imageDF.withColumn("name", getName(col("image"))) \
        .withColumn("label", getLabel(col('name')))
(trainingDF, validationDF) = labelDF.randomSplit([0.9, 0.1])
labelDF.select("name","label").show(10)


# Fine-tune a pre-trained model
# We fine-tune a pre-trained model by removing the last few layers, freezing the first few layers, and adding some new layers.
transformer = ChainedPreprocessing(
        [RowToImageFeature(), ImageResize(256, 256), ImageCenterCrop(224, 224),
         ImageChannelNormalize(123.0, 117.0, 104.0), ImageMatToTensor(), ImageFeatureToTensor()])

# Load a pre-trained model
# We use the Net API to load a pre-trained model, including models saved by Analytics Zoo, BigDL, Torch, Caffe and Tensorflow. 
full_model = Net.load_bigdl(model_path)


# Remove the last few layers
# Here we print all the model layers and you can choose which layer(s) to remove.
# When a model is loaded using Net, we can use the newGraph(output) api to define a Model with the output specified by the parameter.
for layer in full_model.layers:
    print (layer.name())
model = full_model.new_graph(["pool5/drop_7x7_s1"])

# Freeze some layers
# We freeze layers from input to pool4/3x3_s2 inclusive.
model.freeze_up_to(["pool4/3x3_s2"])

# Add a few new layers
inputNode = Input(name="input", shape=(3, 224, 224))
inception = model.to_keras()(inputNode)
flatten = Flatten()(inception)
logits = Dense(2)(flatten)
lrModel = Model(inputNode, logits)
classifier = NNClassifier(lrModel, CrossEntropyCriterion(), transformer) \
        .setLearningRate(0.003).setBatchSize(40).setMaxEpoch(1).setFeaturesCol("image") \
        .setCachingSample(False)
pipeline = Pipeline(stages=[classifier])


# Train the model
# The transfer learning can finish in a few minutes.
catdogModel = pipeline.fit(trainingDF)
predictionDF = catdogModel.transform(validationDF).cache()
predictionDF.select("name","label","prediction").sort("label", ascending=False).show(10)
predictionDF.select("name","label","prediction").show(10)
correct = predictionDF.filter("label=prediction").count()
overall = predictionDF.count()
accuracy = correct * 1.0 / overall
print("Test Error = %g " % (1.0 - accuracy))

# Visualize result
# We randomly select some images to show, and print the prediction results here.
# cat: prediction = 1.0 dog: prediction = 2.0
samplecat=predictionDF.filter(predictionDF.prediction==1.0).limit(3).collect()
sampledog=predictionDF.filter(predictionDF.prediction==2.0).sort("label", ascending=False).limit(3).collect()

from IPython.display import Image, display
for cat in samplecat:
    print ("prediction:"), cat.prediction
    display(Image(cat.image.origin[5:]))
    
for dog in sampledog:
    print ("prediction:"), dog.prediction
    display(Image(dog.image.origin[5:]))    
