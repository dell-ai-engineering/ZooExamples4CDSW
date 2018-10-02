## Image Similarity
* This is a sample example of image similarity calculation. Both semantic and visually similarity are introduced. A real estate example was used to recommend similar houses based on the query image provided by users.

* This example introduces how to calculate the image similarity with deep learning neural networks provided in Analytics Zoo. A large number of problems in the computer vision domain can be solved by ranking images according to their similarity.

###### Environment
* Python 2.7/3.5/3.6
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)

###### Run with CDSW
*  Build the Analytics Zoo Engine using the docker file from this [Build Zoo Engine](https://github.com/dell-ai-engineering/bigdlengine4cdsw/tree/master/analytics-zoo)

*  Create a folder in CDSW under your project and import the image-similarity.py and make sure you have the spark-defaults.conf file to run the engine with necessary configurations for training the model.

* Add some of the parameter settings as you need in the spark-defaults.conf to start your workbench.

###### spark-defaults.conf
    spark.executor.memory                               32g
    spark.executor.cores                                1
    spark.driver.memory                                 15g
    spark.executor.instances                            1

###### Semantic Similarity Model
* For semantic similarity, multiple image classification models are used to category the image scene and add import semantic tags to image. Each tag (E.g. House style, number of stories) normally would require an indivisual multiClass classification model.

* Here we train a scene classification model as a sample. We use training data to train a multiclassification model, related categories are bathroom, bedroom, house, kitchen.



###### We will fine tune the model from a pretrained Inception on Places2 dataset.

1. Download the GoogLeNet-places365 model from the workbench terminal access menu by using the below commands:
```
wget https://github.com/CSAILVision/places365/blob/master/deploy_googlenet_places365.prototxt
wget http://places2.csail.mit.edu/models_places365/googlenet_places365.caffemodel
```

2. Download the images from the workbench terminal access menu by using the below command:
```
wget http://data.csail.mit.edu/places/places365/train_256_places365standard.tar
```
Unpack the downloaded tar file and copy the bathroom, bedroom, house, kitchen folders to a new folder named imageClassification, in which each category resides in its own sub folder. E.g. bedroom images will be in the path ".../imageClassification/bedroom".

3. Download the VGG16-places365 model from the workbench terminal access menu by using the below commands:
```
wget https://github.com/CSAILVision/places365/blob/master/deploy_vgg16_places365.prototxt
wget http://places2.csail.mit.edu/models_places365/vgg16_places365.caffemodel
```

4. Clone the analytics-zoo github for the SAMPLES folder which has the sample images that needs for this example. Here is the [GIT LINK](https://github.com/intel-analytics/analytics-zoo.git) for cloning. After cloning the analytics-zoo, just need to copy the SAMPLES folder from "analytics-zoo/apps/image-similarity/samples/".





#### These below steps are not mandatory unless you have enough space to download and store dataset onto the local.

After downloading all the models and images to your local, here is a hadoop command that copies the local data to HDFS, as we suggest to store and read large datasets from HDFS.
``` 
hdfs dfs -copyFromLocal <local_source_path> <hadoop_dest_path> (This step repeats for step-1,2,3,4 for moving all those downloaded data to HDFS)
```


Here are some sample steps that shows how to mention the hadoop path inside the code to read models and dataset from HDFS;
```
image_path = "hdfs:///user/leelau/zoo/images/img-classify/" (Reading images of bathroom, bedroom, house, kitchen from HDFS)
model_def_path = "hdfs:///user/leelau/zoo/images/googlenet_places365/deploy.prototxt" (Reading deploy.prototxt of googlenet-places365 model from HDFS)
model_weight_path = "hdfs:///user/leelau/zoo/images/googlenet_places365/googlenet_places365.caffemodel" (Reading caffemodel of googlenet-places365 model from HDFS)
vgg_def_path = "hdfs:///user/leelau/zoo/images/vgg_16_places365/deploy.prototxt" (Reading deploy.prototxt of vgg_16_places365 model from HDFS) 
vgg_weight_path = "hdfs:///user/leelau/zoo/images/vgg_16_places365/vgg16_places365.caffemodel" (Reading caffemodel of vgg_16_places365 model from HDFS)
image_path = "hdfs:///user/leelau/zoo/images/samples/*" (Reading sample images from HDFS)
```

