## Object Detection Using SSD
* This is a simple example of object detection using Analytics Zoo Object Detection API. We use SSD-MobileNet to predict instances of target classes in the given video, which can be regarded as a sequence of images. The video is a scene of training a dog from [YouTube-8M dataset](YouTube-8M dataset) and the people and the dog are among target classes. Proposed areas are labeled with boxes and class scores.

* Object detection detects instances of certain classes in digital images and videos (which can be regarded as sequences of images). In this example, we demonstrate how to use pretrained Analytics Zoo model to detect objects in the video.

###### Environment
* Python 2.7/3.5/3.6 (Need moviepy)
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)

###### Run with CDSW
* Build the Analytics Zoo Engine using the docker file from this [Build Zoo Engine](https://github.com/dell-ai-engineering/bigdlengine4cdsw/tree/master/analytics-zoo)

* Create a folder in CDSW under your project and import the object-detection.py and make sure you have the spark-defaults.conf file to run the engine with necessary configurations for training the model.

* Add some of the parameter settings as you need in the spark-defaults.conf to start your workbench.

###### spark-defaults.conf
    spark.executor.memory                               8g
    spark.executor.cores                                2
    spark.driver.memory                                 8g
    spark.driver.cores                                  2
    spark.executor.instances                            4

We used one of the videos in Youtube-8M [LINK](link) for demo, which is a scene of training a dog. 


Download the SSD-MobileNet pretrained by PASCAL VOC dataset for detection from the workbench terminal access menu by using the following command:
```
wget https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model
```

Download the video using any of the video downloader and here is the link of the video [Video Dataset](https://www.youtube.com/watch?v=akcYAuaP4jw).




#### These below steps are not mandatory unless you have enough space to download and store dataset onto the local.

After downloading the pre-trained model to your local, here is a hadoop command that copies the local data to HDFS, as we suggest to store and read large datasets from HDFS. 
```
hdfs dfs -copyFromLocal <local_source_path> <hadoop_dest_path>
```


Here are some sample steps that shows how to mention the hadoop path inside the code to read model and dataset from HDFS;
```
model_path = "hdfs:///user/leelau/zoo/obj-det/apps/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model" (Reading Pre-trained model from HDFS)
path = "hdfs:///user/leelau/zoo/obj-det/apps/train_dog.mp4" (Reading the video dataset from HDFS)
```
