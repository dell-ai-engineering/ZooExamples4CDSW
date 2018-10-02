## Transfer Learning
In this template, we will use a pre-trained Inception_V1 model. But we will operate on the pre-trained model to freeze first few layers, replace the classifier on the top, then fine tune the whole model. And we use the fine-tuned model to solve the dogs-vs-cats classification problem.

###### Environment
* Python 2.7 
* JDK 8
* Apache Spark 2.x
* CDSW 1.3.0

###### Run the CDSW
* Build the Analytics Zoo Engine using the docker file from this [Build Zoo Engine](https://github.com/dell-ai-engineering/bigdlengine4cdsw/tree/master/analytics-zoo)

* Create a folder in CDSW under your project and import the transfer-learning.py and make sure you have the spark-defaults.conf file to run the engine with necessary configurations for training the model.

* Add some of the parameter settings as you need in the spark-defaults.conf to start your workbench

###### spark-defaults.conf
    spark.executor.memory                               10g
    spark.executor.cores                                1
    spark.driver.memory                                 10g
    spark.driver.cores                                  1
    spark.executor.instances                            4


Download the pre-trained model from the workbench terminal access menu using the below command:
```
wget https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1_imagenet_0.4.0.model
```

Download the training dataset from this link [Download Dataset](https://www.kaggle.com/c/dogs-vs-cats/download/train.zip) and extract it by using this commands:
```
unzip train.zip
```

Note: You have to create an account with any of your email accounts in that website before downloading the dataset.

The following commands copy about 1100 images of cats and dogs into demo/cats and demo/dogs separately. Please make sure you download the dataset onto your local and create the folders and copy the images according to the following steps:

>  #> mkdir -p demo/dogs

>  #> mkdir -p demo/cats

>  #> cp train/cat.7* demo/cats

>  #> cp train/dog.7* demo/dogs





#### These below steps are not mandatory unless you have enough space to download and store dataset onto the local.

After downloading the pre-trained model and images to your local, here is a hadoop command that copies the local data to HDFS, as we suggest to store and read large datasets from HDFS.
```
hdfs dfs -copyFromLocal <local_source_path> <hadoop_dest_path>
```


Here are some sample steps that shows how to mention the hadoop path inside the code to read model and dataset from HDFS;
```
model_path = "hdfs:///user/leelau/zoo/demo/bigdl_inception-v1_imagenet_0.4.0.model" (Reading Pre-trained model from HDFS)
image_path = "hdfs:///user/leelau/zoo/demo/*/*" (Reading the dogs and cats images from HDFS)
```
