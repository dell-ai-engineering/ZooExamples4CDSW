## Image Augmentation
* This is a simple example of image augmentation using Analytics ZOO API. We use various ways to transform images to augment the dataset. 

* Image Augmentation augments datasets (especially small datasets) to train model. The way to do image augmentation is to transform images by different ways. In this template we demonstrate how to do image augmentation using Analytics ZOO APIs.

###### Environment
* Python 2.7/3.5/3.6 (numpy 1.11.1)
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)

###### Run with CDSW
* Build the Analytics Zoo Engine using the docker file from this [Build Zoo Engine](https://github.com/dell-ai-engineering/bigdlengine4cdsw/tree/master/analytics-zoo)

* Create a folder in CDSW under your project and import the image-augmentation.py and make sure you have the spark-defaults.conf file to run the engine with necessary configurations for training the model.

* Add some of the parameter settings as you need in the spark-defaults.conf to start your workbench.

###### spark-defaults.conf
    spark.executor.memory                               10g
    spark.executor.cores                                1
    spark.driver.memory                                 10g
    spark.driver.cores                                  1
    spark.executor.instances                            4

* In this example, we have already downloaded the image dataset and put that into a folder as per the example that it needs to be. So you can directly run the example with the dataset provided in the image folder.


* Make sure we also need to create the empty folder "/tmp" under the image-augmentation example **(/home/cdsw/image-augmentation/tmp)** in the project directory of your CDSW since this example check for the "/tmp" folder to write the test image.

