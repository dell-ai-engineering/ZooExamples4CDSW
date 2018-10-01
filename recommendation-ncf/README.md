## Recommendation-NCF
* This example demonstrates how to build a neural network recommendation system (Neural Collaborative Filtering) with explict feedback using Analytics Zoo and BigDL on Spark. We use Recommender API in Analytics Zoo to build a model, and use optimizer of BigDL to train the model.

* The system [Recommendation systems: Principles, methods and evaluation](Recommendation systems: Principles, methods and evaluation) normally prompts the user through the system interface to provide ratings for items in order to construct and improve the model. The accuracy of recommendation depends on the quantity of ratings provided by the user.

* NCF leverages a multi-layer perceptrons to learn the user–item interaction function, at the mean time, NCF can express and generalize matrix factorization under its framework. includeMF(Boolean) is provided for users to build a NCF with or without matrix factorization.

###### Environment
* Python 2.7/3.5/3.6
* JDK 8
* Spark 1.6.0/2.1.1/2.1.2/2.2.0(This version needs to be same with the version you use to build Analytics Zoo)
* Analytics Zoo 0.2.0/0.3.0

###### Run with CDSW
* Build the Analytics Zoo Engine using the docker file from this [Build Zoo Engine](https://github.com/dell-ai-engineering/bigdlengine4cdsw/tree/master/analytics-zoo)

* Create a folder in CDSW under your project and import the image-similarity.py and make sure you have the spark-defaults.conf file to run the engine with necessary configurations for training the model.

* Add some of the parameter settings as you need in the spark-defaults.conf to start your workbench.

###### spark-defaults.conf
    spark.executor.memory                               22g
    spark.executor.cores                                4
    spark.driver.memory                                 22g
    spark.driver.cores                                  4
    spark.executor.instances                            4

The dataset we used is movielens-1M that can be downloaded from the workbench terminal access menu by using the following command:
```
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
```

After downloading the dataset here is the command to extract the file:
```
unzip ml-1m.zip
```

This contains 1 million ratings from 6000 users on 4000 movies. There are 5 levels of rating. We will try classifying each (user,movie) pair into 5 classes and evaluate the effect of algortithms using Mean Absolute Error.  





#### These below steps are not mandatory unless you have enough space to download and store dataset onto the local.

After downloading the dataset to your local, here is a hadoop command that copies the local data to HDFS, as we suggest to store and read large datasets from HDFS.

> hdfs dfs -copyFromLocal <local_source_path> <hadoop_dest_path>


Here are some sample steps that shows how to mention the hadoop path inside the code to read model and dataset from HDFS;

> movielens_data = movielens.get_id_ratings("hdfs:///user/leelau/zoo/recommendation-ncf/*") (Reading the dataset from HDFS)
