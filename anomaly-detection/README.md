## Anomaly Detection
This is a simple example of unsupervised anomaly detection using Analytics Zoo Keras-Style API. We use RNN to predict following data values based on previous sequence (in order) and measure the distance between predicted values and actual values. If the distance is above some threshold, we report those values as anomaly.

###### Environment:
* Python 2.7/3.5/3.6 (pandas 0.22.0)
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)

###### Run with CDSW:
* Build the Analytics Zoo Engine using the docker file from this [Build Zoo Engine](https://github.com/dell-ai-engineering/bigdlengine4cdsw/tree/master/analytics-zoo)
* Create a folder in CDSW under your project and import the anomaly-detection.py and make sure you have the spark-defaults.conf file to run the engine with necessary configurations for training the model.
* Add some of the parameter settings as you need in the spark-defaults.conf to start your workbench

###### spark-default.conf
    spark.executor.memory                               2g
    spark.executor.cores                                4
    spark.driver.memory                                 2g
    spark.driver.cores                                  4
    spark.executor.instances                            4


We used one of the dataset in Numenta Anomaly Benchmark (NAB) for demo, i.e. NYC taxi passengers dataset, which contains 10320 records, each indicating the total number of taxi passengers in NYC at a corresonponding time spot. We use RNN to learn from 50 previous values, and predict just the 1 next value. The data points whose actual values are distant from predicted values are considered anomalies (distance threshold can be adjusted as needed).


For downloading the dataset use this command from the CDS workbench terminal access menu:
```
wget https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv
```

