# Analytics Zoo Examples for Cloudera Data Science Workbench

## Overview
This respository includes a number of Analytics Zoo example models. Analytics Zoo makes it easy to build deep learning applications on Spark and BigDL, by providing an end-to-end Analytics + AI Platform. These examples are packaged for Cloudera Data Science Workbench and can be downloaded as a CDSW template.

## What is Analytics Zoo?
Analytics Zoo seamlessly unites Spark, TensorFlow, Keras and BigDL programs into an integrated pipeline; the entire pipeline can then transparently scale out to a large Hadoop/Spark cluster for distributed training or inference.

Analytics Zoo provides several built-in deep learning models that you can use for a variety of problem types such as anomaly detection, object detection, image classification, etc.

For detailed documentation, please refer to the following: [Intel Analytics-Zoo](https://analytics-zoo.github.io/0.2.0/#)

## Pre-requisites
Environment:
- See each indivdual example model for specific requirements

Run Model with CDSW:
- Build the Analytics Zoo Engine.  See detailed information [here](https://github.com/dell-ai-engineering/bigdlengine4cdsw).
- Create a folder in CDSW under your project and import the model files specified.
- Make sure the spark-defaults.conf file exists with the appropriate configuration for training the model.  Add or change parameter settings as needed.
