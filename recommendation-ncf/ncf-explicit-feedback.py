# Intialization
# import necessary libraries
from zoo.models.recommendation import UserItemFeature
from zoo.models.recommendation import NeuralCF
from zoo.common.nncontext import init_nncontext
import matplotlib
from sklearn import metrics
from operator import itemgetter
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.dataset import movielens
from bigdl.util.common import *

matplotlib.use('agg')
import matplotlib.pyplot as plt
%pylab inline

# Initilaize NN context, it will get a SparkContext with optimized configuration for BigDL performance.
sc = init_nncontext("NCF Example")

# Data Preparation
# Download and read movielens 1M data
movielens_data = movielens.get_id_ratings("hdfs:///user/leelau/zoo/recommendation-ncf/*")

# Understand the data. Each record is in format of (userid, movieid, rating_score). UserIDs range between 1 and 6040. MovieIDs range between 1 and 3952. Ratings are made on a 5-star scale (whole-star ratings only). Counts of users and movies are recorded for later use.
min_user_id = np.min(movielens_data[:,0])
max_user_id = np.max(movielens_data[:,0])
min_movie_id = np.min(movielens_data[:,1])
max_movie_id = np.max(movielens_data[:,1])
rating_labels= np.unique(movielens_data[:,2])

print(movielens_data.shape)
print(min_user_id, max_user_id, min_movie_id, max_movie_id, rating_labels)

# Transform the original data into RDD of sample. We use optimizer of BigDL directly to train the model, it requires data to be provided in format of RDD(Sample). A Sample is a BigDL data structure which can be constructed using 2 numpy arrays, feature and label respectively. The API interface is Sample.from_ndarray(feature, label)
def build_sample(user_id, item_id, rating):
    sample = Sample.from_ndarray(np.array([user_id, item_id]), np.array([rating]))
    return UserItemFeature(user_id, item_id, sample)
pairFeatureRdds = sc.parallelize(movielens_data).map(lambda x: build_sample(x[0], x[1],x[2]))
pairFeatureRdds.take(3)

# Randomly split the data into train (80%) and validation (20%)
trainPairFeatureRdds, valPairFeatureRdds = pairFeatureRdds.randomSplit([0.8, 0.2], seed= 1)
valPairFeatureRdds.cache()
train_rdd= trainPairFeatureRdds.map(lambda pair_feature: pair_feature.sample)
val_rdd= valPairFeatureRdds.map(lambda pair_feature: pair_feature.sample)
print(train_rdd.count())
train_rdd.take(3)

# Build Model
ncf = NeuralCF(user_count=max_user_id, item_count=max_movie_id, class_num=5, hidden_layers=[20, 10], include_mf = False)

# Setup the Optimizer
optimizer = Optimizer(
    model=ncf,
    training_rdd=train_rdd,
    criterion=ClassNLLCriterion(),
    end_trigger=MaxEpoch(10),
    batch_size=2800,
    optim_method=Adam(learningrate=0.001))

optimizer.set_validation(
    batch_size=2800,
    val_rdd=val_rdd,
    trigger=EveryEpoch(),
    val_method=[MAE(), Loss(ClassNLLCriterion())]
)

# Collect Logs 
import datetime as dt
def config_summaries(optimizer):
    """
    :param: optimizer: the optimizer created in start_train()
    """
    app_name = 'recommender-' + dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    # train summary
    train_summary = TrainSummary(log_dir="/home/cdsw/recommendation-ncf/log", app_name=app_name)
    train_summary.set_summary_trigger("Loss", SeveralIteration(1))
    optimizer.set_train_summary(train_summary)

    # validation summary
    val_summary = ValidationSummary(log_dir="/home/cdsw/recommendation-ncf/log", app_name=app_name)
    optimizer.set_val_summary(val_summary)

    print("saving logs to /home/cdsw/recommendation-ncf/log/" + app_name)
    return train_summary, val_summary

train_summary, val_summary = config_summaries(optimizer)

# Train the model
optimizer.optimize()
# Prediction
results = ncf.predict(val_rdd)
results.take(5)

results_class = ncf.predict_class(val_rdd)
results_class.take(5)
userItemPairPrediction = ncf.predict_user_item_pair(valPairFeatureRdds)
for result in userItemPairPrediction.take(5): 
  print(result)
userRecs = ncf.recommend_for_user(valPairFeatureRdds, 3)
for result in userRecs.take(5): 
  print(result)
itemRecs = ncf.recommend_for_item(valPairFeatureRdds, 3)
for result in itemRecs.take(5): 
  print(result)


# Evaluation
#retrieve train and validation summary object and read the loss data into ndarray's. 
loss = np.array(train_summary.read_scalar("Loss"))
val_loss  = np.array(val_summary.read_scalar("Loss"))

#plot the train and validation curves
# each event data is a tuple in form of (iteration_count, value, timestamp)
plt.plot(loss[:,0],loss[:,1],label='train loss')
plt.plot(val_loss[:,0],val_loss[:,1],label='val loss',color='green')
plt.scatter(val_loss[:,0],val_loss[:,1],color='green')
plt.legend();

#retrieve train and validation summary object and read the loss data into ndarray's. 
val_mae  = np.array(val_summary.read_scalar("MAE"))

#plot the train and validation curves
# each event data is a tuple in form of (iteration_count, value, timestamp)
plt.plot(val_mae[:,0],val_mae[:,1],label='mae')
plt.legend();

res2 = ncf.evaluate(val_rdd,2800,[MAE()])
'\n'.join([str(r) for r in res2])
