import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb

# read raw data
train = pd.read_csv('../data/raw_data.csv')

val = train.iloc[0:60000]
train = train.iloc[60000:]

# build train data and labels(y1, y2, y3) for three prediction tasks in the paper
# drop the following 6 features, since they are actually `labels`
X_train = train.drop(['sampleResLife', 'nbRouteChangesInNextSlot','nextMinRTT','nextavgRTT','nextMaxRTT','nextMdevRTT'],axis=1)
y1_train = train.sampleResLife
y2_train = train.nbRouteChangesInNextSlot
y3_train = train.nextavgRTT

X_val = val.drop(['sampleResLife', 'nbRouteChangesInNextSlot','nextMinRTT','nextavgRTT','nextMaxRTT','nextMdevRTT'],axis=1)
y1_val = val.sampleResLife
y2_val = val.nbRouteChangesInNextSlot
y3_val = val.nextavgRTT

# build the DMatrix data structure for three tasks
dtrain=[None]*4
dtest=[None]*4
dtrain[1] = xgb.DMatrix(data=X_train,label=y1_train)
dtest[1] = xgb.DMatrix(data=X_val,label=y1_val)
dtrain[2] = xgb.DMatrix(data=X_train,label=y2_train)
dtest[2] = xgb.DMatrix(data=X_val,label=y2_val)
dtrain[3] = xgb.DMatrix(data=X_train,label=y3_train)
dtest[3] = xgb.DMatrix(data=X_val,label=y3_val)

# set parameters for three tasks
param = [None]*4
param[1] = {
    'max_depth':3,
    'eta':0.3,
    'silent':1,
    'objective':'reg:linear'
        }
param[2] = {
    'max_depth':20,
    'eta':0.3,
    'silent':1,
    'objective':'reg:linear'
        }
param[3] = {
    'max_depth':10,
    'eta':0.3,
    'silent':1,
    'objective':'reg:linear'
        }
watchlist = [None]*4
watchlist[1] = [(dtest[1],'eval'), (dtrain[1],'train')]
watchlist[2] = [(dtest[2],'eval'), (dtrain[2],'train')]
watchlist[3] = [(dtest[3],'eval'), (dtrain[3],'train')]

# num of trees
num_round = 200

# train for task 1
bst1 = xgb.train(param[1], dtrain[1], num_round, watchlist[1])

# train for task 2
bst2 = xgb.train(param[2], dtrain[2], num_round, watchlist[2])

# train for task 3
bst3 = xgb.train(param[3], dtrain[3], num_round, watchlist[3])

