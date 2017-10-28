import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
train = pd.read_csv('../data/raw_data.csv')

val = train.iloc[0:10000]
train = train.iloc[10000:]


X_train = train.drop(['sampleResLife', 'nbRouteChangesInNextSlot','nextMinRTT','nextavgRTT','nextMaxRTT','nextMdevRTT'],axis=1)
y1_train = train.sampleResLife
y2_train = train.nbRouteChangesInNextSlot
y3_train = train.nextavgRTT

X_val = val.drop(['sampleResLife', 'nbRouteChangesInNextSlot','nextMinRTT','nextavgRTT','nextMaxRTT','nextMdevRTT'],axis=1)
y1_val = val.sampleResLife
y2_val = val.nbRouteChangesInNextSlot
y3_val = val.nextavgRTT

print y3_val

dtrain = xgb.DMatrix(data=X_train,label=y1_train)
dtest=xgb.DMatrix(data=X_val,label=y1_val)
param = {
    'max_depth':10,
    'eta':0.3,
    'silent':1,
    'objective':'reg:linear'
        }

watchlist= [(dtest,'eval'), (dtrain,'train')]
num_round = 20
bst = xgb.train(param, dtrain, num_round, watchlist)
