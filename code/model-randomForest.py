import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

max_depth = 10
n_estimators = 10
# train task 1
rf_regressor1 = RandomForestRegressor(n_estimators=n_estimators, n_jobs = -1)
rf_regressor1.fit(X_train, y1_train)
y1_hat = rf_regressor1.predict(X_val)
mse1 = mean_squared_error(y1_hat, y1_val)
print mse1

# train task 1
rf_regressor2 = RandomForestRegressor(n_estimators=n_estimators, n_jobs = -1)
rf_regressor2.fit(X_train, y2_train)
y2_hat = rf_regressor1.predict(X_val)
mse2 = mean_squared_error(y2_hat, y2_val)
print mse1

# train task 1
rf_regressor3 = RandomForestRegressor(n_estimators=n_estimators, n_jobs = -1)
rf_regressor3.fit(X_train, y3_train)
y3_hat = rf_regressor3.predict(X_val)
mse1 = mean_squared_error(y3_hat, y3_val)
print mse1