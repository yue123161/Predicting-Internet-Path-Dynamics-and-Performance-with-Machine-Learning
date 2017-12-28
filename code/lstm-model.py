
import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from math import sin, exp, pi, cos, sqrt
from utils import *
import tensorflow as tf
config  = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

dropout_rate=0.5

epochs=100
batch_size=8192

# reshape input to be [samples, time steps, features]
with open('../data/temp_data/lstm-data/tmp.pkl', 'rb') as fin:  # interface between whole model and training data
    trainX, trainY, testX, testY, feature_len = pkl.load(fin)
train_samples=trainX.shape[0]
test_samples=testX.shape[0]
trainX=np.reshape(trainX,(-1,feature_len))
testX=np.reshape(testX,(-1,feature_len))


#scale the data
trainAll=np.concatenate([trainX, testX],0)
scalerX = MinMaxScaler(feature_range=(0, 1))
scalerX.fit(trainAll)
trainX=scalerX.transform(trainX)
testX=scalerX.transform(testX)
trainX=np.reshape(trainX,(train_samples,-1,feature_len))
testX=np.reshape(testX,(test_samples,-1,feature_len))



trainYAll=np.concatenate([trainY, testY],0)
scalerY = MinMaxScaler(feature_range=(0, 1))
scalerY.fit(trainYAll)
trainY=scalerY.transform(trainY)
testY=scalerY.transform(testY)


#MODLE
model = Sequential()
model.add(LSTM(128, input_shape=(5, feature_len)))
model.add(Dropout(dropout_rate))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#####################################
print trainX.shape
print trainY.shape

#####################################

model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scalerY.inverse_transform(trainPredict)
trainY = scalerY.inverse_transform(trainY)
testPredict = scalerY.inverse_transform(testPredict)
testY = scalerY.inverse_transform(testY)

print testPredict.shape
print testY.shape

trainScore = sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# Train Score: 18214.22 RMSE
# Test Score: 14888.92 RMSE

