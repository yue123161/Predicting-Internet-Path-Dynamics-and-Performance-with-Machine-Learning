import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from math import sin, exp, pi, cos
from utils import *


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


length = 500
data_y = np.zeros([length])
data_y[0] = 1
data_y[1] = 1
for i in range(length):
    data_y[i] = 0.3 * data_y[i - 1] + 0.6 * data_y[i - 2] + 0.6 * sin(
        pi * cos(2 * pi * i / 250)) + 0.9 * exp(-data_y[i - 1] ** 2)

data_y = np.transpose([data_y])
scaler = MinMaxScaler(feature_range=(0, 1))
data_y = scaler.fit_transform(data_y)

look_back = 1
trainX, trainY = create_dataset(data_y, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

train_df = pd.read_csv("/home/fw/Documents/CN-group-project/data/temp_data/1/task3/task3_70+30/70/1.csv")
test_df= pd.read_csv("/home/fw/Documents/CN-group-project/data/temp_data/1/task3/task3_70+30/30/1.csv")

trainX,trainY,testX,testY,feature_len=get_lstm_data_from_df(train_df=train_df,test_df=test_df,task=3,time_steps=5,min_samples_for_path=5)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], -1, trainX.shape[1]))

print trainX

model = Sequential()
model.add(LSTM(128, input_shape=(feature_len, 5)))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
#####################################
print trainX.shape
print trainY.shape

#####################################

model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# trainPredict = model.predict(trainX)
# trainPredict = scaler.inverse_transform(trainPredict)
#
# trainPredictPlot = np.empty_like(data_y)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#
# plt.plot(scaler.inverse_transform(data_y))
# plt.plot(trainPredictPlot)
# plt.show()


