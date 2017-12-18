import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def output_real_and_predict_data(y_real,y_pred,path,filename):
    """
    rt
    :param y_real: list
    :param y_pred: list
    :param path:
    :param filename:
    :return: None
    """
    with open(path+filename+'_real.csv','w') as fout:
        for val in y_real:
            fout.write(str(val)+',\n')
    with open(path + filename + '_predict.csv', 'w') as fout:
        for val in y_pred:
            fout.write(str(val) + ',\n')

    draw(path+filename+'_real.csv',path + filename + '_predict.csv',filename,path+filename+'.jpg')

    print "finish"+path+filename


def draw(realfilePath, predictfilePath, graphTitle, outputPath):
    tableReal = []
    for line in csv.reader(open(realfilePath)):
        tableReal.append(line[0])

    tablePred = []
    for line in csv.reader(open(predictfilePath)):
        tablePred.append(line[0])

    relativeError = []
    for i in range(len(tableReal)):
        if float(tableReal[i])!=0:
            relativeError.append(100 * abs(float(tablePred[i]) - float(tableReal[i])) / float(tableReal[i]))

    relativeError = sorted(relativeError, key=float)

    sampleRate = []
    length = len(relativeError)
    for i in range(length):
        sampleRate.append(100 * (i + 1) / length)

    _draw(relativeError, sampleRate, graphTitle, outputPath)


def _draw(xData, yData, title, outputPath):
    x = np.array(xData)
    y = np.array(yData)
    fig = plt.figure(1, (8, 5))
    ay = fig.add_subplot(1, 1, 1)
    plt.plot(x, y, linestyle='-', color='blue', marker='<')
    plt.grid(True, color='k', linestyle='--', linewidth='1')
    plt.title(title)
    plt.xlabel('relative prediction errors (%)')
    plt.ylabel('samples (%)')

    fmt = '%.2f%%'
    ticks = mtick.FormatStrFormatter(fmt)
    ay.yaxis.set_major_formatter(ticks)
    ay.xaxis.set_major_formatter(ticks)

    axes = plt.gca()
    axes.set_xlim([0, 100])
    axes.set_ylim([0, 100])

    # plt.show()
    plt.savefig(outputPath)
    plt.close()




def get_pd_from_path(path):
    """

    :param path: the path that contains many csv files
    :return: a pd.df that concat all the data
    """
    list=[]
    in_file_list=os.listdir(path)
    for in_file in  in_file_list:
        # print in_file
        try:
            # tmp=
            # print tmp
            list.append(pd.read_csv(path+in_file))
        except:
            pass
    # print list
    big_pd = pd.concat(list, axis=0, join='outer', join_axes=None, ignore_index=False,
                    keys=None, levels=None, names=None, verify_integrity=False,
                    copy=True)
    return big_pd

def get_task_data_from_df(df,task):
    """
    :param df: the input data, could be train df or test df
    :param task: the task that we want to conduct
    :return: X, y
    """
    if task ==1:
        X=df.drop(
            ['sampleResLife'],
            axis=1)
        y = df.sampleResLife

    elif task==2:
        X = df.drop(
            ['nbRouteChangesInNextSlot'],
            axis=1)
        y = df.nbRouteChangesInNextSlot
    elif task==3:
        X = df.drop(
            [ 'nextMinRTT', 'nextavgRTT', 'nextMaxRTT', 'nextMdevRTT'],
            axis=1)
        y = df.nextavgRTT
    else:
        exit(-1)

    return X,y


def RF_task(train_path,test_path,task,task_name,n_estimators=10):
    """
    train a rf model
    :param train_path: the path of dir that contains the .csv for each path, trianing data
    :param test_path: the path of dir that contains the .csv for each path, testing data
    :param task: int, 1,2,3
    :param task_name: the name of this experiment, should include the details
    :param n_estimators: number of trees
    :return: none
    """
    trainDF=get_pd_from_path(train_path)
    testDF = get_pd_from_path(test_path)
    train_X , train_y = get_task_data_from_df(trainDF,task)
    test_X, test_y = get_task_data_from_df(testDF, task)
    rf_regressor = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1)
    rf_regressor.fit(train_X, train_y)
    y_hat = rf_regressor.predict(test_X)
    mse1 = mean_squared_error(y_hat, test_y)

    output_real_and_predict_data(test_y, y_hat, '../data/temp_result/', task_name)


def XGB_task(train_path,test_path,task,task_name,n_estimators=10):
    """
    train a xgb model
    :param train_path: the path of dir that contains the .csv for each path, trianing data
    :param test_path: the path of dir that contains the .csv for each path, testing data
    :param task: int, 1,2,3
    :param task_name: the name of this experiment, should include the details
    :param n_estimators: number of trees
    :return: none
    """
    trainDF=get_pd_from_path(train_path)
    testDF = get_pd_from_path(test_path)
    train_X , train_y = get_task_data_from_df(trainDF,task)
    test_X, test_y = get_task_data_from_df(testDF, task)

    dtrain = xgb.DMatrix(data=train_X, label=train_y)
    dtest= xgb.DMatrix(data=test_X, label=test_y)

    if task==1:
        param = {
                "reg_alpha": 0.0,
                "colsample_bytree": 0.9,
                "scale_pos_weight": 1,
                "learning_rate": 0.05,
                "nthread": 16,
                "min_child_weight": 5,
                "n_estimators": 100,
                "subsample": 0.7,
                "reg_lambda": 0.0032,
                "seed": 27,
                "objective":'reg:linear',
                "max_depth": 7,
                "gamma": 0.0,
                }
    elif task==2:
        param = {
            "reg_alpha": 0.0,
            "colsample_bytree": 0.9,
            "scale_pos_weight": 1,
            "learning_rate": 0.05,
            "nthread": 16,
            "min_child_weight": 5,
            "n_estimators": 100,
            "subsample": 0.7,
            "reg_lambda": 0.0032,
            "seed": 27,
            "objective": 'reg:linear',
            "max_depth": 7,
            "gamma": 0.0,
        }
    elif task==3:
        param = {
            # "reg_alpha": 0.0,
            # "colsample_bytree": 0.9,
            # "scale_pos_weight": 1,
            # "learning_rate": 0.05,
            "nthread": 16,
            # "min_child_weight": 5,
            # "n_estimators": 100,
            # "subsample": 0.7,
            # "reg_lambda": 0.0032,
            # "seed": 27,
            "objective": 'reg:linear',
            # "max_depth": 7,
            # "gamma": 0.0,
        }

    bst1 = xgb.train(param, dtrain, n_estimators)
    y_hat = bst1.predict(dtest)
    output_real_and_predict_data(test_y, y_hat, '../data/temp_result/', task_name)


def get_lstm_data_from_df(train_df,test_df,task,time_steps,min_samples_for_path=5):
    """
    convert the df data structure into lstm suitable form, vector as input

    the current process method: use the previous k trace-route as previous k time-steps,
    and also the label of the trace-route which does not belong to the current route.
    The mark to distinguish different route is `sampleRouteAge`, because every route
    begins with a `sampleRouteAge`=0.
    But one thing left is what should be fed into the `label` feature of the trace-route
    belongs to current route?

    :param train_df: the input data, train df
    :param test_df: the input data, test df
    :param task: the task that we want to conduct
    :param time_steps: the time_steps of a sequence
    :param min_samples_for_path: min_samples_for_path, int, default = 5
    :return: train_lstm_X, train_lstm_y, test_lstm_X, test_lstm_y, feature_dimension

    numpy arrays is ok
    train_lstm_X.shape=[samples, time steps, features]
    train_lstm_y.shape=[samples, time steps]
    test_lstm_X.shape=[samples, time steps, features]
    test_lstm_y.shape=[samples, time steps]
    feature_dimension = X.shape[2]

    """
    # fixme: what's y.shape?
    train_X, train_y = get_task_data_from_df(train_df,task)
    test_X, test_y = get_task_data_from_df(test_df, task)

    train_len = train_X.shape[0]
    test_len = test_X.shape[0]
    min_samples_for_path = 5


    if train_len < time_steps+min_samples_for_path:
        return None, None, None, None, None






