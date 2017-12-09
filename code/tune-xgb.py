# -*- coding:utf-8 -*-

from sklearn.datasets import load_svmlight_file
import argparse
import matplotlib.pyplot as plt
import os


import pandas as pd
import numpy as np
import xgboost as xgb

from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics
import matplotlib.pylab as plt
import pickle


import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
from utils import output_real_and_predict_data,draw


def mkdir(path):
    # 引入模块


    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号



    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # print path + ' 创建成功'
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print path + ' 目录已存在'
        return False

def get_data(file_path):

    data = load_svmlight_file(file_path)
    X=data[0].toarray()
    arrayX=np.array(X)

    return arrayX, data[1]



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


param_data_path='./../data/param_data.pkl'
path = './../data/parted/5_test_5_cv/'
train_x, train_y = get_data(path + 'train_1.data')
X_test, Y_test = get_data(path + 'valid_1.data')


def modelMetrics(clf, train_x, train_y, isCv=True, cv_folds=5, early_stopping_rounds=50):
    if isCv:
        xgb_param = clf.get_xgb_params()
        xgtrain = xgb.DMatrix(train_x, label=train_y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)  # 是否显示目前几颗树额
        clf.set_params(n_estimators=cvresult.shape[0])

    clf.fit(train_x, train_y, eval_metric='auc')

    # 预测
    train_predictions = clf.predict(train_x)
    train_predprob = clf.predict_proba(train_x)[:, 1]  # 1的概率

    # 打印
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y, train_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, train_predprob))

    feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature importance')
    plt.ylabel('Feature Importance Score')

def tun_parameters(train_x,train_y):
    xgb1 = XGBClassifier(learning_rate=0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,
                         colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)
    modelMetrics(xgb1,train_x,train_y)

# param init
def param_init (param_dict,param_data_path='./../data/param_data.pkl'):
    '''
    to initialize the params in the file, run this function when start your praram tuning

    :param_dict: the initial dict

    :param param_data_path: the path that the file saved

    :return: Void
    '''
    output = open(param_data_path, 'wb')
    pickle.dump(param_dict, output)
    output.close()

# get param data from file
def get_param_data(param_data_path='./../data/param_data.pkl'):
    '''
    get the param_dict from the file, run this to get the latest param when run those tun_* functions

    :param param_data_path: path

    :return: param_dict, can be used to init a Booster
    '''
    pkl_file = open(param_data_path, 'rb')
    data1 = pickle.load(pkl_file)
    pkl_file.close()

    return data1

# save param data to file
def save_param_data(param_dict,param_data_path='./../data/param_data.pkl'):
    '''
    save the best params to the param_dict file, in order to update the file. run this at the end of
    those tun_* functions

    :param param_dict: the dict you want to save, should be the best under current situation

    :param param_data_path: path

    :return: void
    '''
    output = open(param_data_path, 'wb')
    pickle.dump(param_dict, output)
    output.close()

# max_depth 和 min_child_weight 参数调优
def tun_max_depth_and_min_child_weight(max_depth_range,min_child_weight,param_data_path,train_x,train_y):
    '''
    fo tune the max_depth and min_child_weight param in xgboost.
    get the best param and save them to the file for further tuning

    :param max_depth_range: the range of max_depth you want to test

    :param min_child_weight:the range of min_child_weight you want to test

    :param_data_path: default './../data/param_data.pkl'

    :return: void
    '''
    # get the newest param first
    param_dict=get_param_data(param_data_path=param_data_path)

    print "max_depth 和 min_child_weight 参数调优"
    param_test1 = {
        'max_depth':range(max_depth_range[0],max_depth_range[1],max_depth_range[2]),
        'min_child_weight':range(min_child_weight[0],min_child_weight[1],min_child_weight[2])
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(**param_dict),
                            param_grid=param_test1,scoring='roc_auc',iid=False,cv=5)
    gsearch1.fit(X=train_x,y=train_y)

    # show the results
    for i in gsearch1.grid_scores_:
        print i
    print "best_params_ and best_score_:"
    print gsearch1.best_params_,gsearch1.best_score_

    # change the dict and save the dict to file
    param_dict['min_child_weight']=gsearch1.best_params_['min_child_weight']
    param_dict['max_depth'] = gsearch1.best_params_['max_depth']

    save_param_data(param_dict=param_dict,param_data_path=param_data_path)

# gamma参数调优
def tun_gamma(gamma_range,param_data_path,train_x,train_y):
    '''
    to tune the gamma param in xgboost.
    get the best param and save them to the file for further tuning

    :param gamma_range: the range of gamma_range you want to test

    :param param_data_path: default './../data/param_data.pkl'

    :return: void
    '''
    # get the newest param first
    param_dict = get_param_data(param_data_path=param_data_path)

    print "gamma参数调优"
    param_test1 = {
        'gamma':gamma_range
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(**param_dict),
                            param_grid=param_test1,scoring='roc_auc',iid=False,cv=5)
    gsearch1.fit(X=train_x,y=train_y)

    # show the results
    for i in gsearch1.grid_scores_:
        print i
    print "best_params_ and best_score_:"
    print gsearch1.best_params_,gsearch1.best_score_

    # change the dict and save the dict to file
    param_dict['gamma'] = gsearch1.best_params_['gamma']

    save_param_data(param_dict=param_dict, param_data_path=param_data_path)

# 调整subsample 和 colsample_bytree 参数
def tun_subsample_and_colsample_bytree(subsample_range,colsample_bytree_range,param_data_path,train_x,train_y):
    '''
    tune the subsample and colsample_bytree param in xgboost
    get the best param and save them to the file for further tuning

    :param subsample_range: the range of subsample you want to test

    :param colsample_bytree_range: the range of colsample_bytree you want to test

    :param param_data_path: default './../data/param_data.pkl'

    :return: void
    '''
    # get the newest param first
    param_dict = get_param_data(param_data_path=param_data_path)

    print "subsample 和 colsample_bytree 参数"
    param_test1 = {
        'subsample':subsample_range,
        'colsample_bytree':colsample_bytree_range
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(**param_dict),
                            param_grid=param_test1,scoring='roc_auc',iid=False,cv=5)
    gsearch1.fit(X=train_x,y=train_y)

    # show the results
    for i in gsearch1.grid_scores_:
        print i
    print "best_params_ and best_score_:"
    print gsearch1.best_params_,gsearch1.best_score_

    # change some param and return
    param_dict['subsample'] = gsearch1.best_params_['subsample']
    param_dict['colsample_bytree'] = gsearch1.best_params_['colsample_bytree']

    save_param_data(param_dict=param_dict, param_data_path=param_data_path)

# 正则化参数reg_alpha调优
def tun_reg_alpha(reg_alpha_range,param_data_path,train_x,train_y):
    '''
    tune the reg_alpha param in xgboost
    get the best param and save them to the file for further tuning

    :param reg_alpha_range: the range of reg_alpha you want to test

    :param param_data_path: default './../data/param_data.pkl'

    :return: void
    '''
    # get the newest param first
    param_dict = get_param_data(param_data_path=param_data_path)

    print "正则化参数reg_alpha调优"
    param_test1 = {
        'reg_alpha':reg_alpha_range
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(**param_dict),
                            param_grid=param_test1,scoring='roc_auc',iid=False,cv=5)
    gsearch1.fit(X=train_x,y=train_y)

    # show the results
    for i in gsearch1.grid_scores_:
        print i
    print "best_params_ and best_score_:"
    print gsearch1.best_params_,gsearch1.best_score_

    # change some param and return
    param_dict['reg_alpha'] = gsearch1.best_params_['reg_alpha']

    save_param_data(param_dict=param_dict, param_data_path=param_data_path)

# 参数reg_lambda调优
def tun_reg_lambda(reg_lambda_range,param_data_path,train_x,train_y):
    '''
    tune the reg_lambda param in xgboost
    get the best param and save them to the file for further tuning

    :param reg_lambda_range: the range of reg_lambda you want to test

    :param param_data_path: default './../data/param_data.pkl'

    :return: void
    '''
    # get the newest param first
    param_dict = get_param_data(param_data_path=param_data_path)

    print "正则化参数reg_lambda调优"
    param_test1 = {
        'reg_lambda':reg_lambda_range
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(**param_dict),
                            param_grid=param_test1,scoring='roc_auc',iid=False,cv=5)
    gsearch1.fit(X=train_x,y=train_y)

    # show the results
    for i in gsearch1.grid_scores_:
        print i
    print "best_params_ and best_score_:"
    print gsearch1.best_params_,gsearch1.best_score_

    # change some param and return
    param_dict['reg_lambda'] = gsearch1.best_params_['reg_lambda']

    save_param_data(param_dict=param_dict, param_data_path=param_data_path)


# 参数learning_rate调优
def tun_learning_rate(learning_rate_range,param_data_path,train_x,train_y):
    '''
    tune the learning_rate param in xgboost
    get the best param and save them to the file for further tuning

    :param learning_rate_range: the range of learning_rate you want to test

    :param param_data_path: default './../data/param_data.pkl'

    :return: void
    '''
    # get the newest param first
    param_dict = get_param_data(param_data_path=param_data_path)

    print "参数learning_rate调优"
    param_test1 = {
        'learning_rate':learning_rate_range
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(**param_dict),
                            param_grid=param_test1,scoring='roc_auc',iid=False,cv=5)
    gsearch1.fit(X=train_x,y=train_y)

    # show the results
    for i in gsearch1.grid_scores_:
        print i
    print "best_params_ and best_score_:"
    print gsearch1.best_params_,gsearch1.best_score_

    # change some param and return
    param_dict['learning_rate'] = gsearch1.best_params_['learning_rate']

    save_param_data(param_dict=param_dict, param_data_path=param_data_path)








param_dict={'learning_rate':0.1, 'n_estimators':300, 'max_depth':5,
    'min_child_weight':1, 'gamma':0, 'subsample':0.8,'colsample_bytree':0.8,
    'objective':'binary:logistic', 'nthread':4,'scale_pos_weight':1,'seed':27
    #, 'reg_alpha': 0.005,'reg_lambda': 0.005
    }

# 1
# init the param file
param_init (param_dict=param_dict,param_data_path='./../data/param_data.pkl')


# tun_parameters(train_x=train_x,train_y=train_y)

param_dict_show=get_param_data()
print param_dict_show

# 2
# max_depth 和 min_child_weight 参数调优
tun_max_depth_and_min_child_weight(
    max_depth_range=[1,14,2],min_child_weight=[1,12,2],param_data_path='./../data/param_data.pkl',train_x=train_x,train_y=train_y)

# # 3
# # gamma参数调优
# tun_gamma(gamma_range= [i / 20.0 for i in range(0, 5)],param_data_path='./../data/param_data.pkl',train_x=train_x,train_y=train_y)


# 4
# 调整subsample 和 colsample_bytree 参数
tun_subsample_and_colsample_bytree(subsample_range=[i / 20.0 for i in range(1, 20)],
                                   colsample_bytree_range=[i / 10.0 for i in range(1, 10)],
                                   param_data_path='./../data/param_data.pkl',train_x=train_x,train_y=train_y)


# # 5
# # 正则化参数reg_alpha调优
# # not useful
# tun_reg_alpha(reg_alpha_range=[i / 10000.0 for i in range(0, 10)],param_data_path='./../data/param_data.pkl',train_x=train_x,train_y=train_y)

# # 6
# # 参数reg_lambda调优
# # not useful, useful actually, but use it later after the learning rate
# tun_reg_lambda(reg_lambda_range=[i / 10000.0 for i in range(30, 50)],param_data_path='./../data/param_data.pkl',train_x=train_x,train_y=train_y)

# # 7
# # 参数learning_rate调优
# tun_learning_rate(learning_rate_range=[i / 400.0 for i in range(1, 40)],param_data_path='./../data/param_data.pkl',train_x=train_x,train_y=train_y)


param_dict_show=get_param_data()
print param_dict_show