# Computer Network Course Project

### Predicting Internet Path Dynamics and Performance with Machine Learning

This project is based on the Workshop paper 

[**NETPerfTrace - Predicting Internet Path Dynamics and Performance with Machine Learning**](http://orbi.ulg.ac.be/handle/2268/211667)

and the authors' release of [**NETPerfTrace**](https://github.com/SAWassermann/NETPerfTrace)

Our report is [here](https://github.com/wrystal/cn/blob/master/paper.pdf)

## 简介

### 文件路径

代码在`/code`路径下

数据存放在`/data`路径下, 由于github有100M的文件大小限制, 因此压缩在根目录的`raw_data.zip`文件中, 请解压后的` raw_data.csv`文件放在`/data`路径下



### 数据描述

在`raw_data.csv`文件中, 共有75列, 其中有69列是用来训练和预测的feature, 而其余的6列的名称和含义分别为

- **sampleResLife**: 文章中第1个预测任务需要预测的值
- **nbRouteChangesInNextSlot**: 文章中第2个预测任务需要预测的值
- **nextMinRTT, nextavgRTT, nextMaxRTT, nextMdevRTT**: 这四个值中的**nextavgRTT**是文章中第3个预测任务需要预测的值

需要注意的是以上6列数据不可用于训练


### 项目目标

每一行数据就是一组用于预测的原始数据, 我们使用其中的69个feature来预测sampleResLife, nbRouteChangesInNextSlot 和 nextavgRTT这3个值. 


### implement dependency

- python 2.7
- numpy
- sk-learn
- xgboost
- pandas
- keras


## Group member

- Zhenghui Wang

- Yuheng Zhi

- Hao Wang

- Shukai Liu

  

