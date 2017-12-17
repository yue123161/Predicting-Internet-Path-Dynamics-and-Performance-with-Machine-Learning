# Computer Network Course Project

### Predicting Internet Path Dynamics and Performance with Machine Learning

This project is based on the Workshop paper 

[**NETPerfTrace - Predicting Internet Path Dynamics and Performance with Machine Learning**](http://orbi.ulg.ac.be/handle/2268/211667)

and the authors' release of [**NETPerfTrace**](https://github.com/SAWassermann/NETPerfTrace)

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

## 进度

- 10/28 初步的代码, 使用random forest, xgboost 进行baseline 实验



## TODO

- ~~仔细阅读论文, 对论文中的细节想清楚, 不清楚的地方在群里讨论 -> Yuheng, Hao, Shukai~~
- ~~绘制 sample rate & relative prediction error 图 -> who?~~
  - ~~输入是两个`.csv`文件, 一个是预测值, 一个是真实值~~
- 进行数据的整理, 按照每个path进行时序的梳理, 增加一些feature, 以方便下一步的时序建模的工作 ->who?
  - 根据论文中对数据的描述以及对数据文件的查看, 目前的数据*应该*(不完全确定)是按照时序排列下来的, 但是还不能用于实际的序列建模
  - 对**sampleResLife**预测问题, 可能需要添加一维feature是`last_duration`, 就是上一次的完整duration的长度,
  - 对**nbRouteChangesInNextSlot**和**nextavgRTT**预测问题, 还需要进一步思考.


## TODO LIST

### Task For 王浩

1. ~~利用前k%的数据，计算feature，然后后1-k%利用这些feature来预测寿命~~

   ~~前k%一个path只算出一个avg，后面不更新~~

2. ~~前K%，要算出num_route_k 个avg~~

3. ~~前K%统一计算统计值，后面route更新~~
4. ~~取一个timeslot，窗口~~
5. 曲线下面积

### Task For 植禹衡

1. ~~raw data，看看traceroute能提供哪些feature，如果没有额外的数据，就不做更多东西了~~
2. 分析数据的周期性，可视化出来
3. ~~看看能不能用一个月的数据~~
4.~~LSTM 的数据处理：有两种想法，要不要喂label进去：目前就只按照route来算。
   1. 但是一个时间窗口
   2. route duration需要按照route来
   3. rtt可以直接连着来~~ `这部分王政晖来做`
5. 论文数据分析部分

### task for 王政晖
0. 普通模型实验
1. LSTM 数据处理
2. LSTM 实验

### 20171214总结

1. LSTM的task1和task3直接使用每行的`traceroute`的数据即可
2. 对于用LSTM时的feature输入问题, 两种处理方式都尝试一下 (i) 只输入一个scalar (ii) 还要输入其他的path的统计值feature
3. 对于task1需要预测的值, 分成两种, 一个(i)直接预测剩余时间, 另一个(ii)预测这次route的duration
4. 找route之间的潜在相似性的论文, 或者问老师
5. 
### 一些需要做的实验
#### 基础模型部分 random forest & XGBoost
一下三种数据处理方式的task1的label一种是剩余时间,一种是这次的route duration
1. **k&fix**: 利用前k%的数据，计算feature，然后后1-k%利用这些feature来预测寿命,前k%一个path只算出一个avg，后面不更新
2. **k&update**: 前K%统一计算统计值，后面route更新
3. **timeslot&updata**: 取一个timeslot，窗口

#### LSTM部分
不按照route来, 还是按照每行的`traceroute`的数据即可. task1同样是预测剩余寿命/duration两种都做
1. 输入数据只有当前的值, 比如task1就是当前的寿命, 然后预测剩余寿命/duration
2. 输入数据是当前值和这个path的一些统计值,作为一个vector输入. 然后预测剩余寿命/duration


## Group member

- Zhenghui Wang

- Yuheng Zhi

- Hao Wang

- Shukai Liu

  

