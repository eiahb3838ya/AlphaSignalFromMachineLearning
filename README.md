# AlphaSignalFromMachineLearning

## 简介
在因子研究的过程中，有许多重复且既定的过程，我们将因子信号与策略回测分成两个不同的 模块，将常见的研究流程实现出来，并且设计可以替换模型的接口，让因子合成的测试可以快速被实现。
在信号端，我们可以用不同的预测模型例如线性模型、树模型、或神经网络模型来替换核心的 cross sectional predict 模型，来达到用一样流程比较不同模型的测试方式。若要达到更高的自由度，使用者可以透过继承多态的方式，对每期的预测流程进行定制。若想要在流程上不使用既定的预测区间设定(例如在每天的收盘前 30 分钟 或是更复杂的预测时间段)则可以继承并重写整个类。这样的设计是为了解决我们在经验上发现我们想要简便的替换常见的实验对象，并且又有高自由度可以随时改变框架的需求。
在策略端我们接收 signal 模块传出来的信号，并使用是当的下单策略，目前实现简单的 Longshort 并算 PnL 来查看信号的效果。

## 模块设计

本次project主要包含以下模块：

- Signal模块：接收上一模块输入的factor，进行一系列操作后，产生Signal，进入Strategy模块
- Strategy模块：接收上一模块产生的Signal后，进一步处理，得到股票配置的策略

### 1.Signal模块

本模块主要包括两大功能：generateSignals和smoothing
即产生 raw signal，signal 平滑处理

#### generateSignals

是一个iteration，每个迭代过程会实现以下几个功能：

- slicefactor：对于Factor模块中输进来的factor进行切片处理
- preprocessing：对于上一步的factor进行预处理
  - 遮罩：剔除上市未满一年、ST、涨跌停的股票等
  - 极值处理、中性化与标准化处理等

- getSignal：使用多种模型（线性、树等）对于处理后的factor进行分析，从而产生signal

最终generateSignals会输出包含这段时间每天signal的一张表

#### smoothing

对于generateSignals产出的raw signals进行平滑处理


### 2.Strategy模块

接收Signal模块输出的信号，并利用不同的方法，产出股票配置的策略：
- long-short
- long only
- 利用cvxopt进行各种组合优化

## 文件夹说明
### \director
存放主要用来 call signal 模块与 strategy 模块的 director 类，用来描述整个回测的流程，可以分为单因子回测与多因子回测。
### \get data
存放用来将数据导入的功能函数 loadData，与其他用来从数据库提取数据的脚本。
### \signal
主要的 signal 模块
### \Tool 
存放通用类 Factor、GeneralData 与 globalVars 模块用于定义 global variables 的 trick
### \report 
回测结果

## 使用说明
