# AlphaSignalFromMachineLearning

## 模块设计

本次project主要包含以下模块：

- Signal模块：接收上一模块输入的factor，进行一系列操作后，产生Signal，进入Strategy模块
- Strategy模块：接收上一模块产生的Signal后，进一步处理，得到股票配置的策略

## Signal模块

本模块主要包括两大功能：generateSignals和smoothing
即产生 raw signal，signal 平滑处理

### generateSignals

是一个iteration，每个迭代过程会实现以下几个功能：

- slicefactor：对于Factor模块中输进来的factor进行切片处理
- preprocessing：对于上一步的factor进行预处理
  - 遮罩：剔除上市未满一年、ST、涨跌停的股票等
  - 极值处理、中性化与标准化处理等

- getSignal：使用多种模型（线性、树等）对于处理后的factor进行分析，从而产生signal

最终generateSignals会输出包含这段时间每天signal的一张表

### smoothing

对于generateSignals产出的raw signals进行平滑处理


## Strategy模块

接收Signal模块输出的信号，并利用不同的方法，产出股票配置的策略：

- long-short
- long only
- 利用cvxopt进行各种组合优化

