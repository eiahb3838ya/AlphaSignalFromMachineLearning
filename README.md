# AlphaSignalFromMachineLearning

## 模块设计

从使用者视角来看，本次project主要包含以下模块：

- Factor模块：实现从raw data到factor的过程，从而进入Signal模块
- Signal模块：接收上一模块输入的factor，进行一系列操作后，产生Signal，进入Strategy模块
- Strategy模块：接收上一模块产生的Signal后，进一步处理，得到股票配置的策略

## Factor模块

这里的重点在于factorManager的设计，他需要包含以下几个功能：

- load_factor：根据使用者的需求从因子库中读取因子
- update_factor：检查截至最新日期，因子库中的因子数据是否有未更新的，如果有，则计算未更新时间段中的因子值
- save_factor：将update factor中新计算出的factor进行保存，以便下次使用

## Signal模块

本模块主要包括两大功能：generateSignals和smoothing，即产生raw signal，signal平滑处理

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

### test

利用叶文轩之前写好的内容对于单个signal进行测试，并输出测试报告

## Strategy模块

接收Signal模块输出的信号，并利用不同的方法，产出股票配置的策略：

- long-short
- long only
- 利用cvxopt进行各种组合优化

# Meeting Log

## 2020-12-02

**讨论**：

整体框架设计，各模块功能

**任务**：

学习类的继承

## 2020-12-09

**讨论**整体框架设计，各模块功能，分配任务，重点在于Signal模块的设计

**任务：**

编写Signal模块中get_signal会用到的方法：

- 线性（叶梦婕）
- 树状（国欣然）
- 网络型（薛岚天）

从以上三个方向编写CrossSectionalModels（AlphaSignalFromMachineLearning\BackTesting\Signal\CrossSectionalModels）以及FeatureSelectors（AlphaSignalFromMachineLearning\BackTesting\Signal\FeatureSelectors）

## 2020-12-16

**讨论**：

- 上周任务完成情况
  - 已编写线性（OLS、Ridge、Lasso），树状，KNN的CrossSectionalModel
  - 线性模型已测试完成
  - parameter的获取方式：jsonPath，paraDict，CrossValidation

- SignalBase的方法设计

**任务**：

- **code review**
- 编写CrossSectionalModelSklearn与ModelTest两个类，使得构造CrossSectionalModel时直接继承这两个
  - CrossSectionalModelSklearn：实现共同init的方式，fit的方式（CV或者直接fit）
  - ModelTest：对模型进行测试的工具包，例如：计算score，画图等等
- 整理文件夹，每个目录都设置README
- SignalBase的方法编写
  - generate_signals（国欣然）
  - train_test_slice（叶梦婕）
  - preprocessing（叶文轩）
  - get_signal（国欣然）
  - smoothing, logger（薛岚天）
