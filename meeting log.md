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

**讨论**:

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

## 2020-12-23

**讨论**:

- 上周任务完成情况
  - train_test_slice的具体参数设置，底层切片方法的需求
  - preprocessing的具体处理，mask的实现与相应类MA设计，preprocessing所涉方法维度选择
  - generate_signals基本框架思路
  - smoothing方法的generalize,logger类的使用
 
 **任务**：
 
 - 各自完善代码
 - 底层切片方法的完善，策略基础类的架构（胡逸凡）
 - 寻找表现较好的因子，具体说明为什么能有较好表现及后续数据的可得
    - 技术指标：动量及反转相关（国欣然）
    - 基本面指标：财务数据相关（叶文轩）
    - 投资者行为相关（叶梦婕）
    - 高频因子低频化（薛岚天）
