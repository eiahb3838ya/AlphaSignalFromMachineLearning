
import numpy as np
from sklearn.datasets import load_iris, load_boston
import matplotlib.pyplot as plt


def create_classification_dataset():
    # features: 4 numeric, predictive attributes
    # target: 3 int classes (0, 1, 2)

    iris = load_iris()
    X = iris.data
    y = iris.target

    n_sample = len(X)

    X_train = X[:int(.7 * n_sample)]
    y_train = y[:int(.7 * n_sample)]
    X_test = X[int(.7 * n_sample):]
    y_test = y[int(.7 * n_sample):]

    return X_train, y_train, X_test, y_test
    

def create_regression_dataset():
    # features: 13 numeric/categorical predictive attributes
    # target: 3 int(0, 1, 2)
    boston = load_boston()
    target = np.array(boston.feature_names) == "DIS"
    X = boston.data[:, np.logical_not(target)]
    y = boston.data[:, target].squeeze()

    n_sample = len(X)

    X_train = X[:int(.7 * n_sample)]
    y_train = y[:int(.7 * n_sample)]
    X_test = X[int(.7 * n_sample):]
    y_test = y[int(.7 * n_sample):]

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
# =============================================================================
#     import sys
#     import os
#     sys.path.append(os.path.abspath('.') + '\..')
#     from CrossSectionalModel.CrossSectionalModelLinearSklearn import CrossSectionalModelOLS, CrossSectionalModelRidge, CrossSectionalModelLasso
# =============================================================================
    from BackTesting.Signal.CrossSectionalModels.CrossSectionalModel.CrossSectionalModelLinearSklearn \
        import CrossSectionalModelOLS, CrossSectionalModelRidge, CrossSectionalModelLasso
    
    paraDictOLS = {'fit_intercept':True}
    paraDictRidge = {'fit_intercept':True,'alpha':0.3}
    paraDictLasso = {'fit_intercept':True,'alpha':1}
    # paraDictLasso2 = {'fit_intercept':True,'alpha':0.2}
    
    paraGridRidge = {'alpha':[x for x in np.arange(0.1,2,0.2)]}
    paraGridLasso = {'alpha':[x for x in np.arange(0.1,2,0.2)]}
    
    modelOLS = CrossSectionalModelOLS(paraDict = paraDictOLS)
    modelRidge = CrossSectionalModelRidge(paraDict = paraDictRidge)
    modelLasso = CrossSectionalModelLasso(paraDict = paraDictLasso)
    
    modelRidgeJson = CrossSectionalModelRidge(jsonPath = 'paraDictRidge.json')
    modelLassoJson = CrossSectionalModelLasso(jsonPath = 'paraDictLasso.json')
    # modelLassoJson2 = CrossSectionalModelLasso(jsonPath = 'paraDictLasso2.json',json_first = False)
    
    modelRidgeCV = CrossSectionalModelRidge(paraGrid = paraGridRidge)
    
    # ????用完modelLassoJson之后 paraDict变了。。。
    # paraDict.update()的问题？？？
    # 先用paraDict.update()的话，会把paraDict存在那个函数里面的感觉,id的问题
    # 直接用self.parameter
    
    # modelLassoJson = CrossSectionalModelLasso(jsonPath = 'paraDictLasso.json',
    #                                           paraDict = paraDictLasso2)
    modelLassoCV = CrossSectionalModelLasso(paraGrid = paraGridLasso)
    
    
    # 用上面这些做测试
    # modelLassoJson2 = CrossSectionalModelLasso(jsonPath = 'paraDictLasso2.json',json_first = False)
    # modelLassoCV2 = CrossSectionalModelLasso(paraGrid = paraGridLasso)
    model = modelLassoCV
    print("+++++++  Before training +++++++")
    print(model.get_model())
    print(model.get_para())

    X_train, y_train, X_test, y_test = create_regression_dataset()  # or create_classification_dataset()
    
    print("+++++++ Training +++++++")
    model.fit(X_train, y_train)

    print("+++++++  After training +++++++")
    print(model.get_model())
    print(model.get_para(verbal = True))

    print("+++++++ Predicting +++++++")
    pred = model.predict(X_test)

# =============================================================================
#     def mse(y, y_hat):
#         return np.mean((y-y_hat)**2)
#     print(mse(y_test, pred))
# =============================================================================
    print("+++++ Rregression Coefficient ++++++")
    print(model.get_coef())
    print(model.get_score(y_test, y_pred=pred, scoreMethod = 'r2'))
    
    plt.scatter(y_test,pred)
    plt.title('y_pred vs y_real')
    plt.xlabel('y_real')
    plt.ylabel('y_pred')
    plt.show()











