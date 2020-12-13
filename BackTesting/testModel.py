import numpy as np
from sklearn.datasets import load_iris, load_boston


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

    from CrossSectionalModelBase import CrossSectionalModelBase
    from CrossSectionalModelLinearSklearn import *
    paraDictOLS = {'fit_intercept':True}
    paraDictRidge = {'fit_intercept':True,'alpha':0.3}
    paraDictLasso = {'fit_intercept':True,'alpha':0.3}
    model = CrossSectionalModelOLS(paraDict = paraDictOLS)
    model = CrossSectionalModelRidge(paraDict = paraDictRidge)
    model = CrossSectionalModelLasso(paraDict = paraDictLasso)

    print("+++++++  Before training +++++++")
    print(model.getModel())
    print(model.getPara())

    X_train, y_train, X_test, y_test = create_regression_dataset()  # or create_classification_dataset()
    
    print("+++++++ Training +++++++")
    model.fit(X_train, y_train)

    print("+++++++  After training +++++++")
    print(model.getModel())
    print(model.getPara())

    print("+++++++ Predicting +++++++")
    pred = model.predict(X_test)

# =============================================================================
#     def mse(y, y_hat):
#         return np.mean((y-y_hat)**2)
#     print(mse(y_test, pred))
# =============================================================================
    print("+++++ Linear Rregression Coefficient ++++++")
    print(model.getCoef())
    print(model.getScore(y_test, y_pred=pred, scoreMethod = 'mse'))
    model.scatterPred(y_test,pred)
