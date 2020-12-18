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

    # from *your python file* import *your model*
    class MyModel(CrossSectionalModelBase):
        def __init__(self, **kwargs):
            super(MyModel, self).__init__(**kwargs)
    
        def fit(self, X_train, y_train):
            # fit the model with the input data
            # self.model.fit(X,y)
            pass
        
        def predict(self, X):
            # the one method that to be called to perform prediction
            # return(self.model.predict(X))
            pass
        
        def getPara(self):
            # return the hyperparameter of the model
            # maybe from another file json-like or another module
            # for the cv cases
            # do some how cv or things to decide the hyperparameter in this
            
            # if self.parameter == {}:
            #     do something
            # else:
            #     return(self.parameter)
            pass
        
        def getModel(self):
            # return the model 
            pass


    model = MyModel()

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

    def mse(y, y_hat):
        return np.mean((y-y_hat)**2)
    print(mse(y_test, pred))