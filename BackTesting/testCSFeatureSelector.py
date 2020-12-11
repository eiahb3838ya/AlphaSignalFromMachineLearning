import numpy as np
from sklearn.datasets import load_boston

def create_dataset():
    iris = load_iris()
    X = iris.data

    return X


if __name__ == "__main__":

    from CrossSectionalFeatureSelectorBase import CrossSectionalFeatureSelectorBase

    # from *your python file* import *your selector*
    class MySelector(CrossSectionalFeatureSelectorBase):
        def __init__(self, **kwargs):
            super(MyModel, self).__init__(**kwargs)

        def fit(self, X_train):
        # fit the model with the input data
        # self.model.fit(X,y)
            pass

        def transform(self, X):
            # the one method that to be called to perform prediction
            # return(self.model.predict(X))
            pass
        
        def fit_transform(self, X):
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
        
        def getSelector(self):
            # return the hyperparameter of the model
            # maybe from another file json-like or another module
            # for the cv cases
            # do some how cv or things to decide the hyperparameter in this
            
            # if self.parameter == {}:
            #     do something
            # else:
            #     return(self.parameter)
            pass


    # fit and predict
    selector = MySelector()

    print("+++++++  Before training +++++++")
    print(selector.getSelector())
    print(selector.getPara())

    X = create_dataset()
    
    print("+++++++ Training +++++++")
    selector.fit(X)

    print("+++++++  After training +++++++")
    print(selector.getSelector())
    print(selector.getPara())

    print("+++++++ Predicting +++++++")
    pred = selector.transform(X)

    # fit_transform directly
    selector = MySelector()

    print("+++++++  Before training +++++++")
    print(selector.getSelector())
    print(selector.getPara())

    X = create_dataset()
    
    print("+++++++ Fit_transform +++++++")
    pred = selector.fit_transform(X)

    print("+++++++  After fit_transform +++++++")
    print(selector.getSelector())
    print(selector.getPara())


    
    
