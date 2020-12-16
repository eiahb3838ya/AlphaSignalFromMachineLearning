import numpy as np
from sklearn.datasets import load_boston

def create_dataset():
    boston = load_boston()
    target = np.array(boston.feature_names) == "DIS"
    X = boston.data[:, np.logical_not(target)]
    y = boston.data[:, target].squeeze()

    return X, y


if __name__ == "__main__":

    
    from CrossSectionalFeatureSelectorLinear import CrossSectionalFeatureSelectionLasso
    

    # fit and predict
    paraDictLasso = {'fit_intercept':True,'alpha':0.3}
    selector = CrossSectionalFeatureSelectionLasso(paraDict = paraDictLasso)

    print("+++++++  Before training +++++++")
    print(selector.getSelector())
    print(selector.getPara())

    X,y = create_dataset()
    
    print("+++++++ Training +++++++")
    selector.fit(X,y)

    print("+++++++  After training +++++++")
    print(selector.getSelector())
    print(selector.getPara())

    print("+++++++ Predicting +++++++")
    pred = selector.transform(X)

    # fit_transform directly
    selector = CrossSectionalFeatureSelectionLasso(paraDict = paraDictLasso)

    print("+++++++  Before training +++++++")
    print(selector.getSelector())
    print(selector.getPara())

    X = create_dataset()
    
    print("+++++++ Fit_transform +++++++")
    pred = selector.fit_transform(X,y)

    print("+++++++  After fit_transform +++++++")
    print(selector.getSelector())
    print(selector.getPara())


    
    
