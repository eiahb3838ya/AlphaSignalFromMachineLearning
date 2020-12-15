from sklearn.tree import DecisionTreeRegressor
from CrossSectionalModelBase import CrossSectionalModelBase
import json
import sys
sys.path.append("../../")
#import matplotlib.pyplot as plt

class CrossSectionalModelDecisionTree(CrossSectionalModelBase):
    # there are two ways to get parameters
    def __init__(self, jsonPath = None, paraDict = {}):
        if jsonPath is not None:
            with open(jsonPath, 'r') as f:
                self.parameter = json.load(f)
        else:
            self.parameter = paraDict

        self.model = DecisionTreeRegressor(**self.parameter)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)


    def predict(self, X):
        return self.model.predict(X)

    def getModel(self):
        return self.model

    def gerscore(self,y_true,y_pre):
        return self.model.score(y_true, y_pre)

    def getPara(self):
        return self.parameter


if __name__ == '__main__':
    from testCSModel import create_regression_dataset

    X_train, y_train, X_test, y_test = create_regression_dataset()

    paraDicts = {'max_depth': 2}

    model = CrossSectionalModelDecisionTree(jsonPath=None, paraDict=paraDicts)
    model.fit(X_train, y_train)
    pred_y = model.predict(X_test)





