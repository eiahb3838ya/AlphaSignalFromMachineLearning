from sklearn.tree import DecisionTreeRegressor
from CrossSectionalModelBase import CrossSectionalModelBase
import json
import sys
import matplotlib.pyplot as plt
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

    def get_model(self):
        return self.model

    def get_score(self,y_true,y_pre):
        return self.model.score(y_true, y_pre)

    def get_para(self):
        return self.parameter


if __name__ == '__main__':
    from testCSModel import create_regression_dataset

    X_train, y_train, X_test, y_test = create_regression_dataset()

    paraDicts = {'max_depth': 6}

    model = CrossSectionalModelDecisionTree(jsonPath=None, paraDict=paraDicts)
    model.fit(X_train, y_train)
    pred_y = model.predict(X_test)

    plt.scatter(y_test,pred_y)
    plt.title('y_pred vs y_real')
    plt.xlabel('y_real')
    plt.ylabel('y_pred')
    plt.show()



