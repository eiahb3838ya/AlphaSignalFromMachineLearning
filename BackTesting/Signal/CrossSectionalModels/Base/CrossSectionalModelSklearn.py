# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 00:19:56 2020

@author: Mengjie Ye
"""

# import os
# import sys
# sys.path.append(os.getcwd())
try:
    from .CrossSectionalModelBase import CrossSectionalModelBase
except:
    from CrossSectionalModelBase import CrossSectionalModelBase
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV
from ModelTest.ModelTest import ModelTest
# import matplotlib.pyplot as plt

class CrossSectionalModelSklearn(CrossSectionalModelBase, ModelTest):
    
    def __init__(self, jsonPath = None, 
                 paraDict = {}, paraGrid = None, 
                 json_first = True):
        # take both jsonPath and the input para into consideration
        # json_first = True, when merge the parameters and there is a conflict
        # we would consider the para in json file first
        self.parameter = paraDict
        if jsonPath is not None:
            if json_first:
                with open(jsonPath, 'r') as f:
                    # !!! should be a copy of paraDict
                    # python: call by reference
                    temp = paraDict.copy()
                    temp.update(json.load(f))
                    # paraDict = temp
                    # use self.parameter directly rather than paraDict
                    # avoid id(paraDict) changes
                    self.parameter = temp
            else:
                with open(jsonPath, 'r') as f:
                    temp = json.load(f)
                    temp.update(paraDict)
                    # paraDict = temp
                    self.parameter = temp
            # restore the new paraDict into that jsonPath
            with open(jsonPath, 'w') as f:
                # json.dump(paraDict, f)
                json.dump(self.parameter, f)
        # self.parameter = paraDict
        # define your model when inherit this class
        self.model = None
        # if we want to use Cross Validation to search for the best para
        # input the paraGrid for Grid Search
        self.paraGrid = paraGrid

    def fit(self, X_train, y_train, **kwargs):
        # use cv to get best model and para
        # or just fit the model
        # set kwargs for GridSearchCV()
        if self.paraGrid is not None:
            reg = GridSearchCV(
                self.model, self.paraGrid, **kwargs
                )
            reg.fit(X_train, y_train)
            self.parameter = reg.best_params_
            self.model = reg.best_estimator_
        else:
            self.model.fit(X_train, y_train)
      
    def predict(self, X):
        return self.model.predict(X)
    
    def get_para(self, verbal = False): 
        if self.parameter!={}:
            # verbal: if True: display the parameter as a dataframe
            # verbal: if False: return a dict
            if verbal is False:
                return self.parameter
            else:
                return pd.DataFrame.from_dict(self.parameter, 
                                          orient='index',
                                          columns= ['ParaValue'])
        else:
            print('Hyper parameters are default')
        
    def get_model(self):
        return self.model

    def get_coef(self):
        # get estimated coefficients for the linear regression problem
        return self.model.coef_
    
    def get_intercept(self):
        # get estimated intercept for the linear regression problem
        return self.model.intercept_





