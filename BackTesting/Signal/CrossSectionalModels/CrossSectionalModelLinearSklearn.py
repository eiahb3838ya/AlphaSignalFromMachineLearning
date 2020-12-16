# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:16:04 2020

@author: Mengjie Ye
"""

from CrossSectionalModelBase import CrossSectionalModelBase
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import abc
from sklearn.model_selection import GridSearchCV
from ModelTest import ModelTest
# import matplotlib.pyplot as plt

class CrossSectionalModelLinear(CrossSectionalModelBase, ModelTest):
    
    def __init__(self, jsonPath = None, paraDict = {}, paraGrid = None):
        ModelTest.__init__(self)
        self.parameter = paraDict
        if jsonPath is not None:
            with open(jsonPath,'r') as f:
                self.parameter = json.load(f)
        self.model = None
        self.paraGrid = paraGrid
        
    
    def fit(self, X_train, y_train):
        pass
        
    def predict(self, X):
        return self.model.predict(X)
    
    def get_para(self):
        if self.parameter!={}:
            return pd.DataFrame.from_dict(self.parameter, 
                                          orient='index',
                                          columns= ['ParaValue'])
        else:
            print('Hyper parameters are default')
        
    def get_model(self):
        return self.model

    def get_coef(self):
        '''
        get estimated coefficients for the linear regression problem
        '''
        return self.model.coef_
    
    def get_intercept(self):
        return self.intercept_
    
#%%
class CrossSectionalModelOLS(CrossSectionalModelLinear):
    
    def __init__(self, jsonPath = None, paraDict = {}):
        CrossSectionalModelLinear.__init__(self, jsonPath = None, paraDict = {})
        self.model = LinearRegression(**self.parameter)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

#%%
class CrossSectionalModelRidge(CrossSectionalModelLinear):
    
    def __init__(self,jsonPath = None, paraDict = {}, paraGrid = None):
        CrossSectionalModelLinear.__init__(self,jsonPath = None, paraDict = {}, paraGrid = None)
        self.model = Ridge(**self.parameter)

    def fit(self, X_train, y_train, **kwargs):
        if self.paraGrid is not None:
            reg = GridSearchCV(
                Ridge(), self.paraGrid, **kwargs
                )
            reg.fit(X_train, y_train)
            self.parameter = reg.best_params_
            self.model = reg.best_estimator_
        else:
            self.model.fit(X_train, y_train)

#%%        
class CrossSectionalModelLasso(CrossSectionalModelLinear):
    
    def __init__(self,jsonPath = None, paraDict = {}, paraGrid = None):
        CrossSectionalModelLinear.__init__(self,jsonPath = None, paraDict = {}, paraGrid = None)
        self.model = Lasso(**self.parameter)
        
    def fit(self, X_train, y_train, **kwargs):
        if self.paraGrid is not None:
            reg = GridSearchCV(
                Lasso(), self.paraGrid, **kwargs
                )
            reg.fit(X_train, y_train)
            self.parameter = reg.best_params_
            self.model = reg.best_estimator_
        else:
            self.model.fit(X_train, y_train)
        
#%%

   
    
    
    



     