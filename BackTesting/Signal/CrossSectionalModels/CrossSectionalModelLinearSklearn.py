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
# import matplotlib.pyplot as plt

class CrossSectionalModelLinear(CrossSectionalModelBase):
    
    def __init__(self, jsonPath = None, paraDict = {}, paraGrid = None):
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
    
    def getPara(self):
        if self.parameter!={}:
            return pd.DataFrame.from_dict(self.parameter, 
                                          orient='index',
                                          columns= ['ParaValue'])
        else:
            print('Hyper parameters are default')
        
    def getModel(self):
        return self.model
        
    
    def getScore(self, y_real, **kwargs):
        '''
        get score of the prediction based on the scoreMethod
        
        ----
            
            y: y_real
            kwargs:
                scoreMethod: str
                        'r2': r2_score
                        'mse': mean_squared_error
                        'mae': mean_absolute_error
                X: ndarray, input X to get y_pred
                y_pred: input y_pred directly
        '''
        if 'y_pred' in kwargs.keys():
            y_pred = kwargs['y_pred']
        elif 'X' in kwargs.keys():
            y_pred = self.predict(kwargs['X'])
        def r2(y_real, y_pred):
            return r2_score(y_real, y_pred)
        def mse(y_real, y_pred):
            return mean_squared_error(y_real, y_pred)
        def mae(y_real, y_pred):
            return mean_absolute_error(y_real, y_pred)
        methodDict = {'r2':r2, 'mse':mse, 'mae':mae}
        scoreMethod = kwargs.get('scoreMethod','r2')
        scoreMethod = methodDict[scoreMethod]
        return scoreMethod(y_real, y_pred)
    
    def getCoef(self):
        '''
        get estimated coefficients for the linear regression problem
        '''
        return self.model.coef_
    
    def getIntercept(self):
        return self.intercept_
    
#%%
class CrossSectionalModelOLS(CrossSectionalModelLinear):
    
    def __init__(self, jsonPath = None, paraDict = {}):
        CrossSectionalModelLinear.__init__(self, jsonPath = None, paraDict = {})
        self.model = LinearRegression(**self.parameter)
        
# =============================================================================
#     def __init__(self, jsonPath = None, paraDict = {}):
#         self.parameter = paraDict
#         if jsonPath is not None:
#             with open(jsonPath,'r') as f:
#                 self.parameter = json.load(f)
#         self.model = LinearRegression(**self.parameter)
# =============================================================================
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

#%%
class CrossSectionalModelRidge(CrossSectionalModelLinear):
    
    def __init__(self,jsonPath = None, paraDict = {}, paraGrid = None):
        CrossSectionalModelLinear.__init__(self,jsonPath = None, paraDict = {}, paraGrid = None)
        self.model = Ridge(**self.parameter)
# =============================================================================
#     def __init__(self, jsonPath = None, paraDict = {}, paraGrid = None):
#         self.parameter = paraDict
#         if jsonPath is not None:
#             with open(jsonPath,'r') as f:
#                 self.parameter = json.load(f)
#         self.model = Ridge(**self.parameter)
#         self.paraGrid = paraGrid
# =============================================================================
        
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
# =============================================================================
#     def __init__(self, jsonPath = None, paraDict = {}, paraGrid = None):
#         self.parameter = paraDict
#         if jsonPath is not None:
#             with open(jsonPath,'r') as f:
#                 self.parameter = json.load(f)
#         self.model = Lasso(**self.parameter)
#         self.paraGrid = paraGrid
# =============================================================================
        
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

   
    
    
    



     