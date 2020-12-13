# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:16:04 2020

@author: Mengjie Ye
"""

from CrossSectionalModelBase import CrossSectionalModelBase
# from sklearn.linear_model import LinearRegression,Ridge,Lasso

import  statsmodels.api as sm
from statsmodels.api import OLS,WLS
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import abc
import matplotlib.pyplot as plt

class CrossSectionalModelLinear(CrossSectionalModelBase):
    
    def __init__(self, jsonPath = None, paraDict = {}):
        self.parameter = paraDict
        if jsonPath is not None:
            with open(jsonPath,'r') as f:
                self.parameter = json.loads(f)
        self.fit_intercept = self.parameter.get('fit_intercept',True)
        self.model = None
        
    def fit(self, X_train, y_train):
        if self.fit_intercept:
            X_train = sm.add_constant(X_train)
        self.model = OLS(y_train, X_train)
        self.res = self.model.fit()
        return self.res
        
        
    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.res.predict(X)
    
    def getPara(self):
        if self.parameter!={}:
            return pd.DataFrame.from_dict(self.parameter, 
                                          orient='index',
                                          columns= ['ParaValue'])
        else:
            print('Hyper parameters are default')
        
        
    def getModel(self):
        try:
            return self.res
        except:
            print('fit your model first!')
            return None
        
    
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
            y_pred = self.res.predict(kwargs['X'])
            
        
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
        return self.res.params
    
    def getModelSummary(self):
        '''
        get summary of the model
        
        return
        ----
        summary of model: coef, pvalue, t-statistics, R2, R2_adj...
        '''
        return self.res.summary()
    
    def scatterPred(self, y_real, y_pred):
        '''
        
        ----
            
            y: y_real
            kwargs: must input one of the following
                X: ndarray, input X to get y_pred
                y_pred: input y_pred directly
        '''
        plt.scatter(y_real,y_pred)
        plt.title('y_pred vs y_real')
        plt.xlabel('y_real')
        plt.ylabel('y_pred')
        plt.show()
# =============================================================================
#         y_pred = kwargs.get('y_pred',self.model.predict(kwargs['X']))
#         kind = kwargs.get('kind','scatter')
#         plt.figure()
#         plt.plot(y_real,y_pred,kind=kind)
#         plt.show()
# =============================================================================
        # print('This function hasn\'t be written')
        pass
    
#%%
class CrossSectionalModelOLS(CrossSectionalModelLinear):
    pass
    

#%%
class CrossSectionalModelRidge(CrossSectionalModelLinear):
    
    def __init__(self, jsonPath = None, paraDict = {}):
        self.parameter = paraDict
        if jsonPath is not None:
            with open(jsonPath,'r') as f:
                self.parameter = json.loads(f)
        self.fit_intercept = self.parameter.get('fit_intercept',True)
        self.model = None
        
    def fit(self, X, y, **kwargs):
        if self.fit_intercept:
            X = sm.add_constant(X)
        try:
            self.alpha = self.parameter['alpha']
        except:
            raise Exception('cannot find alpha! please set the penalty of Ridge')
        else:
            self.model = OLS(y, X)
        self.res = self.model.fit_regularized(alpha = self.alpha, L1_wt = 0, **kwargs)
        
#%%        
class CrossSectionalModelLasso(CrossSectionalModelBase):
    
    def fit(self, X, y, **kwargs):
        if self.fit_intercept:
            X = sm.add_constant(X)
        try:
            self.alpha = kwargs['alpha']
        except:
            raise Exception('cannot find alpha! please set the penalty of Lasso')
        else:
            self.model = OLS(y, X)
        self.res = self.model.fit_regularized(alpha = self.alpha, L1_wt = 1, **kwargs)
#%%


if __name__ == '__main__':
    data = {
        'name' : 'ACME',
        'shares' : 100,
        'price' : 542.23
    }
    
    json_str = json.dumps(data)
    
    # Writing JSON data
    with open(None, 'w') as f:
        json.dump(data, f)
    
    # Reading data back
    with open('data.json', 'r') as f:
        d = json.load(f)
    
    
    
    X = np.random.randn(30,5)
    y = np.random.randn(30)
    model = OLS(y,X)
    
    
    



     