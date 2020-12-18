# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 19:50:33 2020

@author: Mengjie Ye
"""

from CrossSectionalFeatureSelectorBase import  CrossSectionalFeatureSelectionBase
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import pandas as pd
import json

class CrossSectionalFeatureSelectionLasso(CrossSectionalFeatureSelectionBase):
    
    def __init__(self, jsonPath = None, paraDict = {}):
        self.parameter = paraDict
        if jsonPath is not None:
            with open(jsonPath,'r') as f:
                self.parameter = json.load(f)
        self.selector = Lasso(**self.parameter)
        
    def fit(self, X, y):
        self.selector.fit(X, y)
        
    def transform(self, X):
        try:
            coef = self.selector.coef_
            coefIdx = [i for i in range(len(coef)) if coef[i] != 0]
            return X[:,coefIdx]
        except :
            raise Exception('please fit your selector first!')
            
    def fit_transform(self, X, y):
        self.selector.fit(X, y)
        coef = self.selector.coef_
        coefIdx = [i for i in range(len(coef)) if coef[i] != 0]
        return X[:,coefIdx]
    
    def getPara(self):
        if self.parameter!={}:
            return pd.DataFrame.from_dict(self.parameter, 
                                          orient='index',
                                          columns= ['ParaValue'])
        else:
            print('Hyper parameters are default')
            
    def getSelector(self):
        return self.selector
            
            