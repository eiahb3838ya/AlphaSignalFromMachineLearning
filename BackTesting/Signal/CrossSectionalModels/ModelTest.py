# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:10:33 2020

@author: Mengjie Ye
"""

import abc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelTest(object, metaclass=abc.ABCMeta):
    def __init__(self,model):
        '''
        model: sklearn like model
        '''
        self.model = model
    
    def get_score(self, y_true, **kwargs):
        '''
        get score of the prediction based on the scoreMethod
        
        ----
            
            y_true: array-like of shape (n_samples,) or 
                    (n_samples, n_outputs)
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
            y_pred = self.model.predict(kwargs['X'])
            
        def r2(y_true, y_pred):
            return r2_score(y_true, y_pred)
        def mse(y_true, y_pred):
            return mean_squared_error(y_true, y_pred)
        def mae(y_true, y_pred):
            return mean_absolute_error(y_true, y_pred)
        methodDict = {'r2':r2, 'mse':mse, 'mae':mae}
        scoreMethod = kwargs.get('scoreMethod','r2')
        scoreMethod = methodDict[scoreMethod]
        return scoreMethod(y_true, y_pred)