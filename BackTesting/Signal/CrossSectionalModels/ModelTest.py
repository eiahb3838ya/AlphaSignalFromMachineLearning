# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:10:33 2020

@author: Mengjie Ye
"""

import abc
from ScoreMethod import scoreMethodDict

class ModelTest(object, metaclass=abc.ABCMeta):
    def __init__(self, model = None, scoreMethodDict = scoreMethodDict):
        '''
        model: sklearn like model
        '''
        self.model = model
        self.scoreMethodDict = scoreMethodDict
    
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
        
        scoreMethod = kwargs.get('scoreMethod','r2')
        scoreMethod = self.scoreMethodDict[scoreMethod]
        return scoreMethod(y_true, y_pred)