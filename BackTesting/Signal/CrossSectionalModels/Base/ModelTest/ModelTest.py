# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:10:33 2020

@author: Mengjie Ye
"""
import os
import sys
sys.path.append(os.getcwd())
import abc
# =============================================================================
# try:
#     from .ScoreMethod import scoreMethodDict
# except:
#     from ScoreMethod import scoreMethodDict
# =============================================================================
from .ScoreMethod import scoreMethodDict

class ModelTest():
    
    def __init__(self, model = None):
        '''
        model: sklearn like model
        '''
        self.model = model
        # self.scoreMethodDict = scoreMethodDict
    
    def get_score(self, y_true, y_pred, scoreMethod = 'r2', scoreMethodDict = scoreMethodDict, **kwargs):
        '''
        get score of the prediction based on the scoreMethod
        
        ----
            
            y_true: array-like of shape (n_samples,) or 
                    (n_samples, n_outputs)
            y_pred: array-like of shape (n_samples,) or 
                    (n_samples, n_outputs)
            scoreMethod: metrics to score the model
                        'r2', 'mse', 'mae'
        '''
        
        scoreMethod = scoreMethodDict[scoreMethod]
        return scoreMethod(y_true, y_pred)