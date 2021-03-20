# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:23:09 2020

@author: Mengjie Ye
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


scoreMethodDict = {
    'r2':r2_score, 
    'mse':mean_squared_error, 
    'mae':mean_absolute_error
    }