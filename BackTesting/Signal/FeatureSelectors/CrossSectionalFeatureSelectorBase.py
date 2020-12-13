# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:10:59 2020

@author: Evan Hu (Yi Fan Hu)

"""

import abc

class CrossSectionalFeatureSelectionBase(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        self.parameter = {}
        self.selector = None
        pass
    
    @abc.abstractmethod
    def fit(self, X_train):
        # fit the model with the input data
        # self.model.fit(X,y)
        pass
    
    @abc.abstractmethod
    def transform(self, X):
        # the one method that to be called to perform prediction
        # return(self.model.predict(X))
        pass
    
    @abc.abstractmethod
    def fit_transform(self, X):
        # the one method that to be called to perform prediction
        # return(self.model.predict(X))
        pass
    
    @abc.abstractmethod
    def getPara(self):
        # return the hyperparameter of the model
        # maybe from another file json-like or another module
        # for the cv cases
        # do some how cv or things to decide the hyperparameter in this
        
        # if self.parameter == {}:
        #     do something
        # else:
        #     return(self.parameter)
        pass
    
    @abc.abstractmethod
    def getSelector(self):
        # return the hyperparameter of the model
        # maybe from another file json-like or another module
        # for the cv cases
        # do some how cv or things to decide the hyperparameter in this
        
        # if self.parameter == {}:
        #     do something
        # else:
        #     return(self.parameter)
        pass