# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file
"""
import abc
# from abc import ABC
import numpy as np
# import cupy as np
import pandas as pd


class GeneralDataBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        # print('GeneralDataBase __init__')
        self.name = ''
        self.generalData = np.ndarray(0)
        self.timestamp = pd.Index([])
        self.columnNames = []
        self.metadata = {}
        
    def __str__(self):
        outputString = "{} : datashape of {} \n".format(self.name,\
                                                       self.generalData.shape)
        return(outputString)
    
    def __repr__(self):
        outputString = "{} : datashape of {} \n".format(self.name,\
                                                       self.generalData.shape)
        return(outputString)
    
    @abc.abstractmethod
    def get_data_tail(self, n = 10):
        pass
    
    @abc.abstractmethod
    def get_data_head(self, n = 10):
        pass
    
    @abc.abstractmethod
    def get_data(self, start = None, end = None, get_loc_method = None):
        pass
            
    
    
    
    