# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:24:11 2020

@author: eiahb
"""

from Tool.GeneralDataBase import GeneralDataBase
import numpy as np
import pandas as pd

class GeneralData(GeneralDataBase):
    def __init__(self, name, generalData = None, timestamp = None, columnNames = None, **kwargs):
        GeneralDataBase.__init__(self)
        
        # print('GeneralData __init__')
        self.name = name
            
        if generalData is not None:
            if isinstance(generalData,pd.DataFrame):
                try:
                    self.columnNames = generalData.columns
                    self.timestamp = pd.DatetimeIndex(generalData.index)
                    self.generalData = generalData.to_numpy()
                except Exception as e:
                    raise(e)
                
            elif isinstance(generalData, np.ndarray):
                self.generalData = generalData
                
        elif 'filePath' in kwargs:
            try:
                filePath = kwargs['filePath']
                generalData = pd.read_csv(filePath)
                self.columnNames = generalData.columns

                self.timestamp = pd.DatetimeIndex(generalData.index)
                self.generalData = generalData.to_numpy()
                
            except Exception as e:
                raise(e)
        
            
        if timestamp is not None:
            assert len(timestamp) == self.generalData.shape[0], 'the timestammp should match the generalData size' 
            self.timestamp = timestamp
           
            
        if columnNames is not None:
            assert len(columnNames) == self.generalData.shape[1], 'the columnNames should match the generalData size'
            self.columnNames = columnNames

        self.metadata.update({k:v for k, v in kwargs.items()})
        
    def get_data_tail(self, n = 10):
        return(self.generalData[-n:, :])
    
    def get_data_head(self, n = 10):
        return(self.generalData[:n, :])
    
    def get_data(self, start = None, end = None, get_loc_method = None):        
        if start is None:
            start = self.timestamp[0]
        if end is None:
            end = self.timestamp[-1]
            
        if not isinstance(start, int):
            start_loc = self.timestamp.get_loc(start, get_loc_method)
        if not isinstance(end, int):
            end_loc = self.timestamp.get_loc(end, get_loc_method)
        return(self.generalData[start_loc:end_loc, :])
        
if __name__ ==  "__main__":
    DATA_PATH = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\02data\\ElementaryFactor-复权收盘价.csv'
    testData = pd.read_csv(DATA_PATH, index_col = 0)
    testData.index = testData.index.astype(str)
    klass = GeneralData(name = 'close', generalData = testData)
    
    
    
        
        