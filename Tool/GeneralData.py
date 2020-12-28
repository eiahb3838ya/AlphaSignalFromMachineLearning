# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:24:11 2020

@author: Evan
"""
# for convenience to try with spyder
# use python -m Factor is the standard way to call modules main function

try:
    from .GeneralDataBase import GeneralDataBase
    # print('from .GeneralDataBase import GeneralDataBase')
except:
    from GeneralDataBase import GeneralDataBase 
    print('from GeneralDataBase import GeneralDataBase')

    

import copy
import numpy as np
import pandas as pd

class GeneralData(GeneralDataBase):
    def __init__(self, name, generalData = None, timestamp = None, columnNames = None, **kwargs):
        GeneralDataBase.__init__(self)
        
        # print('GeneralData __init__')
        
        self.name = name
        if generalData is None: 
            if 'filePath' in kwargs:
                try:
                    filePath = kwargs['filePath']
                    generalData = pd.read_csv(filePath, index_col=0)
                except Exception as e:
                    print(e)
                    print('We have a filePath but we can not load the generalData to pandas df structure')
            

        if isinstance(generalData, pd.DataFrame):
            try:
                self.columnNames = generalData.columns
                self.timestamp = pd.DatetimeIndex(generalData.index)
                self.generalData = generalData.to_numpy()
            except Exception as e:
                raise(e)
            
        elif isinstance(generalData, np.ndarray):
            assert timestamp is not None and columnNames is not None
            self.generalData = generalData
            
        else:
            raise TypeError('Must be np ndarray or pandas Dat
            aFrame')
                    
        if timestamp is not None:
            assert len(timestamp) == self.generalData.shape[0], 'the timestammp should \
                match the generalData size' 
            self.timestamp = timestamp
           
            
        if columnNames is not None:
            assert len(columnNames) == self.generalData.shape[1], 'the columnNames should \
                match the generalData size'
            self.columnNames = columnNames

        self.metadata.update({k:v for k, v in kwargs.items()})
        
    def get_data_tail(self, n = 10):
        return(self.generalData[-n:, :])
    
    def get_data_head(self, n = 10):
        return(self.generalData[:n, :])
    
    def get_data(self, start = None, end = None, at = None, get_loc_method = 'ffill'):  
        if at is None:
            if start is None:
                start = self.timestamp[0]
            if end is None:
                end = self.timestamp[-1]
        else:
            start = at
            end = at
            
        if not isinstance(start, int):
            try:
                start_loc = self.timestamp.get_loc(start, get_loc_method)
            except KeyError as ke:
                print('''
                      The start time is out of range or not in the index when you are not using default get_loc_method
                      ''')
                raise(ke)
            except Exception as e:
                raise e
        else:
            start_loc = start
            
        if not isinstance(end, int):
            try:
                end_loc = self.timestamp.get_loc(end, get_loc_method)
            except KeyError as ke:
                print('''
                      The end time is out of range or not in the index when you are not using default get_loc_method.
                      ''')
                raise(ke)
        else:
            end_loc = end
        
        assert (isinstance(start_loc, int)), "The input type of start and end should be int of loc \
            or datetime or str that datetimeIndex accessible"
        assert (isinstance(end_loc, int)), "The input type of start and end should be int of loc \
            or datetime or str that datetimeIndex accessible"
            
        if start_loc == end_loc:
            return(self.generalData[start_loc, :])
        else:
            return(self.generalData[start_loc:end_loc, :])
    
    def get_columnNames(self):
        return(self.columnNames)
    
    def get_timestamp(self):
        return(self.timestamp)
    
    def is_same_shape(self, anotherCls):
        assert isinstance(anotherCls, GeneralData)
        return(self.generalData.shape == anotherCls.generalData.shape)
    
    def get_shifted_generalData(self, shiftN):
        toOutput = self.generalData.copy()
        if shiftN >= 0:
            toOutput[-shiftN:, :] = np.nan
        else:
            toOutput[:-shiftN, :] = np.nan
        shifted = np.roll(toOutput, shiftN, axis = 0)
        return(shifted)
    
    def get_shifted(self, shiftN):
        toOutput = copy.copy(self)
        toOutput.generalData = toOutput.get_shifted_generalData(shiftN)
        return(toOutput)
        
if __name__ ==  "__main__":
    DATA_PATH = 'D:\\ml\\AlphaSignalFromMachineLearningLocal\\GetData/tables/S_DQ_ADJOPEN.csv'
    klass = GeneralData(name = 'adj_open', filePath = DATA_PATH)
    # klass.get_data('2005', '2014-01-06')
    isinstance(klass, GeneralData)
    
    
    #%% how to get data of a single slice
    klass.get_data(at = '2018-03-09')
    klass.get_data(start = '2018-03-09', end = '2018-03-09')






























    
    
        
        