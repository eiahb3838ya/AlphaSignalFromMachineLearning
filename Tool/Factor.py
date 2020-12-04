# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:11:11 2020

@author: Evan
@reviewer: Robert
"""
from Tool.FactorProfileBase import FactorProfileBase
from Tool.GeneralData import GeneralData
from Tool import globals

class Factor(FactorProfileBase, GeneralData):
    def __init__(self, name, generalData = None, timestamp = None, columnNames = None,\
                 functionName = None, reliedDatasetNames_list = None, parameters_dict = None, **kwargs):
       
        FactorProfileBase.__init__(self)
        GeneralData.__init__(self, name, generalData, timestamp, columnNames, **kwargs)
        
        
        self.factorName = name
        self.functionName = functionName
        self.reliedDatasetNames_list = reliedDatasetNames_list
        
        # TODO __init__ with GeneralData
        
        
    def get_relied_dataset(self):
        outputDataset = {}
        for setName in self.reliedDatasetNames:
            try:
                outputDataset.update({
                        setName:getattr(globals, setName)
                    })
            except AttributeError :
                print("There is no dataset named {} in global".format(setName))
        return(outputDataset)
        
#%%  
# TODO: delete after test
if __name__ == "__main__":
    factorName = "toyFactor"
    functionName = "test"
    reliedDatasetNames_list = list()
    parameters_dict = dict()
    Klass = Factor(name = factorName, functionName = functionName, reliedDatasetNames_list = reliedDatasetNames_list,\
                   parameters_dict = parameters_dict)
  
