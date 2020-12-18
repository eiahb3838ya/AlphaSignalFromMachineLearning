# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:11:11 2020

@author: Evan
@reviewer: Robert
"""
# for convenience to try with spyder
# use python -m Factor is the standard way to call modules main function
try :
    from .FactorProfileBase import FactorProfileBase
    from .GeneralData import GeneralData
except :
    from Tool import globalVars
    from FactorProfileBase import FactorProfileBase
    from GeneralData import GeneralData

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
                        setName:getattr(globalVars, setName)
                    })
            except AttributeError :
                print("There is no dataset named {} in global".format(setName))
        return(outputDataset)
        
#%%  
# TODO: delete after test
if __name__ == "__main__":
    import pandas as pd
    factorName = "close"
    functionName = "test"
    DATA_PATH = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\02data\\ElementaryFactor-复权收盘价.csv'
    testData = pd.read_csv(DATA_PATH, index_col = 0)
    testData.index = testData.index.astype(str)
    reliedDatasetNames_list = list()
    parameters_dict = dict()
    
    klass = Factor(name = factorName, generalData = testData, functionName = functionName, reliedDatasetNames_list = reliedDatasetNames_list,\
                   parameters_dict = parameters_dict)
        
    isinstance(klass, FactorProfileBase)
    isinstance(klass, GeneralData)
