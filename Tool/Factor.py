# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:11:11 2020

@author: Evan
@reviewer: Robert
"""
# for convenience to try with spyder
# use python -m Factor is the standard way to call modules main function
#%%
import pickle
try :
    from .FactorProfileBase import FactorProfileBase
    from .GeneralData import GeneralData
    # print("from .FactorProfileBase import FactorProfileBase")
except :
    from FactorProfileBase import FactorProfileBase
    from GeneralData import GeneralData
    # print("from FactorProfileBase import FactorProfileBase")


DEFAULT_PROTOCOL = 2
#%%
class Factor(FactorProfileBase, GeneralData):
    def __init__(self, name: str = None, generalData = None, timestamp = None, columnNames = None,\
                 functionName = None, reliedDatasetNames: dict = None, parameters_dict: dict = None, **kwargs):
            
        FactorProfileBase.__init__(self)
        GeneralData.__init__(self, name, generalData, timestamp, columnNames, **kwargs)
        

        self.functionName = functionName
        self.reliedDatasetNames = reliedDatasetNames
        self.parameters_dict = parameters_dict

        
        
    def get_relied_dataset(self):
        outputDataset = {}
        for k, v in self.reliedDatasetNames.items():
            for dataset in v:
                try:
                    outputDataset.update({
                            dataset:getattr(globalVars, k)[dataset]
                        })
                except AttributeError :
                    print("There is no dataset named {} in global".format(k))
        return(outputDataset)

    def save(self, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL):
        with open(f, 'wb') as factorfilehandle:
            pickle_module.dump(self, factorfilehandle)

    @staticmethod
    def load(f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL):
        with open(f, 'rb') as factorfilehandle:
            unpickler = pickle.Unpickler(factorfilehandle)
            factor = unpickler.load()
        return(factor)

  
#%%
if __name__ == '__main__':

    
    import pandas as pd
    import os
    PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'

    os.chdir(PROJECT_ROOT)
    from Tool import globalVars
    from GetData import load_data, align_all_to
    from Tool import Logger
    loggerFolder = PROJECT_ROOT+"Tool\\log\\"
    logger = Logger(loggerFolder, 'log')
    globalVars.initialize(logger)

    # read h5
    # 用例 1 
    load_data("materialData",
        os.path.join(os.path.join(PROJECT_ROOT,"data"), "h5")
    )

    
    factorName = "close"
    functionName = "test"

    reliedDatasetNames= {'materialData':['open', 'close']}
    parameters_dict = dict()
    
    klass = Factor(name = factorName,
                generalData = globalVars.materialData['close'],
                functionName = functionName,
                reliedDatasetNames = reliedDatasetNames,
                parameters_dict = parameters_dict
            )
    reliedDatasets = klass.get_relied_dataset()
    klass.save('try.pickle')
    # factor = Factor.load('try.pickle')
    factor = Factor.load(os.path.join(os.path.join(PROJECT_ROOT,'data'), 'factors\\try.pickle'))
        

# %%


# %%
