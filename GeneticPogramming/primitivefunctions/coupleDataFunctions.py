# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:27:20 2020

@author: Evan Hu (Yi Fan Hu)

"""

#%%
import numpy as np
from Tool.GeneralData import GeneralData
import copy
import warnings
warnings.filterwarnings("ignore")

#%%
def max_(this: GeneralData, that: GeneralData) -> GeneralData:
    assert this.generalData.shape == that.generalData.shape

    outputToReturn = copy.copy(this)
    outputToReturn.generalData = np.maximum(this.generalData, that.generalData)
    return outputToReturn

def min_(this: GeneralData, that: GeneralData) -> GeneralData:
    assert this.generalData.shape == that.generalData.shape

    outputToReturn = copy.copy(this)
    outputToReturn.generalData = np.minimum(this.generalData, that.generalData)
    return outputToReturn

# 𝑚𝑒𝑎𝑛(𝑎, 𝑏) 2 a 和 b 的均值
def mean_(this: GeneralData, that: GeneralData) -> GeneralData:
    assert this.generalData.shape == that.generalData.shape
    outputToReturn = copy.copy(this)
    outputToReturn.generalData = np.add(this.generalData, that.generalData)
    return outputToReturn
#%%
if __name__ == '__main__':
    PROJECT_ROOT = "c:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\"
    import os
    os.chdir(PROJECT_ROOT)
    from Tool import globalVars
    from GetData import load_data
    
    globalVars.initialize()
    load_data("materialData",
        os.path.join(os.path.join(PROJECT_ROOT,"data"), "h5")
    )
    this = globalVars.materialData['close']
    that = globalVars.materialData['open']
#%% max_
    max_(this, that)
#%% min_
    min_(this, that)      
#%% mean_
    mean_(this, that)

# %%
