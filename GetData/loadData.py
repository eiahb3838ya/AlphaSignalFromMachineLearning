# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:43:53 2020

@author: Evan Hu (Yi Fan Hu)

"""

import os
import pandas as pd
from Tool import globalVars
from Tool.GeneralData import GeneralData
from Tool.Factor import Factor
#%%


materialDataFileDict = {
        'close':'S_DQ_ADJCLOSE.csv',
        'high':'S_DQ_ADJHIGH.csv',
        'low':'S_DQ_ADJLOW.csv',
        'open':'S_DQ_ADJOPEN.csv',
        'preclose':'S_DQ_ADJPRECLOSE.csv',
        'amount':'S_DQ_AMOUNT.csv',
        'volume':'S_DQ_VOLUME.csv',
        'pctChange':'S_DQ_PCTCHANGE.csv'
    }

#%%
PROJ_ROOT = 'C:/Users/eiahb/Documents/MyFiles/WorkThing/tf/01task/GeneticProgrammingProject/Local'
# !!! change the PROJ_ROOT 
# PROJ_ROOT = 'D:\AlphaSignalFromMachineLearning'
# PROJ_ROOT = os.getcwd()
# PROJ_ROOT = PROJ_ROOT[:PROJ_ROOT.index('AlphaSignalFromMachineLearning')+len('AlphaSignalFromMachineLearning')]
DATA_PATH = os.path.join(PROJ_ROOT, 'GetData/tables/')

#%% load data functions
# def load_data(dataFileDict = materialDataFileDict, DATA_PATH = DATA_PATH):
#     toReturnList = []
#     for k, v in dataFileDict.items():
#         data = GeneralData(name = k, filePath = os.path.join(DATA_PATH,v))
#         globalVars.register(k, data)
#         toReturnList.append(k)
#     return(toReturnList)

def load_material_data(dataFileDict = materialDataFileDict, DATA_PATH = DATA_PATH):
    toReturnList = []
    
    # add dictionary material_data to globalVars
    if 'materialData' not in globalVars.varList:
        globalVars.register('materialData', {})
        
    for k, v in dataFileDict.items():
        data = GeneralData(name = k, filePath = os.path.join(DATA_PATH,v))
        globalVars.materialData[k] = data
        print('==================================================================\n\
              {} is now in globalVars.materialData\n'.format(k), data)
        toReturnList.append(k)
    return(toReturnList)

def simple_load_factor(factorName):
    if 'factors' not in globalVars.varList:
        globalVars.register('factors', {})
        
    #TODO after done  TODO __init__ with GeneralData, load factors as dtype of Factor
    globalVars.factors[factorName] = globalVars.materialData[factorName]
    print(factorName, 'is now in globalVars.factors')

    
    
    





