# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 18:44:20 2020

@author: Mengjie Ye
"""

import numpy as np
import pandas as pd

from datetime import datetime
def train_test_slice(factors, dependents=None, trainStart=None, trainEnd=None, testStart=None, testEnd=None):
    # split all the factors and toPredicts to train part and test part according to input,
    # if trainStart = trainEnd: the user doesn't use panel data
    # slice factors at that date
    # else we slice factors from trainStart to trainEnd (closed date interval)
    # dependents always sliced by trainEnd
    # if dependents is None, return {} (can be used when we slice maskDict)
    factorTrainDict, factorTestDict = {}, {}
    dependentTrainDict, dependentTestDict = {}, {}
    
    if trainStart == trainEnd:
        for factor in factors:
            factorTrainDict[factor.name] = factor.get_data(at = trainEnd)
            factorTestDict[factor.name] = factor.get_data(at = testEnd)
    else:
        for factor in factors:
            factorTrainDict[factor.name] = np.vstack((factor.get_data(trainStart, trainEnd),
                                                     factor.get_data(at = trainEnd)))
            factorTestDict[factor.name] = np.vstack((factor.get_data(testStart, testEnd),
                                                     factor.get_data(at = testEnd)))
    if dependents is not None:      
        for name, dependent in dependents.items():
            dependentTrainDict[name] = dependent.get_data(at = trainEnd)
            dependentTestDict[name] = dependent.get_data(at = testEnd)
    
        return factorTrainDict, factorTestDict, dependentTrainDict, dependentTestDict



#%% test
from Tool import globalVars
from GetData.loadData import load_material_data, simple_load_factor
globalVars.initialize()
loadedDataList = load_material_data()

#TODO:use logger latter 
shiftedReturn = globalVars.materialData['pctChange'].get_shifted(-1)
# TODO: take the -1 as a para: the period we want to shift
shiftedReturn.metadata.update({'shiftN':-1})
shiftedReturn.name = 'shiftedReturn'
dependents = {}
factorNameList = []
allTradeDatetime = shiftedReturn.timestamp
dependents.update({'shiftedReturn':shiftedReturn})
toLoadFactors = ['close',
                         'high',
                         'low',
                         'open'
                         ] 
        
for aFactor in toLoadFactors:
    simple_load_factor(aFactor)
    factorNameList.append(aFactor)

# shiftedReturn.metadata.update({'shiftN':-1})
# shiftedReturn.name = 'shiftedReturn'
# allTradeDatetime = shiftedReturn.timestamp

# dependents.update({'shiftedReturn':shiftedReturn})


# TODO: should load from some factor.json file latter
############ this part realy socks 
############ to be modified latter
############ with nice designed globalVars
############ the factor in globalVars.factors is a dict 
toLoadFactors = ['adj_close',
                 'adj_high',
                 'adj_low',
                 'adj_open'
                 ] 
factorNameList = []
for aFactor in toLoadFactors:
    simple_load_factor(aFactor)
    factorNameList.append(aFactor)


def get_last_trade_date(date, n=1):
    assert allTradeDatetime[allTradeDatetime<date][-n], 'index out of range'
    return allTradeDatetime[allTradeDatetime<date][-n]

def get_next_trade_date( date, n=1):
    assert allTradeDatetime[allTradeDatetime>date][n], 'index out of range'
    return allTradeDatetime[allTradeDatetime>date][n]

backTestDate = allTradeDatetime[10]
panelSize = 5
trainTestGap = 1
testEnd = backTestDate
testStart = get_last_trade_date(testEnd, panelSize - 1)
trainEnd = get_last_trade_date(testEnd, trainTestGap)
trainStart = get_last_trade_date(trainEnd, panelSize - 1)

factorTrainDict, factorTestDict, dependentTrainDict, dependentTestDict = train_test_slice(
                factors = globalVars.factors.values(), dependents = dependents,
                trainStart = trainStart, trainEnd = trainEnd, testStart = testStart, testEnd = testEnd
            )
#%%

#%%

#%%
trainStart = '2020-11-30'
trainEnd = '2020-12-02'
# trainStart = pd.to_datetime(trainStart)
# trainEnd = pd.to_datetime(trainEnd)
f1 = globalVars.factors['adj_close']
trainFactors = f1.get_data(trainStart,None)

# TODO GeneralData写一个get_data one day之类的东西



