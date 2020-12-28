# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 18:44:20 2020

@author: Mengjie Ye
"""

def train_test_slice(factors, dependents, 
                     trainStart, trainEnd = None, 
                     testStart = None, testEnd = None):
    # split all the factors and dependents to train part and test part according to input,
    # if end part isn't passed in, slice one period as default, 
    # if the test start isn't passed in,
    # take the very next time period of trainEnd,
    # the input of factors could be a list of factors or just one Factor
    timeStamp = factors[0].timestamp
    if trainEnd is None:
        trainEnd = timeStamp[1+list(timeStamp).index(pd.to_datetime(trainStart))]
    if testStart is None:
        # ???? 在get_data的时候end没有被取到
        # testStart = timeStamp[1+list(timeStamp).index(pd.to_datetime(trainEnd))]
        testStart = trainEnd
    if testEnd is None:
        testEnd = timeStamp[1+list(timeStamp).index(pd.to_datetime(testStart))]
     
    trainFactors, testFactors = [], []
    for factor in factors:
        trainFactors.append(factor.get_data(trainStart,trainEnd))
        testFactors.append(factor.get_data(testStart,testEnd))
    trainDependents, testDependents = [], []
    for dependent in dependents:
        trainDependents.append(dependent.get_data(trainStart,trainEnd))
        testDependents.append(dependent.get_data(testStart,testEnd))
    return trainFactors, testFactors, trainDependents, testDependents




#%% test
from Tool import globalVars
from GetData.loadData import load_material_data, simple_load_factor
globalVars.initialize()
loadedDataList = load_material_data()

#TODO:use logger latter 
print('We now have {} in our globalVar now'.format(loadedDataList))
try:
    shiftedReturn = globalVars.pctChange.get_shifted(-1)
except AttributeError:
    raise AttributeError('There\'s no pctChange in globalVars')
except Exception as e :
    print(e)
    raise 
# TODO: take the -1 as a para: the period we want to shift
shiftedReturn.metadata.update({'shiftN':-1})
shiftedReturn.name = 'shiftedReturn'
allTradeDatetime = shiftedReturn.timestamp
dependents = {}
dependents.update({'shiftedReturn':shiftedReturn})


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

#%%
import pandas as pd

from datetime import datetime
#%%
trainStart = '2020-11-30'
trainEnd = '2020-12-02'
# trainStart = pd.to_datetime(trainStart)
# trainEnd = pd.to_datetime(trainEnd)
f1 = globalVars.factors['adj_close']
trainFactors = f1.get_data(trainStart,None)

# TODO GeneralData写一个get_data one day之类的东西



