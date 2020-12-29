# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:03:08 2020

@author: Evan Hu
"""

import pandas as pd
import os


SKIP_N_ROWS = 2

# C:\Users\eiahb\Documents\MyFiles\WorkThing\tf\02data
DATA_PATH = '..\\..\\..\\02data'
dataFileList = ['AShareEODPrices.pkl', 'Barra_CNE5_factordata.pkl']
#%%
OUTPUT_PATH = 'GetData\\tables\\materialData'

os.makedirs(OUTPUT_PATH, exist_ok=bool)

aDataFile = dataFileList[0]
aFilePath = os.path.join(DATA_PATH, aDataFile)

df = pd.read_pickle(aFilePath)
toPivotColumns = df.columns[2:]

for aPivotTarget in toPivotColumns:
    pivotedTable = df.pivot(df.columns[1], df.columns[0], aPivotTarget)
    pivotedTable = pivotedTable.iloc[SKIP_N_ROWS:, :]
    pivotedTable.index = pd.DatetimeIndex(pivotedTable.index)
    pivotedTable.to_csv(os.path.join(OUTPUT_PATH, '{}.csv'.format(aPivotTarget)))


#%%
OUTPUT_PATH = 'GetData\\tables\\barra'
os.makedirs(OUTPUT_PATH, exist_ok=bool)
aDataFile = dataFileList[1]
aFilePath = os.path.join(DATA_PATH, aDataFile)
df = pd.read_pickle(aFilePath)
df = df.drop(columns = 'Unnamed: 0')
toPivotColumns = df.columns[2:]
aPivotTarget = toPivotColumns[0]
SKIP_N_ROWS = 0
for aPivotTarget in toPivotColumns:
    pivotedTable = df.pivot(df.columns[0], df.columns[1], aPivotTarget)
    pivotedTable = pivotedTable.iloc[SKIP_N_ROWS:, :]
    pivotedTable.index = pd.DatetimeIndex(pivotedTable.index.astype(str))
    pivotedTable.to_csv(os.path.join(OUTPUT_PATH, '{}.csv'.format(aPivotTarget)))

