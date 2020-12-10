# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:03:08 2020

@author: Evan Hu
"""

import pandas as pd
import os

SKIP_N_ROWS = 2
DATA_PATH = '.\\pickles'
OUTPUT_PATH = '.\\tables'

dataFileList = ['AShareEODPrices.pkl']
os.makedirs(OUTPUT_PATH, exist_ok=bool)

aDataFile = dataFileList[0]
aFilePath = os.path.join(DATA_PATH, aDataFile)
df = pd.read_pickle(aFilePath)
toPivotColumns = df.columns[2:]

# aPivotTarget = toPivotColumns[0]

for aPivotTarget in toPivotColumns:
    pivotedTable = df.pivot(df.columns[1], df.columns[0], aPivotTarget)
    pivotedTable = pivotedTable.iloc[SKIP_N_ROWS:, :]
    pivotedTable.index = pd.DatetimeIndex(pivotedTable.index)
    pivotedTable.to_csv(os.path.join(OUTPUT_PATH, '{}.csv'.format(aPivotTarget)))
