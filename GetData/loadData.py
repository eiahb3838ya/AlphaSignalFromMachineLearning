# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:43:53 2020

@author: Evan Hu (Yi Fan Hu)

"""

import os
import pandas as pd
from Tool import globals
from Tool.GeneralData import GeneralData
#%%


dataFileDict = {
        'adj_close':'S_DQ_ADJCLOSE.csv',
        'adj_high':'S_DQ_ADJHIGH.csv',
        'adj_low':'S_DQ_ADJLOW.csv',
        'adj_open':'S_DQ_ADJOPEN.csv',
        'adj_preclose':'S_DQ_ADJPRECLOSE.csv',
        'amount':'S_DQ_AMOUNT.csv',
        'volume':'S_DQ_VOLUME.csv'
    }

#%%
DATA_PATH = './GetData/tables/'

# tmp = pd.read_csv(os.path.join(DATA_PATH,'S_DQ_ADJCLOSE.csv'), index_col=0)
# adj_close = GeneralData('adj_close', tmp)
# globals.register('adj_close', adj_close)
#%%
def loadData(dataFileDict = dataFileDict, DATA_PATH = DATA_PATH):
    for k, v in dataFileDict.items():
        tmp = pd.read_csv(os.path.join(DATA_PATH,v), index_col=0)
        data = GeneralData(name = k, generalData = tmp)
        globals.register(k, data)
    





