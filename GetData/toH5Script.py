# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:58:55 2021

@author: eiahb
"""
import os
import pandas as pd
from GetData import load_material_data, load_barra_data, align_all_to
from Tool import globalVars
from Tool import Logger, GeneralData

PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'
loggerFolder = PROJECT_ROOT+"Tool\\log\\"
logger = Logger(loggerFolder, 'log')
globalVars.initialize(logger)

# load data to globalVars
load_material_data() 
load_barra_data()

#%%

H5_PATH =  os.path.join(PROJECT_ROOT, 'GetData/h5/')
barra = pd.HDFStore(os.path.join(H5_PATH, 'barra.h5'))
for k, v in globalVars.barra.items():
    print(k)
    barra[k] = v.to_DataFrame()
barra.close()
#%%
H5_PATH =  os.path.join(PROJECT_ROOT, 'GetData/h5/')
materialData = pd.HDFStore(os.path.join(H5_PATH, 'materialData.h5'))

for k, v in globalVars.materialData.items():
    print(k)
    materialData[k] = v.to_DataFrame()
materialData.close()

