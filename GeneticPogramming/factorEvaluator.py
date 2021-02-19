# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:35:48 2020

在這裡定義所有對單因子評價的方式，輸入的因子是已經輸出成 GeneralData 的因子值
而後面也可以使用其他的數據， typically 就是用收益率做 ic, icir 等等
回傳一個數值

@author: eiahb
"""

#%%
import os
PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'
os.chdir(PROJECT_ROOT)
import numpy as np
from copy import copy, deepcopy
from Tool.GeneralData import GeneralData
from GeneticPogramming.utils import rowwise_corrcoef



#%% evaluate function 评价函数
def rankic_evaluator(factor : GeneralData, shiftedPctChange:GeneralData) -> float:   
    corr_np = rowwise_corrcoef(np.argsort(factor.generalData, axis = 1), np.argsort(shiftedPctChange.generalData, axis = 1))
    if corr_np.mask.sum() > len(corr_np)/2:
        print("trouble factor with unevaluate days {} out of {} days".format(corr_np.mask.sum(), len(corr_np)))
        ic = -1
    else:
        ic = np.nanmean(corr_np)
    return(ic)

# 
def ic_evaluator(factor : GeneralData, shiftedPctChange:GeneralData) -> float:   
    corr_np = rowwise_corrcoef(factor, shiftedPctChange)
    if corr_np.mask.sum() > len(corr_np)/2:
        print("trouble factor with unevaluate days {} out of {} days".format(corr_np.mask.sum(), len(corr_np)))
        ic = -1
    else:
        ic = np.nanmean(corr_np)
    return(ic)

    

def icir_evaluator(factor : GeneralData, shiftedPctChange:GeneralData) -> float: 
    corr_np = rowwise_corrcoef(factor, shiftedPctChange)
    if not corr_np.mask.all():
        ic = np.nanmean(corr_np)
        icir = ic / np.nanstd(corr_np)
        icir_y = icir * np.sqrt(250)
        return(icir_y)
    else:
        return(-1)  


def long_short_return_evaluator(factor : GeneralData, shiftedPctChange:GeneralData) -> float: 
    daily_return = np.ndarray(factor.generalData.shape[0])
    for i in range(factor.generalData.shape[0]):
        valid_daily_factor = np.ma.masked_invalid(factor.generalData[i])
        valid_shiftedPctChange = np.ma.masked_invalid(shiftedPctChange.generalData[i])
        mask = valid_daily_factor.mask | valid_shiftedPctChange.mask
        

        stock_count = valid_daily_factor[~mask].count()
        if stock_count%10 == 0: group_count = stock_count//10
        else: group_count = (stock_count//10) + 1
        argsort_maskedfactor = valid_daily_factor[~mask].argsort()
        group1_indice = argsort_maskedfactor[:group_count]
        group10_indice = argsort_maskedfactor[-group_count:]
        
        daily_return[i] = valid_shiftedPctChange[~mask][group10_indice].mean() - valid_shiftedPctChange[~mask][group1_indice].mean()
    return((daily_return+1).cumprod()[-1])
 

    

#%%
if __name__ == '__main__':
    from Tool import globalVars
    from Tool.logger import Logger
    from GetData import load_data
    logger = Logger(loggerFolder="", exeFileName='log')
    globalVars.initialize(logger)
    
    # load data to globalVars
    load_data("barra",
        os.path.join(os.path.join(PROJECT_ROOT,"data"), "h5")
    )
    load_data("materialData",
        os.path.join(os.path.join(PROJECT_ROOT,"data"), "h5")
    )
    globalVars.logger.info('load all......done')
    shiftedPctChange = globalVars.materialData['pctChange']
    factor = globalVars.materialData['close']

# %%
