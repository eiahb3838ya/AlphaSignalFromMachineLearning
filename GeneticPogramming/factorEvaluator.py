# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:35:48 2020

@author: eiahb
"""

# from functools import partial
import os

import numpy as np
from copy import copy, deepcopy
from Tool.GeneralData import GeneralData
from GeneticPogramming.utils import rowwise_corrcoef



# evaluate function 评价函数
def ic_evaluator(factor : GeneralData, shiftedPctChange:GeneralData) -> float:   
    corr_np = rowwise_corrcoef(factor, shiftedPctChange)
    # print("fail eval {} of {} days".format(corr_np.mask.sum(), shiftedPctChange.generalData.shape[0]))
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




    
    
    
if __name__  ==  '__main__':
    from Tool import globalVars
    try:
        barraStack = np.stack([aB.generalData for aB in globalVars.barra.values()],axis = 2)
    except:
        from GetData import load_all
        globalVars.logger.warning('factorEval is loading things')
        load_all()
        barraStack = np.stack([aB.generalData for aB in globalVars.barra.values()],axis = 2)
    factor = globalVars.materialData['open']
    residual_preprocess(factor, barraStack)