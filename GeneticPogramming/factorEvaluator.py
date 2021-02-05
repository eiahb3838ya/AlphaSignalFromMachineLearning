# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:35:48 2020

在這裡定義所有對單因子評價的方式，輸入的因子是已經輸出成 GeneralData 的因子值
而後面也可以使用其他的數據， typically 就是用收益率做 ic, icir 等等
回傳一個數值

@author: eiahb
"""


import os

import numpy as np
from copy import copy, deepcopy
from Tool.GeneralData import GeneralData
from GeneticPogramming.utils import rowwise_corrcoef



# evaluate function 评价函数
def ic_evaluator(factor : GeneralData, shiftedPctChange:GeneralData) -> float:   
    corr_np = rowwise_corrcoef(np.argsort(factor.generalData, axis = 1), np.argsort(shiftedPctChange.generalData, axis = 1))
    # print("fail eval {} of {} days".format(corr_np.mask.sum(), shiftedPctChange.generalData.shape[0]))
    if corr_np.mask.sum() > len(corr_np)/2:
        print("trouble factor with unevaluate days {} out of {} days".format(corr_np.mask.sum(), len(corr_np)))
        ic = -1
    else:
        ic = np.nanmean(corr_np)
    return(ic)

# 
def rankic_evaluator(factor : GeneralData, shiftedPctChange:GeneralData) -> float:   
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

