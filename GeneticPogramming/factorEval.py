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
from GeneticPogramming.utils import rowwise_corrcoef, get_residual, save_factor



# evaluate function 评价函数
def ic_evaluator(factor : GeneralData, shiftedPctChange:GeneralData) -> float:   
    corr_np = rowwise_corrcoef(factor, shiftedPctChange)
    # print("fail eval {} of {} days".format(corr_np.mask.sum(), shiftedPctChange.generalData.shape[0]))
    if corr_np.mask.sum() > len(corr_np)/2:
        try:
            save_factor(factor,"./factors/troubleFactors")
        except FileNotFoundError as fnfe:
            print(fnfe)
        except Exception as e:
            print(e)
            raise e
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

def residual_preprocess(factor, toRegStack):
    assert toRegStack.shape[:2] == factor.generalData.shape, 'make sure the factor shape is same as the risk factors'
    residualStack = np.full_like(factor.generalData, np.nan)
    thetaStack = np.full((toRegStack.shape[0],toRegStack.shape[2]), np.nan)
    for i in range(factor.generalData.shape[0]):
        residual, theta = get_residual(toRegStack[i], factor.generalData[i])
        residualStack[i] = residual
        thetaStack[i] = theta
    outFactor = copy(factor)
    outFactor.name = factor.name+"_residual"
    outFactor.metadata['theta'] = thetaStack
    outFactor.generalData = residualStack
    return(outFactor)


def MAD_preprocess(factor, multiple = 3):
    try:
        if isinstance(factor, GeneralData):X = (factor.generalData).T
        else:X = (factor).T  
    except Exception as e:
        raise(e, 'The input must be np.array or GeneralData either one')
        
    X_ = deepcopy(X)  
    mean_ = np.nanmean(X_, axis = 0)

    distance_to_mean = np.abs(X_ - mean_)
    
    
    median_of_distance = np.nanmedian(distance_to_mean, axis = 0)

    upper_limit = mean_ + multiple * median_of_distance  # upper bound
    lower_limit = mean_ - multiple * median_of_distance  # lower bound

    X_ = np.where(X_>upper_limit, upper_limit, X_)
    X_ = np.where(X_<lower_limit, lower_limit, X_)
    outFactor = copy(factor)
    outFactor.name = factor.name+"_MAD"
    outFactor.metadata['MAD'] = median_of_distance
    outFactor.generalData = X_.T
    return(outFactor)

def standard_scale_preprocess(factor):
    try:
        if isinstance(factor, GeneralData):X = (factor.generalData).T
        else:X = (factor).T  
    except Exception as e:
        raise(e, 'The input must be np.array or GeneralData either one')
        
    X_ = deepcopy(X)  
    mean_ = np.nanmean(X_, axis = 0)
    std_ = np.nanstd(X_, axis = 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        X_scaled = (X_-mean_)/std_
        X_scaled = np.where(np.isinf(X_scaled), 1, X_scaled)
        X_scaled = np.where((X_-mean_) == 0, 0, X_scaled)
    outFactor = copy(factor)
    outFactor.name = factor.name+"_scaled"
    outFactor.generalData = X_scaled.T
    return(outFactor)


    
    
    
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