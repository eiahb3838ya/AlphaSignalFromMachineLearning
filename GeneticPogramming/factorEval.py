# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:35:48 2020

@author: eiahb
"""

# from functools import partial
import numpy as np
from copy import copy
from Tool.GeneralData import GeneralData
from GeneticPogramming.utils import rowwise_corrcoef, get_residual   



# evaluate function 评价函数
def ic_evaluator(factor : GeneralData, pctChange:GeneralData) -> float:    
    ic = rowwise_corrcoef(factor, pctChange.get_shifted(-1)).mean()
    return(ic)

def icir_evaluator(factor : GeneralData, pctChange:GeneralData) -> float:    
    corr = rowwise_corrcoef(factor, pctChange.get_shifted(-1))
    ic = corr.mean()
    ir = ic / corr.std()
    return(ir)

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