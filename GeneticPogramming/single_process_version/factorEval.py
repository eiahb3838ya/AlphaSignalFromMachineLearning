# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:35:48 2020

@author: eiahb
"""

# from functools import partial
import numpy as np
from copy import copy

from GeneticPogramming.utils import rowwise_corrcoef, get_residual

try:
    globalVars.logger.info('load factorEval')
except:
    from Tool import globalVars
    globalVars.logger.info('load factorEval')
    

try:
    barraStack = np.stack([aB.generalData for aB in globalVars.barra.values()],axis = 2)
except:
    from GetData import load_all
    globalVars.logger.warning('factorEval is loading things')
    load_all()
    barraStack = np.stack([aB.generalData for aB in globalVars.barra.values()],axis = 2)

# evaluate function 评价函数
def ic_evaluator(factor):    
    ic = rowwise_corrcoef(factor, globalVars.materialData['pctChange'].get_shifted(-1))
    return(ic)

def residual_preprocess(factor, toRegStack = barraStack):
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
    factor = globalVars.materialData['open']
    residual_preprocess(factor, barraStack)