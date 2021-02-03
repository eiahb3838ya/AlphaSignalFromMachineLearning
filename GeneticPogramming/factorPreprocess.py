import os
import numpy as np
from copy import copy, deepcopy
from Tool.GeneralData import GeneralData
from GeneticPogramming.utils import rowwise_corrcoef, get_residual



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