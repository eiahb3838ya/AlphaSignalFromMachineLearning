# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:07:25 2020

@author: Evan Hu (Yi Fan Hu)

"""
import numpy as np
from numpy.lib import stride_tricks

def get_strided(toStride2DArray, rollingDaysN):
    nrows, ncols = toStride2DArray.shape
    nan2DArray = np.full((rollingDaysN - 1, ncols), np.nan)
    stacked2DArray = np.vstack((nan2DArray, toStride2DArray))
    axis0Stride, axis1Stride = stacked2DArray.strides
    shape = (nrows, rollingDaysN, ncols)
    strides = (axis0Stride, axis0Stride, axis1Stride)
    strided = stride_tricks.as_strided(stacked2DArray, shape=shape, strides=strides)
    return(strided)

def get_maskAllNaN(toMask3DArray):
    m = np.ma.masked_invalid(toMask3DArray)
    isAllNaN = m.mask.all(axis=1)
    return(m, isAllNaN)

## a simple corrcoef func that calculate the corrcoef through out the valid data
def simple_corrcoef(factor, shiftedReturn):
    validFactor = np.ma.masked_invalid(factor.generalData)
    validShiftedReturn = np.ma.masked_invalid(shiftedReturn.generalData)
    msk = (~validFactor.mask & ~validShiftedReturn.mask)
    corrcoef = np.ma.corrcoef(validFactor[msk],validShiftedReturn[msk])
    if corrcoef.mask[0, 1]:
        return(0)
    else:
        return(corrcoef[0, 1])
    
def rowwise_corrcoef(factor, shiftedReturn):
    validFactor = np.ma.masked_invalid(factor.generalData)
    validShiftedReturn = np.ma.masked_invalid(shiftedReturn.generalData)
    msk = np.ma.mask_or(validFactor.mask, validShiftedReturn.mask)#(~validFactor.mask & ~validShiftedReturn.mask)
    validFactor.mask = msk
    validShiftedReturn.mask = msk
    # get corrcoef of each rowwise pair
    #======================================================
    # A_mA = A - A.mean(1)[:, None]
    # B_mB = B - B.mean(1)[:, None]

    # # Sum of squares across rows
    # ssA = (A_mA**2).sum(1)
    # ssB = (B_mB**2).sum(1)

    # # Finally get corr coeff
    # return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
    
    #=======================================================
    
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    validFactor_m = validFactor - validFactor.mean(1)[:, None]
    validShiftedReturn_m = validShiftedReturn - validShiftedReturn.mean(1)[:, None]
    
    # Sum of squares across rows
    ssA = (validFactor_m**2).sum(1)
    ssB = (validShiftedReturn_m**2).sum(1)
    
    toDivide = np.ma.dot(validFactor_m, validShiftedReturn_m.T).diagonal()
    divider = np.ma.sqrt(np.dot(ssA, ssB))
    return((toDivide/divider).mean())

# 矩阵求解最主要的问题是数据中不能有空值，时间可以很快
def linearRegLsq(x,y):
    '''最小二乘法直接求解回归系数'''
    xtx = np.dot(x.T, x)
    if np.linalg.det(xtx) == 0.0: # 判断xtx行列式是否等于0，奇异矩阵不能求逆
        #print('Can not resolve the problem')
        return None
    theta_lsq = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
    return theta_lsq

def add_constant(x):
    return np.c_[x,np.ones(x.shape[0])]



if __name__ == '__main__':
    x = np.arange(36).reshape((6, 6))
    print(x)
    print(x.shape, x.strides)
    strided = get_strided(x, 3)
    print(strided)
    print(strided.shape, strided.strides)
    
