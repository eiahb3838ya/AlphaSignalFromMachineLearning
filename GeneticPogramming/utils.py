# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:07:25 2020

@author: Evan Hu (Yi Fan Hu)

"""
import numpy as np
from numpy.lib import stride_tricks
from numpy.ma import masked_invalid



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
def linear_regression(x,y):
    '''
    get the theta in of OLS regression 
    the function doesn't deal with nan in the matrix
    make sure all value is valid and the shape is correct
    最小二乘法直接求解回归系数
    '''
    xtx = np.dot(x.T, x)
    if np.linalg.det(xtx) == 0.0: # 判断xtx行列式是否等于0，奇异矩阵不能求逆
        print('the det(xtx) is 0')
        return np.full(x.shape[0], np.nan)
    tmp = np.dot(np.linalg.inv(xtx), x.T)
    try:
        theta = np.dot(tmp, y)
    except ValueError as ve:
        print('''
              plz check the shape of x and y in linear regression
              make sure the shape of x is (y.shape[0] , count of featrues)
              ''')
        print(ve)
        raise ve
        
    return theta

def get_residual(x, y):
    '''
    get the residual of y with OLS regression 
    this function checks the nan values and the theta ignore the datapoint that contain any nans
    but the result will show nan if any nan in x or y for that data point
    make sure all value is valid and the shape is correct
    '''
    masked_x = masked_invalid(x)
    masked_y = masked_invalid(y)
    mask_stocks = np.ma.mask_or(masked_x.mask.any(axis = 1), masked_y.mask)
    if mask_stocks.all():
        print('all data points are invalid on the time')
        return(np.full_like(y, np.nan), np.full(x.shape[1], np.nan))
    theta = linear_regression(x[~mask_stocks, :],y[~mask_stocks])
    residual = y - np.dot(x, theta)
    return(residual, theta)

def add_constant(x):
    return np.c_[x,np.ones(x.shape[0])]



if __name__ == '__main__':
    b1 = np.random.uniform(size = (70)).reshape((5, 14))
    b2 = np.random.uniform(size = (70)).reshape((5, 14))
    f = np.random.uniform(size = (70)).reshape((5, 14))
    bs = np.stack((b1, b2), axis = 2)
    x = b1
    print(x)
    print(x.shape, x.strides)
    strided = get_strided(x, 3)
    print(strided)
    print(strided.shape, strided.strides)
    
