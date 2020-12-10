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

if __name__ == '__main__':
    x = np.arange(36).reshape((6, 6))
    print(x)
    print(x.shape, x.strides)
    strided = get_strided(x, 3)
    print(strided)
    print(strided.shape, strided.strides)
    
