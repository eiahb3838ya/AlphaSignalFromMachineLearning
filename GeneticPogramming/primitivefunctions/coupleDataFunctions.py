# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:27:20 2020

@author: Evan Hu (Yi Fan Hu)

"""


import numpy as np
from Tool.GeneralData import GeneralData
import copy
import warnings
warnings.filterwarnings("ignore")

def max_(this: GeneralData, that: GeneralData) -> GeneralData:
    assert this.generalData.shape == that.generalData.shape

    outputToReturn = copy.copy(this)
    outputToReturn.generalData = np.maximum(this.generalData, that.generalData)
    return outputToReturn

def min_(this: GeneralData, that: GeneralData) -> GeneralData:
    assert this.generalData.shape == that.generalData.shape

    outputToReturn = copy.copy(this)
    outputToReturn.generalData = np.minimum(this.generalData, that.generalData)
    return outputToReturn

# def max_(this: GeneralData, that: GeneralData) -> GeneralData:
#     assert this.generalData.shape == that.generalData.shape

#     outputToReturn = copy.copy(this)
#     outputToReturn.generalData = np.maximum(this.generalData, that.generalData)
#     return outputToReturn

# def max_(this: GeneralData, that: GeneralData) -> GeneralData:
#     assert this.generalData.shape == that.generalData.shape

#     outputToReturn = copy.copy(this)
#     outputToReturn.generalData = np.maximum(this.generalData, that.generalData)
#     return outputToReturn