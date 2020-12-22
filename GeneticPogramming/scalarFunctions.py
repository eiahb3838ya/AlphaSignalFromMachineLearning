# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:49:25 2020

@author: Evan Hu (Yi Fan Hu)

"""
import numpy as np
from Tool.GeneralData import GeneralData
import copy



def add_scalar(this: GeneralData, aNum: float = 1) -> GeneralData:
    # print(id(this))
    # print(id(this.generalData))
    outputToReturn = copy.copy(this)
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    outputToReturn.generalData = np.add(outputToReturn.generalData, aNum)
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    return outputToReturn

def subtract_scalar(this: GeneralData, aNum: float) -> GeneralData:
    # print(id(this))
    # print(id(this.generalData))
    outputToReturn = copy.copy(this)
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    outputToReturn.generalData = np.subtract(outputToReturn.generalData, aNum)
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    return outputToReturn
     
def multiply_scalar(this: GeneralData, aNum: float) -> GeneralData:
    # print(id(this))
    # print(id(this.generalData))
    outputToReturn = copy.copy(this)
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    outputToReturn.generalData = np.multiply(outputToReturn.generalData, aNum)
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    return outputToReturn

def divide_scalar(this: GeneralData, aNum: float) -> GeneralData:
    # print(id(this))
    # print(id(this.generalData))
    outputToReturn = copy.copy(this)
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    if aNum != 0:
        outputToReturn.generalData = np.divide(outputToReturn.generalData, aNum)
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    return outputToReturn

def power_scalar(this: GeneralData, aNum: float) -> GeneralData:
    # print(id(this))
    # print(id(this.generalData))
    outputToReturn = copy.copy(this)
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    outputToReturn.generalData = np.power(outputToReturn.generalData, aNum)
    # print(id(outputToReturn))
    # print(id(outputToReturn.generalData))
    return outputToReturn

# def mod_scalar(this: GeneralData, aNum: float) -> GeneralData:
#     # print(id(this))
#     # print(id(this.generalData))
#     outputToReturn = copy.copy(this)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     if aNum != 0:
#         outputToReturn.generalData = np.mod(outputToReturn.generalData, aNum)
#     # print(id(outputToReturn))
#     # print(id(outputToReturn.generalData))
#     return outputToReturn




