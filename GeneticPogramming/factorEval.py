# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:35:48 2020

@author: eiahb
"""

# from functools import partial
# import numpy as np
from Tool import globalVars
from GeneticPogramming.utils import rowwise_corrcoef


# evaluate function 评价函数
def ic_evaluator(factor):
    ic = rowwise_corrcoef(factor, globalVars.materialData['pctChange'].get_shifted(-1))
    return(ic)
# simpleIC = partial(rowwiseCorrcoef, shiftedReturn = globalVars.materialData['pctChange'].get_shifted(-1))