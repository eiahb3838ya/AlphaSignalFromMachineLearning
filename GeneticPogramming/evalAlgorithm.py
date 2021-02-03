# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:34:07 2021

@author: eiahb
"""
#%%
import ray
import os
import numpy as np

from deap import gp
from time import time
from datetime import datetime

from Tool import Logger
import logging
#%%
# PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'
# # logger
# loggerFolder = PROJECT_ROOT+"Tool\\log\\"
# logger = Logger(loggerFolder, 'log')
logger = logging.getLogger()
#%% define how to evaluate

from GeneticPogramming.factorEval import residual_preprocess, MAD_preprocess, standard_scale_preprocess

# evaluate function
def preprocess_eval_single_period(individual,
                                materialDataDictID,
                                barraStackID,
                                toRegFactorStackID,
                                factorEvalFunc,
                                pset):
    ######################################################
    materialDataDict = ray.get(materialDataDictID)
    barraStack = ray.get(barraStackID)
    toRegFactorStack = ray.get(toRegFactorStackID)
    ######################################################
    
    tiic = time()
    tic = time()
    logger.debug('evaluate in pid {:6} time {} '.format(os.getpid(), str(datetime.now().time())))
    func = gp.compile(expr=individual, pset = pset)
    factor = func(**materialDataDict)
    factor.name = (str(individual))
    toc = time()
    
    #如果新因子全是空值或者都相同则拋棄
    if (~np.isfinite(factor.generalData).any() or np.nanstd(factor.generalData) == 0) : 
        logger.debug('{:6} new factor are not valid or all the same'.format(os.getpid()))
        logger.debug('{:6} used {:.5f}s to calculate the factor'.format(os.getpid(), toc-tic))
        return (-1.),

    logger.debug('{:6} used {:.5f}s to calculate the factor'.format(os.getpid(), toc-tic))
    
    # mad
    tic = time()
    factor = MAD_preprocess(factor)
    toc = time()
    logger.debug('{:6} used {:.5f}s to apply mad'.format(os.getpid(), toc-tic))
    
    # scale
    tic = time()
    factor = standard_scale_preprocess(factor)
    toc = time()
    logger.debug('{:6} used {:.5f}s to apply scale'.format(os.getpid(), toc-tic))
    
    # prepare the factor stack to reg in latter function    
    try:
        toRegStack = np.concatenate([barraStack, toRegFactorStack], axis = 2)
        logger.debug('{:6} total {} factors to regress combining stack barra and toRegFactor'.format(os.getpid(),toRegStack.shape[2]))
    except:
        logger.debug('{:6} fail to stack barra and toRegFactor'.format(os.getpid()))
        toRegStack = barraStack
  
    # get the residual after mutualize with existing factors
    if toRegStack is not None:
        tic = time()
        toScoreFactor = residual_preprocess(factor, toRegStack)
        toc = time()
        logger.debug('{:6} used {:.5f}s to apply regression to the factor'.format(os.getpid(), toc-tic))
    else:
        logger.debug('{:6} did not reg the factor'.format(os.getpid()))
        toScoreFactor = factor

    # evaluate the factor with certain way
    # typically ic, icir, factorReturn, Monotonicity(單調性)
    tic = time()
    score = factorEvalFunc(toScoreFactor)
    toc = time()
    logger.debug('{:6} used {:.5f}s to apply factorEvalFunc to the factor'.format(os.getpid(), toc-tic))
    logger.debug('{:6} used {:.5f}s to evaluate the factor'.format(os.getpid(), toc-tiic))
    if score == np.ma.masked:
        return (-1.),
    return (score),
