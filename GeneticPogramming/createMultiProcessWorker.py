# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:28:07 2021

@author: eiahb
"""
#%% import 
import random
import os
import time
from deap import base, creator, gp, tools
from inspect import getmembers, isfunction
import itertools
import numpy as np
import logging


from Tool.GeneralData import GeneralData
from Tool.logger import Logger
from GeneticPogramming import scalarFunctions, singleDataFunctions, singleDataNFunctions, coupleDataFunctions

#%% set parameters 設定參數 
materialDataNames = [
        'close',
        'high',
        'low',
        'open',
        'preclose',
        'amount',
        'volume',
        'pctChange'
    ]

# the tournsize of tourn selecetion
TOURNSIZE = 10

# The parameter *termpb* sets the probability to choose between 
# a terminal or non-terminal crossover point.
TERMPB = 0.5

# the height min max of a initial generate 
initGenHeightMin, initGenHeightMax = 1, 5

# the height min max of a mutate sub tree
mutGenHeightMin, mutGenHeightMax = 1, 3

# logger
PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'
loggerFolder = PROJECT_ROOT+"Tool\\log\\"
logger = Logger(loggerFolder, 'log')
#%% add primitives

inputOfPset = list(itertools.repeat(GeneralData, len(materialDataNames)))
pset = gp.PrimitiveSetTyped('main', inputOfPset, GeneralData)
for aName, primitive in [o for o in getmembers(singleDataFunctions) if isfunction(o[1])]:
    # print('add primitive from {:<20}: {}'.format('singleDataFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData], GeneralData, aName)
    
for aName, primitive in [o for o in getmembers(scalarFunctions) if isfunction(o[1])]:
    # print('add primitive from {:<20}: {}'.format('scalarFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData, int], GeneralData, aName)
    
for aName, primitive in [o for o in getmembers(singleDataNFunctions) if isfunction(o[1])]:
    # print('add primitive from {:<20}: {}'.format('singleDataNFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData, int], GeneralData, aName)
    
for aName, primitive in [o for o in getmembers(coupleDataFunctions) if isfunction(o[1])]:
    # print('add primitive from {:<20}: {}'.format('coupleDataFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData, GeneralData], GeneralData, aName)

def return_self(this):
    return(this)

pset.addPrimitive(return_self, [int], int, 'self_int')
pset.addPrimitive(return_self, [float], float, 'self_float')

#%% add Arguments
argDict = {'ARG{}'.format(i):argName for i, argName in enumerate(materialDataNames)}
pset.renameArguments(**argDict)

#%% add EphemeralConstant
try:
    pset.addEphemeralConstant(name = 'EphemeralConstant_flaot',
                              ephemeral = lambda: random.uniform(-1, 1),
                              ret_type=float)
    pset.addEphemeralConstant(name = 'EphemeralConstant_int',
                              ephemeral = lambda: random.randint(1, 10),
                              ret_type = int)
except Exception as e:
    print(e)
    del gp.EphemeralConstant_flaot
    pset.addEphemeralConstant(name = 'EphemeralConstant_flaot',
                             ephemeral = lambda: random.uniform(-1, 1),
                              ret_type=float)
    del gp.EphemeralConstant_int
    pset.addEphemeralConstant(name = 'EphemeralConstant_int',
                              ephemeral = lambda: random.randint(1, 10),
                              ret_type = int)
    
    
#%% create the problem
# FitnessMax inherits gp.PrimitiveTree, and define the optimization target
creator.create('FitnessMax', base.Fitness, weights=(1.0,))

# Individual inherits gp.PrimitiveTree, and add fitness and pset as class vars
creator.create('Individual', gp.PrimitiveTree, fitness = creator.FitnessMax, pset = pset) 

# init a toolbox them
toolbox = base.Toolbox()

# Expr as our individual of our gp problem
## call gp.genHalfAndHalf with following inputs to generate a expr, now we can use toolbox.expr() to call genHalfAndHalf to easily generate a expr
toolbox.register('expr', gp.genHalfAndHalf, pset = pset, min_=initGenHeightMin, max_ = initGenHeightMax)  

## call tools.initIterate with following inputs to generate a creator.Individual
## the new func toolbox.inividual is a partial function of tools.initIterate
## the initIterate will call toolbox.expr and put it into creator.Individual which is a class inherits gp.PrimitiveTree
## that is, the output of toolbox.individual is a gp.PrimitiveTree
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)

# register a population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# registor how to compile
# toolbox.register("compile", gp.compile, pset=pset)

# register for tools to evolution 注册进化过程需要的工具：配种选择、交叉、突变
toolbox.register('select', tools.selTournament, tournsize = TOURNSIZE) 
toolbox.register('crossover', gp.cxOnePointLeafBiased, termpb = TERMPB)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_ = mutGenHeightMin, max_ = mutGenHeightMax)

def multi_mutate(individual, expr, pset):
    '''
    apply multiple kinds of mutation in a funciton
    '''
    rand = np.random.uniform(0)
    if rand <= 0.33:
        return gp.mutUniform(individual, expr, pset)
    elif rand <= 0.66:
        return gp.mutShrink(individual)
    else:
        return gp.mutNodeReplacement(individual, pset)
    
toolbox.register("mutate", multi_mutate, expr=toolbox.expr_mut, pset=pset)


#%% define how to evaluate
# import how to evaluate factors
from GeneticPogramming.factorEval import residual_preprocess



# evaluate function
def evaluate(individual,
             materialDataDict : dict,
             barraDict : dict,
             toRegFactorDict : dict,
             factorEvalFunc,
             pset = pset):
    logger.debug('evaluate in pid {}'.format(os.getpid()))
    
    tic = time.time()
    func = gp.compile(expr=individual, pset = pset)
    factor = func(**materialDataDict)
    toc = time.time()
    
    #如果新因子全是空值或者都相同则拋棄
    if (~np.isfinite(factor.generalData).any() or np.nanstd(factor.generalData) == 0) : 
        logger.debug('{} new factor are not valid or all the same'.format(os.getpid()))
        logger.debug('{} used {}s to calculate the factor'.format(os.getpid(), toc-tic))
        return (-1.),

    logger.debug('{} used {}s to calculate the factor'.format(os.getpid(), toc-tic))
    
    # prepare the factor stack to reg in latter function    
    barraStack = None
    toRegFactorStack =  None
    if len(barraDict)>0:
        barraStack = np.stack([aB.generalData for aB in barraDict.values()],axis = 2)
    if len(toRegFactorDict)>0:
        toRegFactorStack = np.stack([aB.generalData for aB in toRegFactorDict.values()],axis = 2)
    try:
        toRegStack = np.stack([barraStack, toRegFactorStack], axis = 2)
    except:
        logger.debug('{} fail to stack barra and toRegFactor'.format(os.getpid()))
        toRegStack = barraStack
        
    # get the residual after mutualize with existing factors
    if toRegStack is not None:
        tic = time.time()
        toScoreFactor = residual_preprocess(factor, toRegStack)
        toc = time.time()
        logger.debug('{} used {}s to apply regression to the factor'.format(os.getpid(), toc-tic))
    else:
        logger.debug('{} did not reg the factor'.format(os.getpid()))
        toScoreFactor = factor
    
    # evaluate the factor with certain way
    # typically ic, icir, factorReturn, Monotonicity(單調性)
    tic = time.time()
    score = factorEvalFunc(toScoreFactor)
    toc = time.time()
    logger.debug('{} used {}s to evaluate the factor'.format(os.getpid(), toc-tic))
    
    if score == np.ma.masked:
        return (-1.),
    return (score),

































