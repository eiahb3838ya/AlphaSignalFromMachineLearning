# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:28:07 2021

@author: eiahb
"""
#%% import 
import random
from deap import base, creator, gp, tools
from inspect import getmembers, isfunction
import itertools
import numpy as np


from Tool.GeneralData import GeneralData
# from GetData import load_all
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

N_POP = 10 # 族群中的个体数量
N_GEN = 5 # 迭代代数

# the tournsize of tourn selecetion
TOURNSIZE = 3

# prob to cross over
CXPB = 0.6 # 交叉概率

# prob to mutate
MUTPB = 0.3 # 突变概率

# The parameter *termpb* sets the probability to choose between 
# a terminal or non-terminal crossover point.
TERMPB = 0.1 

# the height min max of a initial generate 
initGenHeightMin, initGenHeightMax = 1, 3

# the height min max of a mutate sub tree
mutGenHeightMin, mutGenHeightMax = 0, 3

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
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


#%% define how to evaluate
# import how to evaluate factors
from GeneticPogramming.factorEval import ic_evaluator, residual_preprocess

factorEvalFunc = ic_evaluator

# evaluate function
def evaluate(individual,
             materialDataDict : dict,
             barraDict : dict,
             toRegFactorDict : dict,
             factorEvalFunc = factorEvalFunc,
             pset = pset):
    
    func = gp.compile(expr=individual, pset = pset)
    factor = func(**materialDataDict)
    if (~np.isfinite(factor.generalData).any() or np.nanstd(factor.generalData) == 0) : #如果新因子全是空值或者都相同则 0 
        return (-1.),
    
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
        print('fail to stack barra and toRegFactor')
        toRegStack = barraStack
    # get the residual after mutualize with existing factors
    if toRegStack is not None:
        toScoreFactor = residual_preprocess(factor, toRegStack)
    else:
        toScoreFactor = factor
    
    # evaluate the factor with certain way
    # typically ic, icir, factorReturn, Monotonicity(單調性)
    score = factorEvalFunc(toScoreFactor)
    
    if score == np.ma.masked:
        return (-1.),
    return (score),

































