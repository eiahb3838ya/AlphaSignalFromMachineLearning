# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 00:04:44 2020

@author: Evan Hu (Yi Fan Hu)

"""
#%% import 
import random
import time
from tqdm import tqdm, trange
from deap import base,creator,gp,tools
from inspect import getmembers, isfunction
import numpy as np

from Tool import globalVars
from Tool.GeneralData import GeneralData
from GetData.loadData import loadData
from GeneticPogramming import scalarFunctions, singleDataFunctions, singleDataNFunctions, coupleDataFunctions

#%% initialize global vars
globalVars.initialize()
loadData()

#%% add primitives
pset = gp.PrimitiveSetTyped('main', [GeneralData,GeneralData,GeneralData,GeneralData,GeneralData,GeneralData], GeneralData)
for aName, primitive in [o for o in getmembers(singleDataFunctions) if isfunction(o[1])]:
    print('add primitive from {:<20}: {}'.format('singleDataFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData], GeneralData, aName)
    
for aName, primitive in [o for o in getmembers(scalarFunctions) if isfunction(o[1])]:
    print('add primitive from {:<20}: {}'.format('scalarFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData, float], GeneralData, aName)
    
for aName, primitive in [o for o in getmembers(singleDataNFunctions) if isfunction(o[1])]:
    print('add primitive from {:<20}: {}'.format('singleDataNFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData, int], GeneralData, aName)
    
for aName, primitive in [o for o in getmembers(coupleDataFunctions) if isfunction(o[1])]:
    print('add primitive from {:<20}: {}'.format('coupleDataFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData, GeneralData], GeneralData, aName)
    
def return_self(this):
    return(this)

pset.addPrimitive(return_self, [int], int, 'self_int')
pset.addPrimitive(return_self, [float], float, 'self_float')

#%% add Arguments
pset.renameArguments(ARG0 = 'CLOSE')
pset.renameArguments(ARG1 = 'OPEN')
pset.renameArguments(ARG2 = 'HIGH')
pset.renameArguments(ARG3 = 'LOW')
pset.renameArguments(ARG4 = 'AMOUNT')
pset.renameArguments(ARG5 = 'VOLUME')

#%% add EphemeralConstant

pset.addEphemeralConstant(name = 'EphemeralConstant_flaot',
                         ephemeral = lambda: random.uniform(-10, 10),
                         ret_type=float)
pset.addEphemeralConstant(name = 'EphemeralConstant_int',
                         ephemeral = lambda: random.randint(1, 180),
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
toolbox.register('expr', gp.genFull, pset = pset, min_=1, max_ = 3)  

## call tools.initIterate with following inputs to generate a creator.Individual
## the new func toolbox.inividual is a partial function of tools.initIterate
## the initIterate will call toolbox.expr and put it into creator.Individual which is a class inherits gp.PrimitiveTree
## that is, the output of toolbox.individual is a gp.PrimitiveTree
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)

# register a population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# evaluate function 评价函数
## a simple corrcoef func that calculate the corrcoef through out the valid data
def simpleCorrcoef(factor, shiftedReturn):
    validFactor = np.ma.masked_invalid(factor.generalData)
    validShiftedReturn = np.ma.masked_invalid(shiftedReturn.generalData)
    msk = (~validFactor.mask & ~validShiftedReturn.mask)
    corrcoef = np.ma.corrcoef(validFactor[msk],validShiftedReturn[msk])
    return(corrcoef[0, 1])


def evaluate(individual):
    func = toolbox.compile(expr=individual)
    factor = func(globalVars.adj_close,\
                  globalVars.adj_open,\
                  globalVars.adj_high,\
                  globalVars.adj_low,\
                  globalVars.amount,\
                  globalVars.volume\
                  )
    shiftedReturn = globalVars.pctChange.get_shifted(-1)
    score = simpleCorrcoef(factor, shiftedReturn)
    return(score),
    
    

    
toolbox.register('evaluate', evaluate)
#%% generate the population 

# start with generate initial population
# 生成初始族群
N_POP = 10 # 族群中的个体数量
pop = toolbox.population(n = N_POP)

# eval the population
# 评价初始族群
fitnesses = map(toolbox.evaluate, pop)
for ind, fit in tqdm(zip(pop, fitnesses)):
    ind.fitness.values = fit
    
#%%  preparation for evolution iteration 

N_GEN = 5 # 迭代代数
CXPB = 0.5 # 交叉概率
MUTPB = 0.2 # 突变概率

# register for tools to evolution
# 注册进化过程需要的工具：配种选择、交叉、突变
toolbox.register('tourSel', tools.selTournament, tournsize = 2) # 注册Tournsize为2的锦标赛选择
toolbox.register('crossover', gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_ = 0, max_ = 2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# logging
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
logbook = tools.Logbook()

#%% algorthm iteration 

for gen in trange(N_GEN):
    
    # 配种选择
    selectedTour = toolbox.tourSel(pop, N_POP) # 选择N_POP个体
    selectedInd = list(map(toolbox.clone, selectedTour)) # 复制个体，供交叉变异用
    
    # 对选出的育种族群两两进行交叉，对于被改变个体，删除其适应度值
    for child1, child2 in zip(selectedInd[::2], selectedInd[1::2]):
        if random.random() < CXPB:
            toolbox.crossover(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
      
    # 对选出的育种族群进行变异，对于被改变个体，删除适应度值
    for mutant in selectedInd:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
      
    # 对于被改变的个体，重新评价其适应度
    invalid_ind = [ind for ind in selectedInd if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        print(fit)
        ind.fitness.values = fit
    
    # 完全重插入
    pop[:] = selectedInd
    
    # 记录数据
    record = stats.compile(pop)
    logbook.record(gen=gen, **record)


#%%

crossover = gp.cxOnePointLeafBiased


for child1, child2 in zip(selectedInd[::2], selectedInd[1::2]):
        if random.random() < CXPB:
            crossover(child1, child2, 0.1)
            del child1.fitness.values
            del child2.fitness.values









































