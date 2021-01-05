# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 00:04:44 2020

@author: Evan Hu (Yi Fan Hu)

"""
#%% import 
import random
import sys
import time
from tqdm import tqdm, trange
from deap import base, creator, gp, tools
from inspect import getmembers, isfunction
import itertools
import numpy as np

sys.path.append('.')
from Tool import globalVars
from Tool.GeneralData import GeneralData
from GetData import load_material_data
from GeneticPogramming import scalarFunctions, singleDataFunctions, singleDataNFunctions, coupleDataFunctions

#%% initialize global vars
globalVars.initialize()
loadedData = load_material_data()

#%% add primitives
inputOfPset = list(itertools.repeat(GeneralData, len(globalVars.materialData.keys())))

pset = gp.PrimitiveSetTyped('main', inputOfPset, GeneralData)
for aName, primitive in [o for o in getmembers(singleDataFunctions) if isfunction(o[1])]:
    print('add primitive from {:<20}: {}'.format('singleDataFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData], GeneralData, aName)
    
for aName, primitive in [o for o in getmembers(scalarFunctions) if isfunction(o[1])]:
    print('add primitive from {:<20}: {}'.format('scalarFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData, int], GeneralData, aName)
    
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
argDict = {'ARG{}'.format(i):argName for i, argName in enumerate(globalVars.materialData.keys()) }
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
    print('EphemeralConstant already in global')

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


#%% define how to evaluate

# evaluate function 评价函数
## a simple corrcoef func that calculate the corrcoef through out the valid data
def simpleCorrcoef(factor, shiftedReturn):
    validFactor = np.ma.masked_invalid(factor.generalData)
    validShiftedReturn = np.ma.masked_invalid(shiftedReturn.generalData)
    msk = (~validFactor.mask & ~validShiftedReturn.mask)
    corrcoef = np.ma.corrcoef(validFactor[msk],validShiftedReturn[msk])
    if corrcoef.mask[0, 1]:
        return(0)
    else:
        return(corrcoef[0, 1])
    
def simpleIC(factor, shiftedReturn = globalVars.materialData['pctChange'].get_shifted(-1)):
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

def evaluate(individual):
    func = toolbox.compile(expr=individual)
    factor = func(**globalVars.materialData)
    score = simpleIC(factor)
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
    
#%%    
# import multiprocessing

# nbCpu = multiprocessing.cpu_count()
# pool = multiprocessing.Pool(processes = 2) # 调用所有CPU
# # fitnesses = pool.map(toolbox.evaluate, pop)
#%%  preparation for evolution iteration 

N_GEN = 5 # 迭代代数
CXPB = 0.5 # 交叉概率
MUTPB = 0.3 # 突变概率
TERMPB = 0.3 #The parameter *termpb* sets the probability to choose between a terminal or non-terminal crossover point.

# register for tools to evolution
# 注册进化过程需要的工具：配种选择、交叉、突变
toolbox.register('tourSel', tools.selTournament, tournsize = 5) # 注册Tournsize为2的锦标赛选择
toolbox.register('crossover', gp.cxOnePointLeafBiased, termpb = TERMPB)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_ = 0, max_ = 3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# logging
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
logbook = tools.Logbook()

#%% algorthm iteration 

for gen in trange(N_GEN, file=sys.stdout):
    
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
    for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
        print('The {}th ind scored {} in fitness'.format(i, fit[0]))
        ind.fitness.values = fit
    
    # 完全重插入
    pop[:] = selectedInd
    
    # 记录数据
    record = stats.compile(pop)
    logbook.record(gen=gen, **record)












































