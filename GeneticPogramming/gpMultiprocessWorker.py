# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 00:04:44 2020

@author: Evan Hu (Yi Fan Hu)

"""
#%% import 
import random
from deap import base, creator, gp, tools
from inspect import getmembers, isfunction
import itertools
import numpy as np

from Tool import globalVars
from Tool.GeneralData import GeneralData
from GetData import load_material_data
from GeneticPogramming import scalarFunctions, singleDataFunctions, singleDataNFunctions, coupleDataFunctions

#%% set parameters 設定參數 
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

#%% initialize global vars
# load_data 
def initialize():
    globalVars.initialize()
    load_material_data() 
#%% add primitives
try:
    inputOfPset = list(itertools.repeat(GeneralData, len(globalVars.materialData.keys())))  
except AttributeError as ae:
    print(ae)
    initialize()
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
from GeneticPogramming.factorEval import ic_evaluator

# evaluate function
def evaluate(individual, factorEvalFunc = ic_evaluator, pset = pset):
    func = gp.compile(expr=individual, pset = pset)
    factor = func(**globalVars.materialData)
    
    if (~np.isfinite(factor.generalData).any() or np.nanstd(factor.generalData) == 0) : #如果新因子全是空值或者都相同则 0 
        return (-1.),
    
    score = factorEvalFunc(factor)
    if score == np.ma.masked:
        return (-1.),
    return (score),

#### not useful if we use Pool to do multiprocessing

# def evaluate_worker(toEvaluateInds, factorEvalFunc , pset, workerId = None):  
#     fitnesses =  [evaluate(aInd, factorEvalFunc, pset) for aInd in toEvaluateInds]
#     for ind, fit in zip(toEvaluateInds, fitnesses):
#         ind.fitness.values = fit
    
#### not useful if we use Pool to do multiprocessing

# def get_job_length_per_worker(lenPop, nWorkers):
#     if lenPop % nWorkers  == 0:
#         single_length = lenPop // nWorkers 
#     else:
#         single_length = (lenPop // (nWorkers))+1
#     return(single_length)

#%%
# from  multiprocessing import Pool
if __name__ == '__main__':
    
    initialize()
    # logging
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    
    pop = toolbox.population(n = N_POP)
    
    # eval the population 评价初始族群
    ########## single process 
    fitnesses = map(evaluate, pop)
    for i, (ind, fit) in enumerate(zip(pop, fitnesses)):
        print(i, fit)
        ind.fitness.values = fit
        
    # start 
    for gen in range(N_GEN):
        # 配种选择
        offspring = toolbox.select(pop, 2*N_POP)
        offspring = list(map(toolbox.clone, offspring)) # 复制个体，供交叉变异用
        
        # 对选出的育种族群两两进行交叉，对于被改变个体，删除其适应度值
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.crossover(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        # 对选出的育种族群进行变异，对于被改变个体，删除适应度值
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
          
        # 对于被改变的个体，重新评价其适应度
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        ########## single process 
        fitnesses = map(evaluate, invalid_ind)
        for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):   
            print(i, fit)
            ind.fitness.values = fit
        

        # 环境选择 - 保留精英
        pop = tools.selBest(offspring, N_POP, fit_attr='fitness') # 选择精英,保持种群规模
        # pop[:] = selectedInd
        
        # 记录数据
        record = stats.compile(pop)
        print(record)
        logbook.record(gen=gen, **record)
    














































