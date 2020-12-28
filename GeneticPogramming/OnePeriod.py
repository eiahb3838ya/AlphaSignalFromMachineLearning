# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:08:35 2020

@author: eiahb
"""
#%% import 
import random
import sys
import itertools
import warnings


from tqdm import tqdm, trange
from deap import base, creator, gp, tools
from inspect import getmembers, isfunction
import numpy as np

from Tool import globalVars
from Tool.GeneralData import GeneralData
from GetData import load_material_data
from GeneticPogramming import scalarFunctions, singleDataFunctions, singleDataNFunctions, coupleDataFunctions

warnings.filterwarnings("ignore")
#%% load_data 
def initialize():
    globalVars.initialize()
    load_material_data()
#%% pset

def create_pset():
    # add primitives
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
    
    pset.addPrimitive(lambda x :x, [int], int, 'self_int')
    pset.addPrimitive(lambda x :x, [float], float, 'self_float')
    # add Arguments
    argDict = {'ARG{}'.format(i):argName for i, argName in enumerate(globalVars.materialData.keys()) }
    pset.renameArguments(**argDict)
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

    return(pset)
    
#%% create toolbox
def create_toolbox(pset, initGenHeightMin = 1, initGenHeightMax = 3):
    # define creator

    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', gp.PrimitiveTree, fitness = creator.FitnessMax, pset = pset) 
    
    
    # init a toolbox
    toolbox = base.Toolbox()
    
    # register the tools
    toolbox.register('expr', gp.genHalfAndHalf, pset = pset, min_=initGenHeightMin, max_ = initGenHeightMax)  
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    return(toolbox)


def create_logbook():
    # logging
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    return(stats, logbook)

def evaluate(individual, toolbox, factorEvalFunc):
    func = toolbox.compile(expr=individual)
    factor = func(**globalVars.materialData)
    score = factorEvalFunc(factor)
    return(score),

def evolution(pop, pset, toolbox, stats, logbook):
    
    # eval the population 评价初始族群
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    # start 
    for gen in trange(N_GEN, file=sys.stdout):
        # 配种选择
        # selectedTour = toolbox.tourSel(pop, N_POP) # 选择N_POP个体
        offspring = toolbox.select(pop, 2*N_POP)
        offspring = list(map(toolbox.clone, tqdm(offspring))) # 复制个体，供交叉变异用
        
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

        fitnesses = toolbox.map(toolbox.evaluate, tqdm(invalid_ind))

        for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):           
            ind.fitness.values = fit
            
        # 环境选择 - 保留精英
        pop = tools.selBest(offspring, N_POP, fit_attr='fitness') # 选择精英,保持种群规模
        # 记录数据
        record = stats.compile(pop)
        print(record)
        logbook.record(gen=gen, **record)
    
def main():
    global N_POP, TOURNSIZE, N_GEN, CXPB, MUTPB, TERMPB
    N_POP = 10 # 族群中的个体数量
    TOURNSIZE = 3
    N_GEN = 5 # 迭代代数
    CXPB = 0.6 # 交叉概率
    MUTPB = 0.3 # 突变概率
    TERMPB = 0.1 #The parameter *termpb* sets the probability to choose between a terminal or non-terminal crossover point.
    try:
        globalVars.materialData
    except AttributeError as ae:
        print(ae)
        initialize()
    pset = create_pset()
    toolbox = create_toolbox(pset, initGenHeightMin = 1, initGenHeightMax = 3)
    stats, logbook = create_logbook()
    
    from GeneticPogramming.factorEval import simple_ic
    # toolbox.register('map', pool.map)
    toolbox.register('evaluate', evaluate, toolbox = toolbox, factorEvalFunc = simple_ic)
    toolbox.register('select', tools.selTournament, tournsize = TOURNSIZE) 
    toolbox.register('crossover', gp.cxOnePointLeafBiased, termpb = TERMPB)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_ = 0, max_ = 3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    pop = toolbox.population(n = N_POP)
    evolution(pop, pset, toolbox, stats, logbook)
    
    print([str(apop) for apop in pop])
    
    
    
    
    
if __name__ == '__main__':
    main()

    
        


































