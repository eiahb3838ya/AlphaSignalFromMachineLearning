# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:49:02 2021

@author: eiahb
"""
#%% import 

import warnings
import random
from time import time
from functools import partial


from multiprocessing import Pool
from multiprocessing_logging import install_mp_handler

from deap import tools
import numpy as np



warnings.filterwarnings("ignore")


#%% set parameters 設定參數 
POOL_SIZE = 8

N_POP = 10 # 族群中的个体数量
N_GEN = 15 # 迭代代数

# prob to cross over
CXPB = 0.4 # 交叉概率

# prob to mutate
MUTPB = 0.2 # 突变概率


#%%
def easimple(toolbox, stats, logbook, evaluate, materialDataDict, barraDict, toRegFactorDict, logger):
    pop = toolbox.population(n = N_POP)
    # eval the population 评价初始族群
    # singleprocess ###################################
    # fitnesses = map(evaluate, pop)
    # for i, (ind, fit) in enumerate(zip(pop, fitnesses)):
    #     print(i, fit)
    #     ind.fitness.values = fit
    ############################################################
    # multiprocess

    logger.info('evaluating initial pop......start')
    tic = time()
    # print('start evaluating initial pop......')
    
    with Pool(processes=POOL_SIZE, initializer = install_mp_handler) as pool: 
        fitnesses = pool.map(evaluate, pop)       
        
    for i, (ind, fit) in enumerate(zip(pop, fitnesses)):
        ind.fitness.values = fit
    toc = time()
    logger.info('evaluating initial pop......done  {}'.format(toc-tic))
    record = stats.compile(pop)
    logger.info("The initial record:{}".format(str(record)))
    
    # start evolution
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
        logger.info('start evaluate for {}th Generation new individual......'.format(gen))
        tic = time()
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        with Pool(processes=POOL_SIZE, initializer = install_mp_handler) as pool: 
            fitnesses = pool.map(evaluate, pop)  
        for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
            ind.fitness.values = fit
        toc = time()
        logger.info('evaluate for {}th Generation new individual......done  {}'.format(gen, toc-tic))

        # select best 环境选择 - 保留精英
        pop = tools.selBest(offspring, N_POP, fit_attr='fitness') # 选择精英,保持种群规模
        
        # 记录数据
        record = stats.compile(pop)
        logger.info("The {} th record:{}".format(gen, str(record)))

        logbook.record(gen=gen, **record)
    return(pop)
        
#%% main
if __name__ == '__main__':
    
    from GeneticPogramming.createMultiProcessWorker import toolbox, evaluate
    from GeneticPogramming.factorEval import ic_evaluator, icir_evaluator
    from Tool import globalVars
    from GetData import load_all
    
    load_all()
    globalVars.logger.info('load all......done')
    logger = globalVars.logger
    install_mp_handler(logger = logger.logger)
    materialDataDict = globalVars.materialData
    barraDict = globalVars.barra
    toRegFactorDict = {}
    
    


    logger.info('start the easimple')
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    
    evaluateIC = partial(evaluate,
                         materialDataDict = materialDataDict,
                         barraDict = barraDict,
                         toRegFactorDict = toRegFactorDict,
                         factorEvalFunc = partial(icir_evaluator, pctChange = globalVars.materialData['pctChange'])
                         )
    
    pop = easimple(toolbox = toolbox,
                   stats = stats,
                   logbook = logbook,
                   evaluate = evaluateIC,
                   materialDataDict = materialDataDict,
                   barraDict = barraDict,
                   toRegFactorDict = toRegFactorDict,
                   logger = logger.logger
                   )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    