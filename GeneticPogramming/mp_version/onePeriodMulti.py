# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:49:02 2021

@author: eiahb
"""
#%% import 

import warnings
import random
import os

from time import time
from functools import partial
# from multiprocessing import Pool
from ray.util.multiprocessing import Pool

from deap import tools

import numpy as np

from GeneticPogramming.factorEval import ic_evaluator, icir_evaluator
from GeneticPogramming.utils import save_factor
from Tool.GeneralData import GeneralData
warnings.filterwarnings("ignore")
#%% set parameters 設定參數
ITERTIMES = 30
PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'
USEFUL_FACTOR_RECORD_PATH = os.path.join(PROJECT_ROOT, "GeneticPogramming\\factors\\usefulFactors")
BEST_FACTOR_RECORD_PATH = os.path.join(PROJECT_ROOT, "GeneticPogramming\\factors\\bestFactors")

POOL_SIZE = 8

N_POP = 100 # 族群中的个体数量
N_GEN = 7 # 迭代代数

# prob to cross over
CXPB = 0.6 # 交叉概率

# prob to mutate
MUTPB = 0.2 # 突变概率

EVALUATE_FUNC = ic_evaluator
#%% easimple
def easimple(toolbox, stats, logbook, evaluate, logger):
    pop = toolbox.population(n = N_POP)

    tic = time()
    logger.info('start easimple at {:.2f}'.format(tic))
    logger.info('evaluating initial pop......start')
    
    # with Pool(processes=POOL_SIZE) as pool: 
    #     fitnesses = pool.map(evaluate, pop)    
    fitnesses = toolbox.map(evaluate, pop)    
        
    for i, (ind, fit) in enumerate(zip(pop, fitnesses)):
        ind.fitness.values = fit
    toc = time()
    logger.info('evaluating initial pop......done with {:.5f} sec'.format(toc-tic))
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

        
        tic = time()
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        logger.info('start evaluate for {}th Generation new {} individuals......'.format(gen, len(invalid_ind)))

        # with Pool(processes=POOL_SIZE) as pool: 
        #     fitnesses = pool.map(evaluate, invalid_ind)  
        fitnesses = toolbox.map(evaluate, invalid_ind)   
        for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
            ind.fitness.values = fit
            if(fit[0]>0.03):
                # get something useful
                logger.info('got a expr useful in gen:{}, end gp algorithm'.format(gen))
                return(True, ind, logbook)

        toc = time()
        logger.info('evaluate for {}th Generation new individual......done with {:.5f} sec'.format(gen, toc-tic))
        
        # select best 环境选择 - 保留精英
        pop = tools.selBest(offspring, N_POP, fit_attr='fitness') # 选择精英,保持种群规模
        
        
        # 记录数据
        record = stats.compile(pop)
        logger.info("The {} th record:{}".format(gen, str(record)))
        logbook.record(gen=gen, **record)
    ind = tools.selBest(offspring, 1, fit_attr='fitness')[0]
    logger.info('none expr useful, terminate easimple')
    logger.info('end easimple {:.2f}'.format(tic))
    return(False, ind, logbook)
#%% define main
startEvaluateDate = "2017-01-01"
endEvaluateDate = "2018-01-01"

if __name__ == '__main__':
    from GeneticPogramming.createMultiProcessWorker import toolbox, evaluate, materialDataNames, stats, logbook, compileFactor, creator, pset
    from Tool import globalVars, Logger
    from GetData import load_material_data, load_barra_data, align_all_to
    from GeneticPogramming.tryRay import ray_deap_map
    import ray
    
    # set up logger
    loggerFolder = PROJECT_ROOT+"Tool\\log\\"
    logger = Logger(loggerFolder, 'log')
    globalVars.initialize(logger)
    
    # load data to globalVars
    load_material_data() 
    load_barra_data()
    globalVars.logger.info('load all......done')
    
    # prepare data
    materialDataDict = {k:globalVars.materialData[k] for k in materialDataNames} # only take the data specified in materialDataNames
    barraDict = globalVars.barra
    toRegFactorDict = {}
    
    # get the return to compare 
    open_ = globalVars.materialData['open']
    shiftedPctChange_df = open_.to_DataFrame().pct_change().shift(-2)
        
    # align data within 2Y
    shiftedPctChange2Y_df = shiftedPctChange_df.loc[startEvaluateDate:endEvaluateDate]
    shiftedPctChange2Y = GeneralData('shiftedPctChange2Y', shiftedPctChange2Y_df)
    materialDataDict2Y = align_all_to(materialDataDict, shiftedPctChange2Y)
    barraDict2Y = align_all_to(barraDict, shiftedPctChange2Y)
    del shiftedPctChange_df, shiftedPctChange2Y_df
        
    # stack barra data
    barraStack = None
    toRegFactorStack =  None
    if len(barraDict)>0:
        barraStack = np.stack([aB.generalData for aB in barraDict2Y.values()],axis = 2)
    if len(toRegFactorDict)>0:
        toRegFactorStack = np.stack([aB.generalData for aB in toRegFactorDict.values()],axis = 2)

    evaluate2Y = partial(
        evaluate,
        materialDataDict = materialDataDict2Y,
        barraStack = barraStack,
        toRegFactorStack = toRegFactorStack,
        factorEvalFunc = partial(EVALUATE_FUNC, shiftedPctChange = shiftedPctChange2Y)
    )
        
    #%% map

    ray.init(num_cpus=POOL_SIZE)
    toolbox.register("map", ray_deap_map, creator_setup=creator, pset_creator=pset)
    #%%
    for i in range(ITERTIMES):
        logger.info("start easimple algorithm from iteration {}th time".format(i+1))

        findFactor, returnIndividual, logbook = easimple(
                                        toolbox = toolbox,
                                        stats = stats,
                                        logbook = logbook,
                                        evaluate = evaluate2Y,
                                        logger = globalVars.logger
                                    )
        if findFactor:
            func, factor = compileFactor(individual = returnIndividual, materialDataDict = materialDataDict2Y)
            factor.name = str(returnIndividual)
            save_factor(factor, USEFUL_FACTOR_RECORD_PATH)
            toRegFactorDict.update({str(returnIndividual):factor})
            if len(toRegFactorDict)>0:
                toRegFactorStack = np.stack([aB.generalData for aB in toRegFactorDict.values()],axis = 2)
        
            evaluate2Y = partial(
                evaluate,
                materialDataDict = materialDataDict2Y,
                barraStack = barraStack,
                toRegFactorStack = toRegFactorStack,
                factorEvalFunc = partial(EVALUATE_FUNC, shiftedPctChange = shiftedPctChange2Y)
            )
                
        else:
            func, factor = compileFactor(individual = returnIndividual, materialDataDict = materialDataDict2Y)
            factor.name = str(returnIndividual)
            save_factor(factor, BEST_FACTOR_RECORD_PATH)
            logger.info("end easimple algorithm from iteration {}th time".format(i+1))


    
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    