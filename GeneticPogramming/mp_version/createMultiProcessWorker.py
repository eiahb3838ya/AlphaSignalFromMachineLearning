# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:28:07 2021

@author: eiahb
"""
#%% import 
import random
import os
from time import time
from deap import base, creator, gp, tools
from inspect import getmembers, isfunction
import itertools
import numpy as np

from Tool.GeneralData import GeneralData
from Tool.logger import Logger
from GeneticPogramming import scalarFunctions, singleDataFunctions, singleDataNFunctions, coupleDataFunctions

#%% set parameters 設定參數 
PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'

materialDataNames = [
        'close',
        'high',
        'low',
        'open',
        # 'preclose',
        'amount',
        'volume',
        'pctChange'
    ]

# the tournsize of tourn selecetion
TOURNSIZE = 3

# The parameter *termpb* sets the probability to choose between 
# a terminal or non-terminal crossover point.
TERMPB = 0.1

# the height min max of a initial generate 
initGenHeightMin, initGenHeightMax = 2, 4

# the height min max of a mutate sub tree
mutGenHeightMin, mutGenHeightMax = 1, 2

# logger
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
    if rand <= 0.4:
        return gp.mutUniform(individual, expr, pset)
    elif rand <= 0.75:
        return gp.mutShrink(individual)
    else:
        return gp.mutNodeReplacement(individual, pset)
    
toolbox.register("mutate", multi_mutate, expr=toolbox.expr_mut, pset=pset)


#%% define how to evaluate
# import how to evaluate factors
from GeneticPogramming.factorEval import residual_preprocess, MAD_preprocess, standard_scale_preprocess

def compileFactor(individual, materialDataDict, pset = pset):
    func = gp.compile(expr=individual, pset = pset)
    factor = func(**materialDataDict)
    return(func, factor)

# evaluate function
def evaluate(individual,
             materialDataDict : dict,
             barraStack,
             toRegFactorStack,
             factorEvalFunc,
             pset = pset):
    tiic = time()
    tic = time()
    logger.debug('evaluate in pid {:6} time {:.5f} '.format(os.getpid(), tic))
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

#%% define stat and logbook
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
logbook = tools.Logbook()
#%% main for test
if __name__ == '__main__':
    from GeneticPogramming.factorEval import ic_evaluator, icir_evaluator
    from functools import partial
    
    #%% set parameters 設定參數 
    PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'
    
    POOL_SIZE = 6
    
    N_POP = 100 # 族群中的个体数量
    N_GEN = 5 # 迭代代数
    
    # prob to cross over
    CXPB = 0.4 # 交叉概率
    
    # prob to mutate
    MUTPB = 0.1 # 突变概率
    
    EVALUATE_FUNC = ic_evaluator
#%% easiple for test
    def easimple(toolbox, stats, logbook, evaluate, logger):
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
        logger.info('time in parent is {}'.format(tic))
        
        fitnesses = map(evaluate, pop)     
            
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
    
            
            tic = time()
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            logger.info('start evaluate for {}th Generation new {} individuals......'.format(gen, len(invalid_ind)))
    
            
            fitnesses = map(evaluate, invalid_ind)  
                
            for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
                ind.fitness.values = fit
                if(fit[0]>0.01):
                    # get something useful
                    logger.info('got a expr useful in gen:{} ,end gp algorithm'.format(gen))
                    return(True, ind, logbook)
            toc = time()
            logger.info('evaluate for {}th Generation new individual......done  {}'.format(gen, toc-tic))
            
            # select best 环境选择 - 保留精英
            pop = tools.selBest(offspring, N_POP, fit_attr='fitness') # 选择精英,保持种群规模
            
            # 记录数据
            record = stats.compile(pop)
            logger.info("The {} th record:{}".format(gen, str(record)))
    
            logbook.record(gen=gen, **record)
        return(False, pop, logbook)
    
    from GeneticPogramming.createMultiProcessWorker import toolbox, evaluate, materialDataNames, stats, logbook, compileFactor
    from Tool import globalVars, Logger
    from GetData import load_material_data, load_barra_data, align_all_to
    
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
    startEvaluateDate = "2017-01-01";endEvaluateDate = "2019-01-01"
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
    
    

    
    globalVars.logger.info('start the easimple')
    
    
    ITERTIMES = 30
    PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'
    FACTOR_RECORD_PATH = os.path.join(PROJECT_ROOT, "GeneticPogramming\\factors\\factorRecord.txt")
    for i in range(ITERTIMES):
        logger.info("start easimple algorithm {} time".format(i+1))

        findFactor, pop, logbook = easimple(
                                        toolbox = toolbox,
                                        stats = stats,
                                        logbook = logbook,
                                        evaluate = evaluate2Y,
                                        logger = globalVars.logger
                                    )
        if findFactor:
            func, factor = compileFactor(individual = pop, materialDataDict = materialDataDict2Y)
            with open(FACTOR_RECORD_PATH, "a") as f:
                f.write("\n"+str(pop))
                toRegFactorDict.update({i:factor})
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
            logger.info("end easimple algorithm {} time".format(i+1))






























