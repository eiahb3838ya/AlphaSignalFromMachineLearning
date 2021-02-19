# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:06:31 2021

@author: eiahb
"""

#%% import
import sys
import os
import ray
import json
import numpy as np
import numpy.random as random

from datetime import datetime
from deap import base, creator, gp, tools
from functools import partial

PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'
os.chdir(PROJECT_ROOT)
print("change wd to {}".format(PROJECT_ROOT))
from GeneticPogramming.psetCreator import pset_creator
from GeneticPogramming.rayMapper import ray_deap_map

from GeneticPogramming.evalAlgorithm import preprocess_eval_single_period
from GeneticPogramming.evolutionAlgorithm import easimple
from GeneticPogramming.factorEvaluator import *
from GeneticPogramming.utils import compileFactor

from Tool import Logger, GeneralData, Factor
from GetData import load_data, align_all_to

# use up to 16 core as limit
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ["PYTHONPATH"] = PROJECT_ROOT + ";" + os.environ.get("PYTHONPATH", "")
#%% set parameters 挖因子過程參數

# data path to save factors 用來儲存挖到的因子的路徑
FACTOR_PATH = os.path.join(PROJECT_ROOT,"data\\factors")

# parameters for our digging 關於這次挖因子過程的參數，單時段開始與結束
PERIOD_START = "2017-01-01"
PERIOD_END = "2019-01-01"

# the total iteration times in this process 總共要輪迴幾次
ITERTIMES = 30

# core count to use in multiprocessing 多進程使用的邏輯數
POOL_SIZE = 16

# 以及這次使用的適應度，適應度函數在別的地方定義
EVALUATE_FUNC = ic_evaluator
#%% hyperparameters 魔仙超參數

# population count in initialization and each selection period 初始化的 種群 個體數
N_POP = 200

# max generation in one iteration 最多繁衍的代數
N_GEN = 5

# cross probability 交叉的機率
CXPB = 0.45

# mutation prob 變異的機率
MUTPB = 0.1
# the tournsize of tourn selecetion
TOURNSIZE = 5

# The parameter *termpb* sets the probability to choose between 
# a terminal or non-terminal crossover point.
TERMPB = 0.1

# the height min max of a initial generate 
initGenHeightMin, initGenHeightMax = 1, 3

# the height min max of a mutate sub tree
mutGenHeightMin, mutGenHeightMax = 1, 2

# material data names used to consturct factors
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

# barra factors to apply regression
barraNames = [
    'beta',
    'bp',
    'mom',
    'size',
    'stoa',
    'stom',
    'stoq'
]

config = {}
config.update({
    "PERIOD_START":PERIOD_START,
    "PERIOD_END":PERIOD_END,
    "ITERTIMES":ITERTIMES,
    "POOL_SIZE":POOL_SIZE,
    "EVALUATE_FUNC":EVALUATE_FUNC.__name__,
    "N_POP":N_POP,
    "N_GEN":N_GEN,
    "CXPB":CXPB,
    "MUTPB":MUTPB,
    "TOURNSIZE":TOURNSIZE,
    "TERMPB":TERMPB,
    "initGenHeightMin":initGenHeightMin,
    "initGenHeightMax":initGenHeightMax,
    "mutGenHeightMin":mutGenHeightMin,
    "mutGenHeightMax":mutGenHeightMax,
    "materialDataNames":materialDataNames,
    "barraNames":barraNames
})

print("start try out with config {}".format(str(config)))
# TODO
# import configparser
# config = configparser.ConfigParser()
# config['DEFAULT'] = {'ServerAliveInterval': '45',
#                       'Compression': 'yes',                  
#                       'CompressionLevel': '9'}

#%% pset & creator
def creator_setup():
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', gp.PrimitiveTree, fitness = creator.FitnessMax) 
    

pset = pset_creator(materialDataNames)
creator_setup()
#%% toolbox
# toolbox setup
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_= initGenHeightMin, max_ = initGenHeightMax)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# register for tools to evolution 注册进化过程需要的工具：配种选择、交叉、突变
toolbox.register('select', tools.selTournament, tournsize = TOURNSIZE) 
toolbox.register('crossover', gp.cxOnePointLeafBiased, termpb = TERMPB)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_ = mutGenHeightMin, max_ = mutGenHeightMax)

def multi_mutate(individual, expr, pset):
    '''
    apply multiple kinds of mutation in a funciton
    '''
    rand = random.uniform(0)
    if rand <= 0.4:
        return gp.mutUniform(individual, expr, pset)
    elif rand <= 0.75:
        return gp.mutShrink(individual)
    else:
        return gp.mutNodeReplacement(individual, pset)
    
toolbox.register("mutate", multi_mutate, expr=toolbox.expr_mut, pset=pset)

# use ray to implement multiprocess
# 這裡用的是 ray 包的多進程，會自動生成 Actors 然後進行計算，我們向上封裝成 map (好用) 爽
# 定義在GeneticPogramming.rayMapper
# 如果換成一般的 map 或是 multiprocess.map 也可以跑，很慢
# 可見 single process version
toolbox.register("map", ray_deap_map, creator_setup=creator_setup,
                 pset_creator=partial(pset_creator, materialDataNames = materialDataNames))


#%% define stat and logbook
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

#%% define main 
def main():
    random.seed(318)
    ray.init(num_cpus=POOL_SIZE, ignore_reinit_error=True, log_to_driver=False)
    logbook = tools.Logbook()
    
    # make a new folder 替每次實驗都產生一個文件夾，內含合格的因子，以及最高分的因子(如果 iter 結束都沒找到合格因子)
    test_number = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    this_test_path = os.path.join(FACTOR_PATH,test_number)
    os.makedirs(this_test_path, exist_ok=True)
    os.makedirs(os.path.join(this_test_path, "best_factors"), exist_ok=True)
    os.makedirs(os.path.join(this_test_path, "found_factors"), exist_ok=True)
    
    # 將 config 存下來
    with open(os.path.join(this_test_path,'config.json'), 'w') as outfile:
        json.dump(config, outfile)

    # set up logger
    loggerFolder = os.path.join(this_test_path, 'log')
    os.makedirs(loggerFolder, exist_ok=True)
    logger = Logger(loggerFolder=loggerFolder, exeFileName='log')
    globalVars.initialize(logger)
    
    # load data to globalVars
    load_data("barra",
        os.path.join(os.path.join(PROJECT_ROOT,"data"), "h5")
    )
    load_data("materialData",
        os.path.join(os.path.join(PROJECT_ROOT,"data"), "h5")
    )
    globalVars.logger.info('load all......done')
    
    # prepare data
    # 將我們要用的數據取出，只使用前面指定的數據
    materialDataDict = {k:globalVars.materialData[k] for k in materialDataNames} # only take the data specified in materialDataNames
    barraDict = {k:globalVars.barra[k] for k in barraNames} # only take the data specified in barraNames
    toRegFactorDict = {}
    
    # get the return to compare 
    # 定義用來放進 evaluation function 的 收益率
    open_ = globalVars.materialData['close']
    shiftedPctChange_df = open_.to_DataFrame().pct_change().shift(-1) 
    
    # align data within shiftedPctChange_df data
    # 將所有數據與 收益率數據對齊
    periodShiftedPctChange_df = shiftedPctChange_df.loc[PERIOD_START:PERIOD_END]
    periodShiftedPctChange = GeneralData('periodShiftedPctChange_df', periodShiftedPctChange_df)
    periodMaterialDataDict = align_all_to(materialDataDict, periodShiftedPctChange)
    periodBarraDict = align_all_to(barraDict, periodShiftedPctChange)
    del shiftedPctChange_df, periodShiftedPctChange_df
    
    # stack barra data
    # 事先將要用來回歸的數據，合併為三維數據 np array
    barraStack = None
    toRegFactorStack =  None
    if len(barraDict)>0:
        barraStack = np.stack([aB.generalData for aB in periodBarraDict.values()],axis = 2)
    if len(toRegFactorDict)>0:
        toRegFactorStack = np.stack([aB.generalData for aB in toRegFactorDict.values()],axis = 2)
        
    # put data to ray for latter use
    materialDataDictID = ray.put(periodMaterialDataDict)
    barraStackID = ray.put(barraStack)
    toRegFactorStackID = ray.put(toRegFactorStack)
    
    # combine the func with data, so that the new partial function only needs input of factor ind
    evaluate = partial(
        preprocess_eval_single_period,
        materialDataDictID = materialDataDictID,
        barraStackID = barraStackID,
        toRegFactorStackID = toRegFactorStackID,
        factorEvalFunc = partial(EVALUATE_FUNC, shiftedPctChange = periodShiftedPctChange),
        pset = pset
    )
    
    for i in range(ITERTIMES):
        logger.info("start easimple algorithm from iteration {}th time".format(i+1))

        # start easimple 開始用遺傳算法繁衍，使用對應參數與已經準備好的 evaluate 函數做適應度測試，
        # 回傳有兩種可能:
        # 1 找到合格的 因子 findFactor==True
        # 2 沒找到且 到達 N_GEN 次的繁衍 findFactor==False
        findFactor, returnIndividual, logbook = easimple(toolbox = toolbox,
                                                         stats = mstats,
                                                         logbook = logbook,
                                                         evaluate = evaluate,
                                                         logger = globalVars.logger,
                                                         N_POP = N_POP,
                                                         N_GEN = N_GEN,
                                                         CXPB = CXPB,
                                                         MUTPB = MUTPB
                                                        )
        if findFactor:
            # 若找到因子，存入 found_factors
            func, factor_data = compileFactor(individual = returnIndividual, materialDataDict = periodMaterialDataDict, pset = pset)
            factor = Factor(name=str(returnIndividual),
                            generalData=factor_data,
                            functionName=str(returnIndividual),
                            reliedDatasetNames={"materialData":list(materialDataDict.keys())},
                            parameters_dict={},
                            **config
                        )
            factor.save(os.path.join(this_test_path,"found_factors\\{}.pickle".format(factor.name)))

            # 如果找到因子，後續找到的因子都要與前面的回歸，所以要重新定義 evaluate func
            toRegFactorDict.update({str(returnIndividual):factor})
            if len(toRegFactorDict)>0:
                toRegFactorStack = np.stack([aB.generalData for aB in toRegFactorDict.values()],axis = 2)
                toRegFactorStackID = ray.put(toRegFactorStack)
        
            evaluate = partial(
                preprocess_eval_single_period,
                materialDataDictID = materialDataDictID,
                barraStackID = barraStackID,
                toRegFactorStackID = toRegFactorStackID,
                factorEvalFunc = partial(EVALUATE_FUNC, shiftedPctChange = periodShiftedPctChange),
                pset = pset
            )
            continue;
                
        else:
            # 若沒找到因子，存入 best_factors
            func, factor_data = compileFactor(individual = returnIndividual, materialDataDict = periodMaterialDataDict, pset = pset)
            factor = Factor(name=str(returnIndividual),
                generalData=factor_data,
                functionName=str(returnIndividual),
                reliedDatasetNames={"materialData":list(materialDataDict.keys())},
                parameters_dict={},
                **config
            )
            factor.save(os.path.join(this_test_path,"best_factors\\{}.pickle".format(factor.name)))
            logger.info("end easimple algorithm from iteration {}th time".format(i+1))


#%% main
if __name__ == '__main__':
    from Tool import globalVars
    main()
    






# %%
