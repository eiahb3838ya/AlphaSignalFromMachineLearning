# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:21:37 2020

@author: Evan Hu (Yi Fan Hu)

"""

import numpy as np
from abc import abstractmethod, ABCMeta, abstractstaticmethod
from copy import copy

from Tool import globalVars
from Tool.GeneralData import GeneralData
from GetData.loadData import load_material_data, simple_load_factor
from BackTesting.Signal.SignalBase import SignalBase
from GeneticProgramming import get_strided
#%%

class SignalFactorFromCSV(SignalBase,  metaclass=ABCMeta):

    def __init__(self):
        self.rawSignals = GeneralData('rawSignals')
        self.factorNameList = []
        self.dependents = {}
        self.allTradeDatetime = []
        self.metadata = {}

    def initialize(self):
        globalVars.initialize()
        loadedDataList = load_material_data()
        
        #TODO:use logger latter 
        print('We now have {} in our globalVar now'.format(loadedDataList))
        
        try:
            shiftedReturn = globalVars.materialData['pctChange'].get_shifted(-1)
        except AttributeError as ae:
            print(ae)
            raise AttributeError('There\'s no pctChange in globalVars')
        except Exception as e :
            print(e)
            raise 
        
        shiftedReturn.metadata.update({'shiftN':-1})
        shiftedReturn.name = 'shiftedReturn'
        self.allTradeDatetime = shiftedReturn.timestamp
        self.dependents.update({'shiftedReturn':shiftedReturn})
        
        
        # TODO: should load from some factor.json file latter
        ############ this part realy socks 
        ############ to be modified latter
        ############ with nice designed globalVars
        ############ the factor in globalVars.factors is a dict 
        toLoadFactors = ['close',
                         'high',
                         'low',
                         'open'
                         ] 
        
        for aFactor in toLoadFactors:
            simple_load_factor(aFactor)
            self.factorNameList.append(aFactor)
        
    def get_last_trade_date(self, date, n=1):
        toGet = self.allTradeDatetime.index(date)-n
        assert toGet >= 0, 'index out of range'
        return self.allTradeDatetime[toGet]

    def get_next_trade_date(self, date, n=1):
        toGet = self.allTradeDatetime.index(date)+n
        assert toGet < len(self.allTradeDatetime), 'index out of range' 
        return self.allTradeDatetime[toGet]



    @abstractmethod
    def generate_signals(self, startDate = None, endDate = None, panelSize = 1, trainTestGap = 1):
        # set startDate & endDate is input is None
        # [startDate,endDate] is the dates interval for backTesting, closed interval
        if startDate is None:
            startDate = self.allTradeDatetime[0]
        if endDate is None:
            endDate = self.allTradeDatetime[-1]
        # assert whether panelSize is out of range
        # default panelSize should be 1
        toStart = self.allTradeDatetime.index(startDate) - panelSize - trainTestGap + 1
        assert toStart >= 0, 'panelSize out of range'
        backTestDates = self.allTradeDatetime[self.allTradeDatetime.index(startDate):(self.allTradeDatetime.index(endDate) + 1)]
        for backTestDate in backTestDates:
            # if use default panelSize = 1, Start == End
            # set dates for train_test_slice
            testEnd = backTestDate
            testStart = get_last_trade_date(testEnd, panelSize - 1)
            trainEnd = get_last_trade_date(testEnd, trainTestGap)
            trainStart = get_last_trade_date(trainEnd, panelSize - 1)
            # get factors and dependents for each backTestingDate
            factorTrainDict, factorTestDict, dependentTrainDict, dependentTestDict = train_test_slice(
                factors = globalVars.factors.values(), dependents = self.dependents,
                trainStart = trainStart, trainEnd = trainEnd, testStart = testStart, testEnd = testEnd
            )

        # the main main func of this class
        # iter through all time periods and get the signals
        # for each iteration: call train_test_slice, preprocessing, get_signal
        pass

    @abstractstaticmethod
    def train_test_slice(factors, dependents=None, trainStart=None, trainEnd=None, testStart=None, testEnd=None):
        
        factorTrainDict, factorTestDict = {}, {}
        dependentTrainDict, dependentTestDict = {}, {}
        
        if trainStart == trainEnd:
            for factor in factors:
                factorTrainDict[factor.name] = factor.get_data(at = trainEnd)
                factorTestDict[factor.name] = factor.get_data(at = testEnd)
        else:
            for factor in factors:
                factorTrainDict[factor.name] = np.vstack(factor.get_data(trainStart, trainEnd),
                                                         factor.get_data(at = trainEnd))
                factorTestDict[factor.name] = np.vstack(factor.get_data(testStart, testEnd),
                                                         factor.get_data(at = testEnd))
        if dependents is not None:      
            for dependent in dependents:
                dependentTrainDict[dependent.name] = dependent.get_data(at = trainEnd)
                dependentTestDict[dependent.name] = dependent.get_data(at = testEnd)
        # split all the factors and toPredicts to train part and test part according to input,
        # if end part isn't passed in, slice one period as default, 
        # if the test start isn't passed in,
        # take the very next time period of trainEnd,
        # the input of factors could be a list of factors or just one Factor
        return factorTrainDict, factorTestDict, dependentTrainDict, dependentTestDict

    @abstractmethod
    def preprocessing(self):
        # apply preprocess in here including 
        # clean up nans and 停牌 ST ect,
        # deal with extreme points
        # and other stuff
        # use np.ma module technic here should be suitable 
        # please make it modulized and easy to maintain (take cleanUpRules as inputs ect.)
        pass

    @abstractmethod
    def get_signal(self):
        # define how we get signal for one interation
        # the obviuos version will be use feature selection and models 
        # to predict crossSectional expected returns of next period
        pass
    
    @staticmethod
    def smoothing(data,periods = 10,method = 'linear'):
        # smoothing methods defind at the end
        # typicaly is the moving average of n days
        # use partial function technic here will be suitable 
        toOutputGeneral = copy(data)
        if method=='linear':  
            npdata = toOutputGeneral.generalData
            strided = get_strided(npdata,  periods)
            toOutput = strided.mean(axis = 1)
            toOutputGeneral.generalData = toOutput    
        elif method=='exp':
            pass
        else:
            print('non-existing method when smoothing')
        return(toOutputGeneral)

