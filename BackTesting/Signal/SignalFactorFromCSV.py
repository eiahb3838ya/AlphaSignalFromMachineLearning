# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:21:37 2020

@author: Evan Hu (Yi Fan Hu)

"""

import numpy as np
from abc import abstractmethod, ABCMeta, abstractstaticmethod

from Tool import globalVars
from Tool.GeneralData import GeneralData
from GetData.loadData import load_material_data, simple_load_factor
from BackTesting.Signal.SignalBase import SignalBase
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
        




    @abstractmethod
    def generate_signals(self):
        # the main main func of this class
        # iter through all time periods and get the signals
        # for each iteration: call train_test_slice, preprocessing, get_signal
        pass


    def train_test_slice(self, 
                         factorNameList, dependentNameList, 
                         trainStart, trainEnd = None, 
                         testStart = None, testEnd = None):
        if trainEnd is None:
            trainEnd = self.allTradeDatetime[1 + self.allTradeDatetime.index(trainStart)]
        if testStart is None:
            testStart = trainEnd
        if testEnd is None:
            testEnd = testStart
        self.factorTrainDict, self.factorTestDict = {}, {}
        self.dependentTrainDict, self.dependentTestDict = {}, {}
        for factorName in factorNameList:
            factor = globalVars.factors[factorName]
            self.factorTrainDict[factor.name] = factor.get_data(trainStart,trainEnd)
            self.factorTestDict[factor.name] = factor.get_data(testStart,testEnd)
        for dependentName in dependentNameList:
            dependent = self.dependents[dependentName]
            self.dependentTrainDict[dependent.name] = dependent.get_data(at = trainEnd)
            self.dependentTestDict[dependent.name] = dependent.get_data(at = testEnd)
        # split all the factors and toPredicts to train part and test part according to input,
        # if end part isn't passed in, slice one period as default, 
        # if the test start isn't passed in,
        # take the very next time period of trainEnd,
        # the input of factors could be a list of factors or just one Factor
        return self.factorTrainDict, self.factorTestDict, self.dependentTrainDict, self.dependentTestDict

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

    def smoothing(self, periods=10, factors=None):
        # smoothing methods defind at the end
        # typicaly is the moving average of n days
        # use partial function technic here will be suitable 
        '''
        now the self here is something like what we see in 
        the generalData.py, in there must be some differences, 
        cause i haven't understand the whole procedure...
        it left to be improved later
        '''
        weights = np.ones(periods)/periods
        for factor in factors:
            if(self.columnNames.count(factor)==0):
                print('non-exist factor '+factor)
                continue
            index = self.generalData.index(factor)
            self.generalData[:,index] = np.convolve(self.generalData[:,index],weights)
