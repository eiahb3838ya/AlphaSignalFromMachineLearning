# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:21:37 2020

@author: Evan Hu (Yi Fan Hu)

"""

import numpy as np
import numpy.ma as ma
from collections import OrderedDict
from abc import abstractmethod, ABCMeta, abstractstaticmethod
from copy import copy
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

from Tool import globalVars
from Tool.GeneralData import GeneralData
from Tool.DataPreProcessing import *
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
    def generate_signals(self, startDate = None, endDate = None, panelSize = 1, trainTestGap = 1, 
                         maskList=[],
                         deExtremeMethod=None, imputeMethod=None,
                         standardizeMethod=None, pipeline=None):
        # set startDate & endDate is input is None
        # [startDate,endDate] is the dates interval for backTesting, closed interval
        prereturn = []
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
            testStart = self.get_last_trade_date(testEnd, panelSize - 1)
            trainEnd = self.get_last_trade_date(testEnd, trainTestGap)
            trainStart = self.get_last_trade_date(trainEnd, panelSize - 1)
            # get the mask of train and test sets
            maskTrainDict, maskTestDict, maskdependentTrainDict, maskdepentTestDict = train_test_slice(
                factors=coversign, dependents=self.dependents,
                trainStart=trainStart, trainEnd=trainEnd, testStart=testStart, testEnd=testEnd
            )
            # get factors and dependents for each backTestingDate
            factorTrainDict, factorTestDict, dependentTrainDict, dependentTestDict = train_test_slice(
                factors = globalVars.factors.values(), dependents = self.dependents,
                trainStart = trainStart, trainEnd = trainEnd, testStart = testStart, testEnd = testEnd
            )
            # get the transform data of factorTrain data
            factorTrainresult = sf_csv.preprocessing(factorTrainDict, maskTrainDict,
                                          imputeMethod=ImputeMethod.JustMask(),
                                          standardizeMethod=StandardizeMethod.MinMaxScaler(feature_range=(0, 1)),
                                          deExtremeMethod=DeExtremeMethod.Quantile(method='clip'))

            # get the transform data of factortest data
            factorTestresult = sf_csv.preprocessing(factorTestDict, maskdepentTestDict,
                                         imputeMethod=ImputeMethod.JustMask(),
                                         standardizeMethod=StandardizeMethod.MinMaxScaler(feature_range=(0, 1)),
                                         deExtremeMethod=DeExtremeMethod.Quantile(method='clip'))

            # get the transform data of dependent data
            dependentPreresult = sf_csv.preprocessing(dependentTrainDict, maskdependentTrainDict,
                                         imputeMethod=ImputeMethod.JustMask(),
                                         standardizeMethod=StandardizeMethod.MinMaxScaler(feature_range=(0, 1)),
                                         deExtremeMethod=DeExtremeMethod.Quantile(method='clip'))

            






            mL = []
            for mask in maskList:
                mL.append(globalVars.factors[mask])

            maskTrainDict, maskTestDict, _, _ = train_test_slice(
                factors=mL, dependents=None,
                trainStart=trainStart, trainEnd=trainEnd, testStart=testStart, testEnd=testEnd
            )
            processedTrainDict = self.preprocessing(factorTrainDict, maskTrainDict, deExtremeMethod=deExtremeMethod,
                                                     imputeMethod=imputeMethod, standardizeMethod=standardizeMethod,
                                                     pipeline=pipeline)
            processedTestDict = self.preprocessing(factorTestDict, maskTestDict, deExtremeMethod=deExtremeMethod,
                                                    imputeMethod=imputeMethod, standardizeMethod=standardizeMethod,
                                                    pipeline=pipeline)
        # the main main func of this class
        # iter through all time periods and get the signals
        # for each iteration: call train_test_slice, preprocessing, get_signal
    


    def train_test_slice(factors, dependents=None, trainStart=None, trainEnd=None, testStart=None, testEnd=None):
        # split all the factors and toPredicts to train part and test part according to input,
        # if trainStart = trainEnd: the user doesn't use panel data
        # slice factors at that date
        # else we slice factors from trainStart to trainEnd (closed date interval)
        # dependents always sliced by trainEnd
        # if dependents is None, return {} (can be used when we slice maskDict)
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
        
        return factorTrainDict, factorTestDict, dependentTrainDict, dependentTestDict

    @staticmethod
    def preprocessing(dataDict, maskDict, *, deExtremeMethod=None, imputeMethod=None,
                      standardizeMethod=None, pipeline=None):
        # generating the mask
        mask = None
        for _, maskData in maskDict.items():
            if mask is None:
                mask = np.zeros(maskData.shape)
            mask = np.logical_or(mask, maskData)

        # generating the pipeline
        if pipeline is not None:
            assert (isinstance(pipeline, Pipeline))
        else:
            l = []
            if deExtremeMethod is not None:
                assert (isinstance(deExtremeMethod, TransformerMixin))
                l.append(("de extreme", deExtremeMethod))
            if imputeMethod is not None:
                assert (isinstance(imputeMethod, TransformerMixin))
                l.append(("impute", imputeMethod))
            if standardizeMethod is not None:
                assert (isinstance(standardizeMethod, TransformerMixin))
                l.append(("standardize", standardizeMethod))
            l.append(('passthrough', 'passthrough'))
            pipeline = Pipeline(l)

        # processing the data
        processedDataDict = dict()
        for dataField, data in dataDict.items():
            for _, maskData in maskDict.items():
                assert (data.shape == maskData.shape)
            maskedData = ma.masked_array(data, mask=mask)
            # transforming horizontally(stocks-level)
            maskedData = pipeline.fit_transform(maskedData.T, None).T

            # check the masked proportion
            # minNoMaskProportion = min(1 - np.mean(maskedData.mask, axis=0))
            # if minNoMaskProportion < maskThreshold:
            #     raise ValueError("The remained proportion of data {} is {:.2%} ，"
            #                      "lower than the setting threshold {:.2%}"
            #                      .format(dataField, minNoMaskProportion, maskThreshold))
            processedDataDict[dataField] = maskedData

        return processedDataDict

    @abstractmethod
    # define how we get signal for one interation
    # the obviuos version will be use feature selection and models
    # to predict crossSectional expected returns of next perio
    def get_signal(self,X_train, y_train, X_test, y_test,model=None):
        if model is None:
            premodel = CrossSectionalModel.CrossSectionalModelDecisionTree(jsonPath=None, paraDict=paraDicts)
        else:
            premodel = CrossSectionalModel.model(jsonPath=None, paraDict=paraDicts)

        premodel.fit(X_train, y_train)
        pred_y = model.predict(X_test)

        return pred_y
    
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

