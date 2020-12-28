# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:21:37 2020

@author: Evan Hu (Yi Fan Hu)

"""

import numpy as np
import numpy.ma as ma
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from abc import abstractmethod, ABCMeta, abstractstaticmethod

from Tool import globalVars
from Tool.GeneralData import GeneralData
from Tool.DataPreProcessing import *
from GetData.loadData import load_material_data, simple_load_factor
from BackTesting.Signal.SignalBase import SignalBase
from CrossSectionalModels import CrossSectionalModel
# %%

class SignalFactorFromCSV(SignalBase, metaclass=ABCMeta):

    def __init__(self):
        self.rawSignals = GeneralData('rawSignals')
        self.factorNameList = []
        self.dependents = {}
        self.allTradeDatetime = []
        self.metadata = {}

    def initialize(self):
        globalVars.initialize()
        loadedDataList = load_material_data()

        # TODO:use logger latter
        print('We now have {} in our globalVar now'.format(loadedDataList))

        try:
            shiftedReturn = globalVars.materialData['pctChange'].get_shifted(-1)
        except AttributeError as ae:
            print(ae)
            raise AttributeError('There\'s no pctChange in globalVars')
        except Exception as e:
            print(e)
            raise

        shiftedReturn.metadata.update({'shiftN': -1})
        shiftedReturn.name = 'shiftedReturn'
        self.allTradeDatetime = shiftedReturn.timestamp
        self.dependents.update({'shiftedReturn': shiftedReturn})

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
    def generate_signals(self,*,model=None,panelsize=None,trainTestGap=1,
                         prestart=None, preend=None,coverlist=['ST','suspended'],
                         data_dict, control_dict,
                         imputeMethod=ImputeMethod.JustMask(),
                         standardizeMethod=StandardizeMethod.MinMaxScaler(feature_range=(0, 1)),
                         deExtremeMethod=DeExtremeMethod.Quantile(method='clip')):
        # the main main func of this class
        # iter through all time periods and get the signals
        # for each iteration: call train_test_slice, preprocessing, get_signal

        if prestart is None and preend is None:
            alldate = self.allTradeDatetime
        else:
            alldate = self.allTradeDatetime[prestart,preend+1]
        if panelsize is not None and trainTestGap is not None:
            for date in alldate:
                testStart = getLastTradeDate








    @abstractmethod
    def train_test_slice(factors, dependents,
                         trainStart, trainEnd,
                         testStart, testEnd):

        factorTrainDict, factorTestDict = {}, {}
        dependentTrainDict, dependentTestDict = {}, {}

        if trainStart is None:
            for factor in factors:
                factorTrainDict[factor.name] = factor.get_data(at=trainEnd)
                factorTestDict[factor.name] = factor.get_data(at=testEnd)
        else:
            for factor in factors:
                factorTrainDict[factor.name] = np.vstack(factor.get_data(trainStart, trainEnd),
                                                         factor.get_data(at=trainEnd))
                factorTestDict[factor.name] = np.vstack(factor.get_data(testStart, testEnd),
                                                        factor.get_data(at=testEnd))

        for dependent in dependents:
            dependentTrainDict[dependent.name] = dependent.get_data(at=trainEnd)
            dependentTestDict[dependent.name] = dependent.get_data(at=testEnd)
        # split all the factors and toPredicts to train part and test part according to input,
        # if end part isn't passed in, slice one period as default,
        # if the test start isn't passed in,
        # take the very next time period of trainEnd,
        # the input of factors could be a list of factors or just one Factor
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
            maskedData = pipeline.fit_transform(maskedData.T, None).T  # transforming horizontally(stocks-level)

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
        weights = np.ones(periods) / periods
        for factor in factors:
            if (self.columnNames.count(factor) == 0):
                print('non-exist factor ' + factor)
                continue
            index = self.generalData.index(factor)
            self.generalData[:, index] = np.convolve(self.generalData[:, index], weights)
