# -*- coding: utf-8 -*-
"""
Created on Sat Jan 2 13:30:30 2021

@author: Ye Donggua

"""

import pandas as pd
import numpy as np
import numpy.ma as ma
from copy import copy
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from tqdm.notebook import tqdm

from Tool import globalVars
from Tool.GeneralData import GeneralData
from Tool.DataPreProcessing import *
from BackTesting.Signal.SignalBase import SignalBase
from GeneticPogramming.utils import get_strided


# %%

class SignalSynthesis(SignalBase):

    def __init__(self, model=None, logger=None):
        super().__init__()
        # ml model
        self.model = model
        # 同一个logger
        self.logger = logger
        # director传进来的
        self.factorNameList = []
        # smoothing之前的signal
        self.rawSignals = None
        # 因变量的dict
        self.dependents = {}
        self.metadata = {}
        # dateTimeIndex
        self.allTradeDatetime = None

        self.initialize()

    def initialize(self):
        '''
        initialize self.dependents & allTradeDatetime

        Raises
        ------
        AttributeError
            'There\'s no pctChange in globalVars'

        Returns
        -------
        None.

        '''
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

        self.dependents.update({'shiftedReturn': shiftedReturn})
        self.allTradeDatetime = shiftedReturn.timestamp

    def get_last_trade_date(self, date, n=1):
        if n == 0:
            return date
        try:
            return self.allTradeDatetime[self.allTradeDatetime < date][-n]
        except IndexError:
            raise IndexError("The given {0} days before date {1} out of the lower bound {2}"
                             .format(n, date, self.allTradeDatetime[0]))

    def get_next_trade_date(self, date, n=1):
        if n == 0:
            return date
        try:
            return self.allTradeDatetime[self.allTradeDatetime > date][n-1]
        except IndexError:
            raise IndexError("The given {0} days after date {1} out of the upper bound {2}"
                             .format(n, date, self.allTradeDatetime[-1]))

    def generate_signals(self, startDate, endDate, panelSize=1, trainTestGap=1, maskList=None,
                         deExtremeMethod=None, imputeMethod=None,
                         standardizeMethod=None, pipeline=None, factorNameList=None,
                         modelParams=None, metric_func=None,
                         periods=10, method='linear'):
        '''
        

        Parameters
        ----------
        startDate : TYPE
            startDate of the back testing date interval (left-closed, right-closed).
        endDate : TYPE
            endDate of the back testing date interval (left-closed, right-closed).
        panelSize : TYPE, optional
            time length of the panel factor used. The default is 1: use one-period factor.
        trainTestGap : TYPE, optional
            gap between trainData and testData. 
            The default is 1, use day-T to train model and feed day-T+1 testData to predict.
        maskList : TYPE, optional
            maskNameList. The default is None.
        deExtremeMethod : TYPE, optional
            method for deExtreme factors. The default is None.
        imputeMethod : TYPE, optional
            method for impute factors. The default is None.
        standardizeMethod : TYPE, optional
            method for standardize factors. The default is None.
        pipeline : TYPE, optional
            can input preprocessing method like deExtremeMethod...
            can also input a pipeline directly. The default is None.
        factorNameList : TYPE, optional
            factors used to generateSignals. The default is None.
        modelParams : TYPE, optional
            paras of the model that generates signals. The default is None.
        metric_func : TYPE, optional
            DESCRIPTION. The default is None.
        periods : TYPE, optional
            smoothing params. The default is 10.
        method : TYPE, optional
            smoothing method. The default is 'linear'.

        Returns
        -------
        resultDict : TYPE
            DESCRIPTION.

        '''
        # set startDate & endDate is input is None
        # [startDate,endDate] is the dates interval for backTesting, closed interval
        self.factorNameList = factorNameList

        if maskList is None:
            maskList = []

        
        # assert whether panelSize is out of range
        # default panelSize should be 1
        toStart = len(self.allTradeDatetime[self.allTradeDatetime <= startDate]) - panelSize - trainTestGap + 1
        assert toStart >= 0, 'panelSize out of range'
        endDateShiftOneDay = self.get_next_trade_date(endDate, 1)
        backTestDates = self.allTradeDatetime[(self.allTradeDatetime >= startDate) &
                                              (self.allTradeDatetime <= endDateShiftOneDay)]
        self.logger.info('start to generate signals from {} ot {}'.format(startDate, endDateShiftOneDay))
        # mL存mask的generalData
        mL = []
        for mask in maskList:
            mL.append(globalVars.factors[mask])

        factorL = []
        for factor in factorNameList:
            factorL.append(globalVars.factors[factor])

        signalList = []
        for backTestDate in tqdm(backTestDates):
            # if use default panelSize = 1, Start == End
            # set dates for train_test_slice
            testEnd = backTestDate
            testStart = self.get_last_trade_date(testEnd, panelSize - 1)
            trainEnd = self.get_last_trade_date(testEnd, trainTestGap)
            trainStart = self.get_last_trade_date(trainEnd, panelSize - 1)

            # get the mask of train and test sets
            if mL:
                maskTrainDict, maskTestDict, _, _ = self.train_test_slice(
                    factors=mL, dependents=None,
                    trainStart=trainStart, trainEnd=trainEnd, testStart=testStart, testEnd=testEnd
                )
            else:
                maskTrainDict, maskTestDict = {}, {}

            # get factors and dependents for each backTestingDate
            factorTrainDict, factorTestDict, dependentTrainDict, dependentTestDict = self.train_test_slice(
                factors=factorL, dependents=self.dependents,
                trainStart=trainStart, trainEnd=trainEnd, testStart=testStart, testEnd=testEnd
            )
            # preprocess factors
            processedTrainDict = self.preprocessing(factorTrainDict, maskTrainDict, deExtremeMethod=deExtremeMethod,
                                                    imputeMethod=imputeMethod, standardizeMethod=standardizeMethod,
                                                    pipeline=pipeline)

            processedTestDict = self.preprocessing(factorTestDict, maskTestDict, deExtremeMethod=deExtremeMethod,
                                                   imputeMethod=imputeMethod, standardizeMethod=standardizeMethod,
                                                   pipeline=pipeline)
            
            # stack factorDict to 3D
            trainStack = np.ma.stack([processedTrainDict[factor] for factor in factorNameList]).transpose(2, 1, 0)
            testStack = np.ma.stack([processedTestDict[factor] for factor in factorNameList]).transpose(2, 1, 0)
            # reshape to 2D: XTrain.shape = (nStocks, panelSize*nFields)
            XTrain = trainStack.reshape(trainStack.shape[0], -1)
            XTest = testStack.reshape(testStack.shape[0], -1)

            signalDict = {}
            # there may be several dependents, for loop
            for k in dependentTrainDict.keys():
                signal = np.zeros(dependentTestDict[k].shape) * np.nan
                # concantenate X and y
                dataTrain = np.ma.concatenate([dependentTrainDict[k].reshape(-1, 1), XTrain], axis=1)
                dataTest = np.ma.concatenate([dependentTestDict[k].reshape(-1, 1), XTest], axis=1)

                # clean data
                # 只要dataTrain或mask里有nan就直接mask
                naOrMaskTrain = np.sum(np.logical_or(np.isnan(dataTrain).data, dataTrain.mask), axis=1)
                naOrMaskTest = np.sum(np.logical_or(np.isnan(dataTest).data, dataTest.mask), axis=1)

                dataTrainCleaned = dataTrain[naOrMaskTrain == 0, :]
                dataTestCleaned = dataTest[naOrMaskTest == 0, :]

                self.logger.debug('Actual available training data account for {:.2%} ({} / {})'
                                  .format(len(dataTrainCleaned)/len(dataTrain), len(dataTrainCleaned), len(dataTrain)))
                self.logger.debug('Actual available testing data account for {:.2%} ({} / {})'
                                  .format(len(dataTestCleaned) / len(dataTest), len(dataTestCleaned), len(dataTest)))

                predictY = self.get_signal(X_train=dataTrainCleaned[:, 1:], y_train=dataTrainCleaned[:, 0],
                                           X_test=dataTestCleaned[:, 1:], y_test=dataTestCleaned[:, 0],
                                           model=self.model(**modelParams), metric_func=metric_func)
                signal[naOrMaskTest == 0] = predictY
                signalDict[k] = signal
            signalList.append(signalDict)
            self.logger.debug('{} finished'.format(backTestDate))

        resultDict = {}
        rawSignals = {}
        for dependent in self.dependents.keys():
            signal = np.c_[[x[dependent] for x in signalList]]
            signalGeneralData = GeneralData(name=dependent, generalData=signal,
                                            timestamp=pd.DatetimeIndex(backTestDates),
                                            columnNames=factorL[0].columnNames)
            rawSignals[dependent] = signalGeneralData
            smoothedSignalGeneralData = self.smoothing(signalGeneralData)
            resultDict[dependent] = smoothedSignalGeneralData
        self.rawSignals = rawSignals
        return resultDict

    @staticmethod
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
                factorTrainDict[factor.name] = factor.get_data(at=trainEnd).reshape(1, -1)
                factorTestDict[factor.name] = factor.get_data(at=testEnd).reshape(1, -1)
        else:
            for factor in factors:
                factorTrainDict[factor.name] = np.vstack((factor.get_data(trainStart, trainEnd),
                                                          factor.get_data(at=trainEnd)))
                factorTestDict[factor.name] = np.vstack((factor.get_data(testStart, testEnd),
                                                         factor.get_data(at=testEnd)))
        if dependents is not None:
            for name, dependent in dependents.items():
                dependentTrainDict[name] = dependent.get_data(at=trainEnd)
                dependentTestDict[name] = dependent.get_data(at=testEnd)

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
            if mask is None:
                maskedData = ma.masked_array(data, mask=np.zeros(data.shape))
            else:
                maskData = ma.masked_array(data, mask=mask)

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

    # define how we get signal for one interation
    # the obviuos version will be use feature selection and models
    # to predict crossSectional expected returns of next perio
    def get_signal(self, X_train, y_train, X_test, y_test, model=None, metric_func=None):

        model.fit(X_train, y_train)
        pred_y = model.predict(X_test)

        trainLoss = metric_func(model.predict(X_train), y_train)
        testLoss = metric_func(pred_y, y_test)

        self.logger.info("Model {} training loss: {}, testing loss: {}".format(model.model, trainLoss, testLoss))

        return pred_y

    @staticmethod
    def smoothing(data, periods=10, method='linear'):
        # smoothing methods defind at the end
        # typicaly is the moving average of n days
        # use partial function technic here will be suitable
        toOutputGeneral = copy(data)
        if method == 'linear':
            npdata = toOutputGeneral.generalData
            strided = get_strided(npdata, periods)
            toOutput = strided.mean(axis=1)
            toOutputGeneral.generalData = toOutput
        elif method == 'exp':
            pass
        else:
            print('non-existing method when smoothing')
        return (toOutputGeneral)


# %%
if __name__ == '__main__':
    ss = SignalSynthesis()
