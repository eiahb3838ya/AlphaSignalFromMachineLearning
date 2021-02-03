import numpy as np
import numpy.ma as ma
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

from Tool import globalVars
from Tool.DataPreProcessing import *


class Signal:
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


if __name__ == '__main__':
    from Tool.DataPreProcessing import ImputeMethod
    sf_csv = Signal()

    input_ = np.random.rand(20, 3)
    input_[0, :] = np.nan
    data_dict = {'field_a': input_}
    control_dict = {'mask_a': np.around(np.random.rand(20, 3))}

    print(data_dict['field_a'][:5, :])
    print(control_dict['mask_a'][:5, :])
    result = sf_csv.preprocessing(data_dict, control_dict,
                                  imputeMethod=ImputeMethod.JustMask(),
                                  standardizeMethod=StandardizeMethod.MinMaxScaler(feature_range=(0, 1)),
                                  deExtremeMethod=DeExtremeMethod.Quantile(method='clip'))

    print(result['field_a'][:5, :])
