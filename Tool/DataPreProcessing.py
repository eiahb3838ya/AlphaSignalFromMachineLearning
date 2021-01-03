import copy as cp
import numpy as np
import numpy.ma as ma
import sklearn
import sklearn.impute
import sklearn.preprocessing
from sklearn.base import TransformerMixin, BaseEstimator

from Tool import globalVars


__all__ = ['DeExtremeMethod', 'StandardizeMethod', 'ImputeMethod', 'TransformerBase']


class TransformerBase(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.transformer = None

    def _mask_x(self, X):
        if isinstance(X, np.ma.masked_array):
            X_ = X.compressed()
        elif isinstance(X, np.ndarray):
            X_ = X
        else:
            raise ValueError("X should be np.array or np.ma.masked_array")
        return X_.reshape(-1, 1)

    def fit_transform(self, X, y=None):
        X_ = cp.deepcopy(X)
        for i in range(X_.shape[1]):
            X_t = X_[:, i]
            if len(X_t[~X_t.mask & np.isnan(X_t)]) <= 1:
                # globalVars.logger.logger.warning("the number of remaining data after masking"
                #                                  " is lower than 2")
                continue
            X_t[~X_t.mask] = self.transformer.fit_transform(self._mask_x(X_t)).reshape(-1)
        return X_

    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X):
        return self.fit_transform(X)


# %% impute method
class SimpleImputer(TransformerBase):
    def __init__(self, *, missing_values=np.nan, strategy="mean", fill_value=None, verbose=0, copy=True,
                 add_indicator=False):
        super(SimpleImputer, self).__init__()
        self.transformer = sklearn.impute.SimpleImputer(missing_values=missing_values,
                                                        strategy=strategy,
                                                        fill_value=fill_value,
                                                        verbose=verbose, copy=copy,
                                                        add_indicator=add_indicator)


class JustMask(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        X_ = cp.deepcopy(X)
        X_[np.isnan(X_)] = ma.masked
        return X_

    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X):
        return self.fit_transform(X)


class ImputeMethod:
    SimpleImputer = SimpleImputer
    JustMask = JustMask


# %% de extreme method
class MedianStd(TransformerMixin, BaseEstimator):
    def __init__(self, *, multiple=5.2, method='clip'):
        assert method in ['clip', 'mask']

        self.multiple = multiple
        self.method = method

    def fit_transform(self, X, y=None, **fit_params):
        X_ = cp.deepcopy(X)

        median = np.median(X_, axis=0)
        distance_to_median = np.abs(X_ - median)
        median_of_distance = np.median(distance_to_median)

        upper_limit = median + self.multiple * median_of_distance  # upper bound
        lower_limit = median - self.multiple * median_of_distance  # lower bound

        u_outlier = X_[X_ > upper_limit]
        l_outliner = X_[X_ < lower_limit]
        if self.method == 'clip':
            if len(u_outlier) > 0:
                u_outlier = upper_limit
            if len(l_outliner) > 0:
                l_outliner = lower_limit
        elif self.method == 'mask':
            if len(u_outlier) > 0:
                u_outlier = ma.masked
            if len(l_outliner) > 0:
                l_outliner = ma.masked

        return X_

    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X):
        return self.fit_transform(X)


class MeanStd(TransformerMixin, BaseEstimator):
    def __init__(self, *, multiple=5.2, method='clip'):
        assert method in ['clip', 'mask']

        self.multiple = multiple
        self.method = method

    def fit_transform(self, X, y=None, **fit_params):
        X_ = cp.deepcopy(X)

        mean = np.mean(X_, axis=0)
        distance_to_mean = np.abs(X_ - mean)
        median_of_distance = np.median(distance_to_mean)

        upper_limit = mean + self.multiple * median_of_distance  # upper bound
        lower_limit = mean - self.multiple * median_of_distance  # lower bound

        u_outlier = X_[X_ > upper_limit]
        l_outliner = X_[X_ < lower_limit]
        if self.method == 'clip':
            if len(u_outlier) > 0:
                u_outlier = upper_limit
            if len(l_outliner) > 0:
                l_outliner = lower_limit
        elif self.method == 'mask':
            if len(u_outlier) > 0:
                u_outlier = ma.masked
            if len(l_outliner) > 0:
                l_outliner = ma.masked

        return X_

    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X):
        return self.fit_transform(X)


class Quantile(TransformerMixin, BaseEstimator):
    def __init__(self, *, multiple=5.2, method='clip'):
        assert method in ['clip', 'mask']

        self.multiple = multiple
        self.method = method

    def fit_transform(self, X, y=None, **fit_params):
        X_ = cp.deepcopy(X)

        quantile = np.quantile(X_, [0.25, 0.5, 0.75], axis=0)
        gap1 = quantile[2] - quantile[1]
        gap2 = quantile[1] - quantile[0]

        upper_limit = quantile[2] + self.multiple * gap1  # upper bound
        lower_limit = quantile[0] - self.multiple * gap2  # lower bound

        u_outlier = X_[X_ > upper_limit]
        l_outliner = X_[X_ < lower_limit]
        if self.method == 'clip':
            if len(u_outlier) > 0:
                u_outlier = upper_limit
            if len(l_outliner) > 0:
                l_outliner = lower_limit
        elif self.method == 'mask':
            if len(u_outlier) > 0:
                u_outlier = ma.masked
            if len(l_outliner) > 0:
                l_outliner = ma.masked

        return X_

    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X):
        return self.fit_transform(X)


class DeExtremeMethod:
    MedianStd = MedianStd
    MeanStd = MeanStd
    Quantile = Quantile


# %% standardize method
class MaxAbsScaler(TransformerBase):
    def __init__(self, *, copy=True):
        super(MaxAbsScaler, self).__init__()
        self.transformer = sklearn.preprocessing.MaxAbsScaler(copy=copy)


class MinMaxScaler(TransformerBase):
    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        super(MinMaxScaler, self).__init__()
        if sklearn.__version__ >= '0.24':
            self.transformer = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range,
                                                                  copy=copy, clip=clip)
        else:
            self.transformer = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range,
                                                                  copy=copy)


class StandardScaler(TransformerBase):
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        super(StandardScaler, self).__init__()
        self.transformer = sklearn.preprocessing.StandardScaler(copy=copy,
                                                                with_mean=with_mean,
                                                                with_std=with_std)


class RobustScalar(TransformerBase):
    def __init__(self, *, with_centering=True, with_scaling=True,
                 quantile_range=(25.0, 75.0), copy=True, unit_variance=False):
        super(RobustScalar, self).__init__()
        self.transformer = sklearn.preprocessing.RobustScalar(with_centering=with_centering,
                                                              with_scaling=with_scaling,
                                                              quantile_range=quantile_range,
                                                              copy=copy,
                                                              unit_variance=unit_variance)


class StandardizeMethod:
    MaxAbsScaler = MaxAbsScaler
    MinMaxScaler = MinMaxScaler
    StandardScaler = StandardScaler
    RobustScalar = RobustScalar

