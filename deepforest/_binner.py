"""
Implementation of the Binner class in Deep Forest.

This class is modified from:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/_hist_gradient_boosting/binning.py
"""


__all__ = ["Binner"]

# 应该是给特征的取值分箱

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array

from . import _cutils as _LIB


X_DTYPE = np.float64
X_BINNED_DTYPE = np.uint8 # unsigned integer 8 bits 无符号8位整数
ALMOST_INF = 1e300 # 远远超出了宇宙的原子数量1e80


# 对每列数据分箱，返回每个箱子的阈值，有两种方法：百分位分箱法和等长间隔分箱法
def _find_binning_thresholds_per_feature(
    col_data, n_bins, bin_type="percentile"
):
    """
    Private function used to find midpoints for samples along a
    specific feature.
    """
    if len(col_data.shape) != 1:

        msg = (
            "Per-feature data should be of the shape (n_samples,), but"
            " got {}-dims instead."
        )
        raise RuntimeError(msg.format(len(col_data.shape)))

    missing_mask = np.isnan(col_data) # 判断每个值是否是NAN，返回一个布尔数组
    if missing_mask.any():
        col_data = col_data[~missing_mask] # 从原始的 col_data 数组中提取出所有不是缺失值的元素
    col_data = np.ascontiguousarray(col_data, dtype=X_DTYPE) # 将数组转换为连续的内存块
    distinct_values = np.unique(col_data)

    # 这里是样本数小于等于箱子数的情况，直接将每个样本作为一个箱子，且最终数量英爱也没有达到n_bins的要求数
    # 举个例子[1, 2, 3, 4]，如果n_bins=5
    # 按照下面的代码给他分为[1.5, 2.5, 3.5]
    if len(distinct_values) <= n_bins: 
        midpoints = distinct_values[:-1] + distinct_values[1:]
        midpoints *= 0.5
    else:
        # Equal interval in terms of percentile
        if bin_type == "percentile": # 用分位数来分箱
            percentiles = np.linspace(0, 100, num=n_bins + 1) # 是包括100的，共n_bins+1个数
            percentiles = percentiles[1:-1] # 去掉最后一个100
            midpoints = np.percentile(
                col_data, percentiles, interpolation="midpoint"
            ).astype(X_DTYPE)
            assert midpoints.shape[0] == n_bins - 1
            np.clip(midpoints, a_min=None, a_max=ALMOST_INF, out=midpoints)
        # Equal interval in terms of value
        elif bin_type == "interval": # 用等长间隔来分箱
            min_value, max_value = np.min(col_data), np.max(col_data)
            intervals = np.linspace(min_value, max_value, num=n_bins + 1)
            midpoints = intervals[1:-1]
            assert midpoints.shape[0] == n_bins - 1
        else:
            raise ValueError("Unknown binning type: {}.".format(bin_type))

    return midpoints

# 对样本集的特征找出分箱阈值
def _find_binning_thresholds(
    X, n_bins, bin_subsample=2e5, bin_type="percentile", random_state=None
):
    n_samples, n_features = X.shape
    rng = check_random_state(random_state) # 传入整数代表使用这个固定的随机数生成器，传入None代表使用随机的随机数生成器

    if n_samples > bin_subsample:
        subset = rng.choice(np.arange(n_samples), bin_subsample, replace=False) # replace=False代表不放回抽样
        X = X.take(subset, axis=0) # 从原始数据 X 中取出相应的子样本

    binning_thresholds = []
    for f_idx in range(n_features):
        threshold = _find_binning_thresholds_per_feature(
            X[:, f_idx], n_bins, bin_type
        )
        binning_thresholds.append(threshold)

    return binning_thresholds

# 分箱类
class Binner(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_bins=255,
        bin_subsample=2e5,
        bin_type="percentile",
        random_state=None,
    ):
        self.n_bins = n_bins + 1  # + 1 for missing values
        self.bin_subsample = int(bin_subsample)
        self.bin_type = bin_type
        self.random_state = random_state
        self._is_fitted = False

    def _validate_params(self):

        if not 2 <= self.n_bins - 1 <= 255:
            msg = (
                "`n_bins` should be in the range [2, 255], bug got"
                " {} instead."
            )
            raise ValueError(msg.format(self.n_bins - 1))

        if not self.bin_subsample > 0:
            msg = (
                "The number of samples used to construct the Binner"
                " should be strictly positive, but got {} instead."
            )
            raise ValueError(msg.format(self.bin_subsample))

        if self.bin_type not in ("percentile", "interval"):
            msg = (
                "The type of binner should be one of {{percentile, interval"
                "}}, bug got {} instead."
            )
            raise ValueError(msg.format(self.bin_type))

    def fit(self, X):

        self._validate_params()

        self.bin_thresholds_ = _find_binning_thresholds(
            X,
            self.n_bins - 1, # 之前在初始化的时候把n_bins加了1，这里要减回去，因为多的一个箱子是给缺失值的，而这里的分箱是去除了缺失值的
            self.bin_subsample,
            self.bin_type,
            self.random_state,
        )

        self.n_bins_non_missing_ = np.array(
            [thresholds.shape[0] + 1 for thresholds in self.bin_thresholds_],
            dtype=np.uint32,
        )

        self.missing_values_bin_idx_ = self.n_bins - 1
        self._is_fitted = True

        return self

    def transform(self, X):

        if not self._is_fitted:
            msg = (
                "The binner has not been fitted yet when calling `transform`."
            )
            raise RuntimeError(msg)

        if not X.shape[1] == self.n_bins_non_missing_.shape[0]: # 这里判断如果样本的特征数和之前fit的时候的特征数不一致，就报错
            msg = (
                "The binner was fitted with {} features but {} features got"
                " passed to `transform`."
            )
            raise ValueError(
                msg.format(self.n_bins_non_missing_.shape[0], X.shape[1])
            )

<<<<<<< HEAD
        X = check_array(X, dtype=X_DTYPE, force_all_finite=False)

        # X_binned 表示每个特征的每个值所属的箱子的编号
=======
        X = check_array(X, dtype=X_DTYPE, force_all_finite=False) # 经过check_array之后，X为array类型， force_all_finite=False代表允许有缺失值和无穷大值
>>>>>>> bc0534fd7646c705cb903b33cc9a2a578c9917f2
        X_binned = np.zeros_like(X, dtype=X_BINNED_DTYPE, order="F")

        # 进行分箱操作，将每个特征的每个值所属的箱子的编号存储到 X_binned 中
        # _LIB是pyx文件，采用编译好的C代码，能够加速运行
        _LIB._map_to_bins(
            X, self.bin_thresholds_, self.missing_values_bin_idx_, X_binned
        )

        return X_binned
