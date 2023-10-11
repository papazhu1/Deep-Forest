# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3


cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport isnan
ctypedef np.npy_bool BOOL
ctypedef np.npy_intp SIZE_t # 是整数的指针吧
ctypedef np.npy_int32 INT32_t
ctypedef np.npy_float64 X_DTYPE_C
ctypedef np.npy_uint8 X_BINNED_DTYPE_C

np.import_array()


# 对多个类向量进行取平均值
cpdef void _c_merge_proba(np.ndarray[X_DTYPE_C, ndim=2] probas,
                          SIZE_t n_outputs,
                          np.ndarray[X_DTYPE_C, ndim=2] out):
    cdef:
        SIZE_t n_features = probas.shape[1]
        SIZE_t start = 0
        SIZE_t count = 0

    while start < n_features:

        # 这里probas应该是一个二维数组，行暂时不懂是什么，
        # 每n_outputs列是一个类向量，将类向量相加取平均值
        out += probas[:, start : (start + n_outputs)]
        start += n_outputs
        count += 1

    out /= count


# 根据indices对样本生成掩码,indices存被选中的样本序号
cpdef np.ndarray _c_sample_mask(const INT32_t [:] indices,
                                int n_samples):
    """
    Generate the sample mask given indices without resorting to `np.unique`."""
    cdef:
        SIZE_t i
        SIZE_t n = indices.shape[0] # indices为选中的样本序号，不等于总共的样本数
        SIZE_t sample_id
        np.ndarray[BOOL, ndim=1] sample_mask = np.zeros((n_samples,),
                                                        dtype=np.bool)
    # 这是一个 Cython 的语句块，表示下面的代码将在没有全局解释器锁（GIL）的情况下运行，以提高并行性能。
    with nogil:
        for i in range(n):
            sample_id = indices[i]
            if not sample_mask[sample_id]:
                sample_mask[sample_id] = True

    return sample_mask

# 对所有特征分桶
# Modified from HGBDT in Scikit-Learn
cpdef _map_to_bins(object X,
                   list binning_thresholds,
                   const unsigned char missing_values_bin_idx,
                   X_BINNED_DTYPE_C [::1, :] binned):
    """Bin numerical values to discrete integer-coded levels.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        The numerical data to bin.
    binning_thresholds : list of arrays
        For each feature, stores the increasing numeric values that are
        used to separate the bins.
    binned : ndarray, shape (n_samples, n_features)
        Output array, must be fortran aligned.
    """
    cdef:
        const X_DTYPE_C[:, :] X_ndarray = X
        SIZE_t n_features = X.shape[1]
        SIZE_t feature_idx

    # 对每列调用_map_num_col_to_bins函数进行分桶
    for feature_idx in range(n_features):
        _map_num_col_to_bins(X_ndarray[:, feature_idx],
                             binning_thresholds[feature_idx],
                             missing_values_bin_idx,
                             binned[:, feature_idx])


# 该函数给每个样本的某个特征分桶，解果保存在binned数组中
# binned保存了每个样本的特征值对应的bin序号，如果是缺失值则为missing_values_bin_idx
cdef void _map_num_col_to_bins(const X_DTYPE_C [:] data,
                               const X_DTYPE_C [:] binning_thresholds,
                               const unsigned char missing_values_bin_idx,
                               X_BINNED_DTYPE_C [:] binned):
    """Binary search to find the bin index for each value in the data."""
    cdef:
        SIZE_t i
        SIZE_t left
        SIZE_t right
        SIZE_t middle

    for i in range(data.shape[0]):

        if isnan(data[i]):
            binned[i] = missing_values_bin_idx
        else:
            # for known values, use binary search
            left, right = 0, binning_thresholds.shape[0]
            while left < right:
                middle = (right + left - 1) // 2
                if data[i] <= binning_thresholds[middle]:
                    right = middle
                else:
                    left = middle + 1
            binned[i] = left
