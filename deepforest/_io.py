"""
Implement methods on dumping and loading large objects using joblib. This
class is designed to support the partial mode in deep forest.
"""


__all__ = ["Buffer"]

import os
import shutil
import warnings
import tempfile
from joblib import load, dump


class Buffer(object):
    """
    The class of dumping and loading large array objects including the data
    and estimators.

    Parameters
    ----------
    partial_mode : bool

        - If ``True``, a temporary buffer on the local disk is created to
          cache objects such as data and estimators.
        - If ``False``, all objects are directly stored in memory without
          extra processing.
    store_est : bool, default=True
        Whether to cache the estimators to the local buffer.
    store_pred : bool, default=True
        Whether to cache the predictor to the local buffer.
    store_data : bool, default=False
        Whether to cache the intermediate data to the local buffer.
    """

    # Buffer类的init函数，创建了所需要的buffer存储文件夹
    def __init__(
        self,
        use_buffer,
        buffer_dir=None,
        store_est=True,
        store_pred=True,
        store_data=False,
    ):

        # 共三种东西可以存到buffer中，分别是estimator，predictor，data
        self.use_buffer = use_buffer
        self.store_est = store_est and use_buffer
        self.store_pred = store_pred and use_buffer
        self.store_data = store_data and use_buffer
        self.buffer_dir = os.getcwd() if buffer_dir is None else buffer_dir

        # tempfile.TemporaryDirectory()创建一个临时目录，当程序结束时，该目录及其内容将被自动删除
        # 这里的参数dir代表将要创建的文件的副目录，存放在工作目录里
        # Create buffer
        if self.use_buffer:
            self.buffer = tempfile.TemporaryDirectory(
                prefix="buffer_", dir=self.buffer_dir
            )

            # 如果需要存储数据，那么就在buffer文件夹中创建一个data_开头的文件夹
            if store_data:
                self.data_dir_ = tempfile.mkdtemp(
                    prefix="data_", dir=self.buffer.name
                )

            # 如果需要存储estimator或predictor那么就在buffer文件夹中创建一个model_开头的文件夹
            if store_est or store_pred:
                self.model_dir_ = tempfile.mkdtemp(
                    prefix="model_", dir=self.buffer.name
                )
                self.pred_dir_ = os.path.join(self.model_dir_, "predictor.est")

    # 用于返回buffer的名字，而这个名字应该在tempfile.TemporaryDirectory()这个函数中会生成吧
    @property
    def name(self):
        """Return the buffer name."""
        if self.use_buffer:
            return self.buffer.name
        else:
            return None


    def cache_data(self, layer_idx, X, is_training_data=True):
        """
        When ``X`` is a large array, it is not recommended to directly pass the
        array to all processors because the array will be copied multiple
        times and cause extra overheads. Instead, dumping the array to the
        local buffer and reading it as the ``numpy.memmap`` mode across
        processors is able to speed up the training and evaluating process.

        就是说并行生成estimator的时候会将样本复制很多份给每个processor，这样会造成很大的开销
        将数据先存入本地缓冲区再进行内存映射就很高效

        Parameters
        ----------
        layer_idx : int
            The index of the cascade layer that utilizes ``X``.
        X : ndarray of shape (n_samples, n_features)
            The training / testing data to be cached.
        is_training_data : bool, default=True
            Whether ``X`` is the training data.

        Returns
        -------
        X: {ndarray, ndarray in numpy.memmap mode}

            - If ``self.store_data`` is ``True``, return the memory-mapped
            object of `X` cached to the local buffer.
            - If ``self.store_data`` is ``False``, return the original ``X``.
        """
        if not self.store_data:
            return X

        # 给该层的数据创建一个缓冲区，区分训练数据和测试数据
        if is_training_data:
            cache_dir = os.path.join(
                self.data_dir_, "joblib_train_{}.mmap".format(layer_idx)
            )
            # 如果当前目录下已经存在该数据文件了，则需要删除它
            # 通过操作系统中的解除链接关系，如果链接数为0，则删除该文件
            # Delete
            if os.path.exists(cache_dir):
                os.unlink(cache_dir)
        else:
            cache_dir = os.path.join(
                self.data_dir_, "joblib_test_{}.mmap".format(layer_idx)
            )
            # Delete
            if os.path.exists(cache_dir):
                os.unlink(cache_dir)

        # 将数据存放在.mmap文件中，这是一个内存映射文件
        # Dump and reload data in the numpy.memmap mode
        dump(X, cache_dir)

        # 对于数据，返回的应该是存在这个文件中的对象
        # 而estimator和predictor返回的是文件的路径
        X_mmap = load(cache_dir, mmap_mode="r+")

        return X_mmap

    def cache_estimator(self, layer_idx, est_idx, est_name, est):
        """
        Dumping the fitted estimator to the buffer is highly recommended,
        especially when the python version is below 3.8. When the size of
        estimator is large, for instance, several gigabytes in the memory,
        sending it back from each processor will cause the struct error.

        Reference:
            https://bugs.python.org/issue17560

        Parameters
        ----------
        layer_idx : int
            The index of the cascade layer that contains the estimator to be
            cached.
        est_idx : int
            The index of the estimator in the cascade layer to be cached.
        est_name : {"rf", "erf", "custom"}
            The name of the estimator to be cached.
        est : object
            The object of base estimator.

        Returns
        -------
        cache_dir : {string, object}

            - If ``self.store_est`` is ``True``, return the absolute path to
              the location of the cached estimator.
            - If ``self.store_est`` is ``False``, return the estimator.
        """
        if not self.store_est:
            return est

        filename = "{}-{}-{}.est".format(layer_idx, est_idx, est_name)
        cache_dir = os.path.join(self.model_dir_, filename)
        dump(est, cache_dir)

        # 对于存储的estimator，返回的应该是存储的路径，所有后面还有load函数
        return cache_dir

    def cache_predictor(self, predictor):
        """
        Please refer to `cache_estimator`.

        Parameters
        ----------
        predictor : object
            The object of the predictor.

        Returns
        -------
        pred_dir : {string, object}

            - If ``self.store_pred`` is ``True``, return the absolute path to
              the location of the cached predictor.
            - If ``self.store_pred`` is ``False``, return the predictor.
        """
        if not self.store_pred:
            return predictor

        dump(predictor, self.pred_dir_)

        # 对于存储的predictor，返回的应该是存储的路径，所有后面还有load函数
        return self.pred_dir_

    def load_estimator(self, estimator_path):
        if not os.path.exists(estimator_path):
            msg = "Missing estimator in the path: {}."
            raise FileNotFoundError(msg.format(estimator_path))

        # load函数返回任意python的对象，这里返回的应该是estimator
        estimator = load(estimator_path)

        return estimator

    def load_predictor(self, predictor):
        # Since this function is always called from `cascade.py`, the input
        # `predictor` could be the actual predictor object. If so, this
        # function will directly return the predictor.
        if not isinstance(predictor, str):
            return predictor

        if not os.path.exists(predictor):
            msg = "Missing predictor in the path: {}."
            raise FileNotFoundError(msg.format(predictor))

        # load函数返回任意python的对象，这里返回的应该是predictor
        predictor = load(predictor)

        return predictor

    def del_estimator(self, layer_idx):
        """Used for the early stopping stage in deep forest."""
        for est_name in os.listdir(self.model_dir_):
            if est_name.startswith(str(layer_idx)):

                # 直接删掉这些estimator的映射文件
                try:
                    os.unlink(os.path.join(self.model_dir_, est_name))
                except OSError:
                    msg = (
                        "Permission denied when deleting the dumped"
                        " estimators during the early stopping stage."
                    )
                    warnings.warn(msg, RuntimeWarning)

    # 删除buffer下的所有临时文件
    def close(self):
        """Clean up the buffer."""
        try:
            self.buffer.cleanup()
        except OSError:
            msg = "Permission denied when cleaning up the local buffer."
            warnings.warn(msg, RuntimeWarning)


# 为模型的存储创建一个文件夹
def model_mkdir(dirname):
    """Make the directory for saving the model."""
    if os.path.isdir(dirname):
        msg = "The directory to be created already exists {}."
        raise RuntimeError(msg.format(dirname))

    os.mkdir(dirname)
    os.mkdir(os.path.join(dirname, "estimator"))


# 这里应该就是保存训练好的模型，所以与训练数据没什么关系
def model_saveobj(dirname, obj_type, obj, partial_mode=False):
    """Save objects of the deep forest according to the specified type."""

    if not os.path.isdir(dirname):
        msg = "Cannot find the target directory: {}. Please create it first."
        raise RuntimeError(msg.format(dirname))

    # 保存参数和binner，不需要查看partial_mode，直接就保存在pkl文件中了
    if obj_type in ("param", "binner"):
        if not isinstance(obj, dict):
            msg = "{} to be saved should be in the form of dict."
            raise RuntimeError(msg.format(obj_type))
        dump(obj, os.path.join(dirname, "{}.pkl".format(obj_type)))

    elif obj_type == "layer":
        if not isinstance(obj, dict):
            msg = "The layer to be saved should be in the form of dict."
            raise RuntimeError(msg)

        est_path = os.path.join(dirname, "estimator")
        if not os.path.isdir(est_path):
            msg = "Cannot find the target directory: {}."
            raise RuntimeError(msg.format(est_path))

        # If `partial_mode` is True, each base estimator in the model is the
        # path to the dumped estimator, and we only need to move it to the
        # target directory.
        # 意思就是：如果partial_mode为True，那么estimator就是一个路径，直接移动到目标文件夹就行了
        if partial_mode:
            for _, layer in obj.items():
                for estimator_key, estimator in layer.estimators_.items():
                    dest = os.path.join(est_path, estimator_key + ".est")
                    shutil.move(estimator, dest)
        # Otherwise, we directly use `joblib.dump` to save the estimator to
        # the target directory.
        else:
            for _, layer in obj.items():
                for estimator_key, estimator in layer.estimators_.items():
                    dest = os.path.join(est_path, estimator_key + ".est")
                    dump(estimator, dest)
    elif obj_type == "predictor":
        pred_path = os.path.join(dirname, "estimator", "predictor.est")

        # Same as `layer`
        if partial_mode:
            shutil.move(obj, pred_path)
        else:
            dump(obj, pred_path)
    else:
        raise ValueError("Unknown object type: {}.".format(obj_type))


def model_loadobj(dirname, obj_type, d=None):
    """Load objects of the deep forest from the given directory."""

    if not os.path.isdir(dirname):
        msg = "Cannot find the target directory: {}."
        raise RuntimeError(msg.format(dirname))

    if obj_type in ("param", "binner"):
        obj = load(os.path.join(dirname, "{}.pkl".format(obj_type)))
        return obj
    elif obj_type == "layer":
        from ._layer import (
            ClassificationCascadeLayer,
            RegressionCascadeLayer,
            CustomCascadeLayer,
        )

        if not isinstance(d, dict):
            msg = "Loading layers requires the dict from `param.pkl`."
            raise RuntimeError(msg)

        n_estimators = d["n_estimators"]
        n_layers = d["n_layers"]
        layers = {}

        for layer_idx in range(n_layers):

            if not d["use_custom_estimator"]:
                if d["is_classifier"]:
                    layer_ = ClassificationCascadeLayer(
                        layer_idx=layer_idx,
                        n_outputs=d["n_outputs"],
                        criterion=d["criterion"],
                        n_estimators=d["n_estimators"],
                        partial_mode=d["partial_mode"],
                        buffer=d["buffer"],
                        verbose=d["verbose"],
                    )
                else:
                    layer_ = RegressionCascadeLayer(
                        layer_idx=layer_idx,
                        n_outputs=d["n_outputs"],
                        criterion=d["criterion"],
                        n_estimators=d["n_estimators"],
                        partial_mode=d["partial_mode"],
                        buffer=d["buffer"],
                        verbose=d["verbose"],
                    )

                for est_type in ("rf", "erf"):
                    for est_idx in range(n_estimators):
                        est_key = "{}-{}-{}".format(
                            layer_idx, est_idx, est_type
                        )
                        dest = os.path.join(
                            dirname, "estimator", est_key + ".est"
                        )

                        if not os.path.isfile(dest):
                            msg = "Missing estimator in the path: {}."
                            raise RuntimeError(msg.format(dest))

                        if d["partial_mode"]:
                            layer_.estimators_.update(
                                {est_key: os.path.abspath(dest)}
                            )
                        else:
                            est = load(dest)
                            layer_.estimators_.update({est_key: est})
            else:

                layer_ = CustomCascadeLayer(
                    layer_idx=layer_idx,
                    n_splits=1,  # will not be used
                    n_outputs=d["n_outputs"],
                    estimators=[None] * n_estimators,  # will not be used
                    partial_mode=d["partial_mode"],
                    buffer=d["buffer"],
                    verbose=d["verbose"],
                )

                for est_idx in range(n_estimators):
                    est_key = "{}-{}-custom".format(layer_idx, est_idx)
                    dest = os.path.join(dirname, "estimator", est_key + ".est")

                    if not os.path.isfile(dest):
                        msg = "Missing estimator in the path: {}."
                        raise RuntimeError(msg.format(dest))

                    if d["partial_mode"]:
                        layer_.estimators_.update({est_key: dest})
                    else:
                        est = load(dest)
                        layer_.estimators_.update({est_key: est})

            layer_key = "layer_{}".format(layer_idx)
            layers.update({layer_key: layer_})
        return layers
    elif obj_type == "predictor":

        if not isinstance(d, dict):
            msg = "Loading the predictor requires the dict from `param.pkl`."
            raise RuntimeError(msg)

        pred_path = os.path.join(dirname, "estimator", "predictor.est")

        if not os.path.isfile(pred_path):
            msg = "Missing predictor in the path: {}."
            raise RuntimeError(msg.format(pred_path))

        if d["partial_mode"]:
            return os.path.abspath(pred_path)
        else:
            predictor = load(pred_path)
            return predictor
    else:
        raise ValueError("Unknown object type: {}.".format(obj_type))
