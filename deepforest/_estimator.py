"""A wrapper on the base estimator for the naming consistency."""


__all__ = ["Estimator"]

import numpy as np
from .forest import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier as sklearn_RandomForestClassifier,
    ExtraTreesClassifier as sklearn_ExtraTreesClassifier,
    RandomForestRegressor as sklearn_RandomForestRegressor,
    ExtraTreesRegressor as sklearn_ExtraTreesRegressor,
)
from rich.progress import Progress

def make_classifier_estimator(
    name,
    criterion,
    n_trees=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    backend="custom",
    n_jobs=None,
    random_state=None,
):
    # RandomForestClassifier
    if name == "rf":
        # backend是指用的是自己写的还是sklearn的，自己写的调用forest.py里的包
        if backend == "custom":
            # 所以一个estimator就是一个随机森林分类器或回归器
            estimator = RandomForestClassifier(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        elif backend == "sklearn":
            estimator = sklearn_RandomForestClassifier(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                oob_score=True,
                n_jobs=n_jobs,
                random_state=random_state,
            )
    # ExtraTreesClassifier
    elif name == "erf":
        if backend == "custom":
            estimator = ExtraTreesClassifier(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        elif backend == "sklearn":
            estimator = sklearn_ExtraTreesClassifier(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                oob_score=True,
                n_jobs=n_jobs,
                random_state=random_state,
            )
    else:
        msg = "Unknown type of estimator, which should be one of {{rf, erf}}."
        raise NotImplementedError(msg)

    return estimator


def make_regressor_estimator(
    name,
    criterion,
    n_trees=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    backend="custom",
    n_jobs=None,
    random_state=None,
):
    # RandomForestRegressor
    if name == "rf":
        if backend == "custom":
            estimator = RandomForestRegressor(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        elif backend == "sklearn":
            estimator = sklearn_RandomForestRegressor(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                oob_score=True,
                n_jobs=n_jobs,
                random_state=random_state,
            )
    # ExtraTreesRegressor
    elif name == "erf":
        if backend == "custom":
            estimator = ExtraTreesRegressor(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        elif backend == "sklearn":
            estimator = sklearn_ExtraTreesRegressor(
                criterion=criterion,
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                oob_score=True,
                n_jobs=n_jobs,
                random_state=random_state,
            )
    else:
        msg = "Unknown type of estimator, which should be one of {{rf, erf}}."
        raise NotImplementedError(msg)

    return estimator


class Estimator(object):
    def __init__(
        self,
        name,
        criterion,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        backend="custom",
        n_jobs=None,
        random_state=None,
        is_classifier=True,
    ):

        self.backend = backend
        self.is_classifier = is_classifier
        if self.is_classifier:
            self.estimator_ = make_classifier_estimator(
                name,
                criterion,
                n_trees,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                backend,
                n_jobs,
                random_state,
            )
        else:
            self.estimator_ = make_regressor_estimator(
                name,
                criterion,
                n_trees,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                backend,
                n_jobs,
                random_state,
            )

    # 袋外数据估计器
    @property
    def oob_decision_function_(self):
        # Scikit-Learn uses `oob_prediction_` for ForestRegressor
        # 表示使用的是sklearn中的随机森林回归器
        if self.backend == "sklearn" and not self.is_classifier:
            oob_prediction = self.estimator_.oob_prediction_
            if len(oob_prediction.shape) == 1:
                oob_prediction = np.expand_dims(oob_prediction, 1)
            return oob_prediction
        return self.estimator_.oob_decision_function_

    # 计算森林的特征重要性
    @property
    def feature_importances_(self):
        """Return the impurity-based feature importances from the estimator."""

        return self.estimator_.feature_importances_

    def fit_transform(self, X, y, sample_weight=None):
        self.estimator_.fit(X, y, sample_weight)
        return self.oob_decision_function_

    # 输出单个森林对样本集的 二维类概率向量数组 ，而不是概率最高的类
    def transform(self, X):
        """Preserved for the naming consistency."""
        return self.predict(X)


    # 输出单个森林对样本集的 二维类概率向量数组
    def predict(self, X):
        if self.is_classifier:
            # 返回的是一个二维数组，每行是一个样本的类概率向量
            return self.estimator_.predict_proba(X)
        pred = self.estimator_.predict(X)
        if len(pred.shape) == 1:
            pred = np.expand_dims(pred, 1)
        return pred
