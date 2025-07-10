# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2021/1/08
# License: MIT License
# update log:2023-12-10 by sunxiwang 18822197631@163.com


import random
import warnings
from typing import Optional, Union, Dict
from collections import defaultdict

import numpy as np
from numpy.random import RandomState
from pandas import DataFrame
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    LeaveOneGroupOut,
)
import torch

### ==============================添加内容=============================== ###
import os
import copy
import pickle
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import softmax
from metabci.brainda.utils.performance import Performance
# from visdom import Visdom
### ==============================添加内容=============================== ###


def set_random_seeds(seed: int):
    """Set seeds for python random module numpy.random and torch.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    seed: int
        Random seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        # Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = False
        # Certain operations in Cudnn are not deterministic, and this line will force them to behave!
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


class EnhancedStratifiedKFold(StratifiedKFold):
    """Enhanced Stratified KFold cross-validator.

    if return_validate is True, split return (train, validate, test) indexs,
    else (train, test) as the sklearn StratifiedKFold.fit the validate size should be the same as the test size.

    Hierarchical K-fold cross-validation.
    When the samples are unbalanced,
    the data set is divided according to the proportion of each type of sample to the total sample.

    Performs hierarchical k-fold cross-validation that can contain validation sets.
    The sample size of the validation set will be the same as that of the test set.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    n_splits : int
        Cross validation fold, default is 5.
    shuffle: bool
        Whether to scramble the sample order. The default is False.
    return_validate: bool
        Whether a validation set is required, which defaults to True.
    random_state: int or numpy.random.RandomState()
        Random initial state. When shuffle is True,
        random_state determines the initial ordering of the samples,
        hrough which the randomness of the selection of various data samples in each compromise can be controlled.
        See sklearn. Model_selection. StratifiedKFold () for details. The default is None.

    Attributes
    ----------
    return_validate: bool
        Same as return_validate in Parameters.
    validate_spliter: sklearn.model_selection.StratifiedShuffleSplit()
        Validate set divider, valid only if return_validate is True.
        See sklearn.model_selection.StratifiedShuffleSplit() for details.


    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        return_validate: bool = True,
        random_state: Optional[Union[int, RandomState]] = None,
    ):

        self.return_validate = return_validate
        if self.return_validate:
            # test_size = 1/(n_splits - 1) if n_splits > 2 else 0.5
            test_size = 1 / n_splits
            self.validate_spliter = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def split(self, X, y, groups=None):
        """Returns the training, validation,
        and test set index subscript (return_validate is True) or the training,
        test set data (return_validate is False).

        author:Swolf <swolfforever@gmail.com>

        Created on:2021-11-29

        update log:
           2023-12-26 by sunchang<18822197631@163.com>

            Parameters
            ----------
            X: array-like, shape(n_samples, n_features)
                Training data. n_samples indicates the number of samples, and n_features indicates the number of features.
            y: array-like, shape(n_samples,)
                Category label.
            groups: None
                Ignorable parameter, used only for version matching.


            Yields
            -------
            train: ndarray
                Training set sample index subscript or training set data.
            validate: ndarray
                Validate set sample index index subscript (return_validate is True).
            test: ndarray
                Test set sample index subscript or test set data.
            """
        for train, test in super().split(X, y, groups=groups):
            if self.return_validate:
                train_ind, validate_ind = next(
                    self.validate_spliter.split(X[train], y[train], groups=groups)
                )
                yield train[train_ind], train[validate_ind], test
            else:
                yield train, test


class EnhancedStratifiedShuffleSplit(StratifiedShuffleSplit):
    """Hierarchical random cross validation.
    When the samples are unbalanced,
    the data set is divided according to the proportion of each type of sample to the total sample.
    Perform hierarchical random cross validation that can contain validation sets.
    The sample size of the validation set will be the same as that of the test set.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    test_size: float
        Test set ratio (0-1).
    train_size: float
        Train set ratio (0-1).
    n_splits: int
        Cross validation fold, default is 5.
    validate_size: float or None
        The proportion of the validation set (when return_validate is True) (0-1), defaults to None.
    return_validate: bool
        Whether a validation set is required, which defaults to True.
    random_state: int or numpy.random.RandomState()
        Random initial state. See sklearn. Model_selection. StratifiedShuffleSplit () for details,
        the default value is None.


    Attributes
    ----------
    return_validate: bool
        Same as return_validate in Parameters.
    validate_spliter: sklearn.model_selection.StratifiedShuffleSplit()
        Validate set divider, valid only if return_validate is True.
        See sklearn.model_selection.StratifiedShuffleSplit() for details.



    """
    def __init__(
        self,
        test_size: float,
        train_size: float,
        n_splits: int = 5,
        validate_size: Optional[float] = None,
        return_validate: bool = True,
        random_state: Optional[Union[int, RandomState]] = None,
    ):

        self.return_validate = return_validate
        if self.return_validate:
            if validate_size is None:
                validate_size = 1 - test_size - train_size
        else:
            validate_size = 0

        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size + validate_size,
            random_state=random_state,
        )

        if self.return_validate:
            total_size = validate_size + train_size
            self.validate_spliter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=validate_size / total_size,
                train_size=train_size / total_size,
                random_state=random_state,
            )

    def split(self, X, y, groups=None):
        """Returns the training, validation,
        and test set index subscript (return_validate is True) or the training,
        test set data (return_validate is False).


        author:Swolf <swolfforever@gmail.com>

        Created on:2021-11-29

        update log:
           2023-12-26 by sunchang<18822197631@163.com>

            Parameters
            ----------
            X: array-like, shape(n_samples, n_features)
                Training data. n_samples indicates the number of samples, and n_features indicates the number of features.
            y: array-like, shape(n_samples,)
                Category label.
            groups: None
                Ignorable parameter, used only for version matching.


            Yields
            -------
            train: ndarray
                Training set sample index subscript or training set data.
            validate: ndarray
                Validate set sample index index subscript (return_validate is True).
            test: ndarray
                Test set sample index subscript or test set data.
        """
        for train, test in super().split(X, y, groups=groups):
            if self.return_validate:
                train_ind, validate_ind = next(
                    self.validate_spliter.split(X[train], y[train], groups=groups)
                )
                yield train[train_ind], train[validate_ind], test
            else:
                yield train, test


class EnhancedLeaveOneGroupOut(LeaveOneGroupOut):
    """
    Leave one method for cross-validation.
    Performs leave-one method cross validation that can contain validation sets.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    return_validate: bool
        Whether a validation set is required, which defaults to True.


    Attributes
    ----------
    return_validate: bool
        Same as return_validate in Parameters.
    validate_spliter: sklearn.model_selection.StratifiedShuffleSplit()
        Validate set divider, valid only if return_validate is True.
        See sklearn.model_selection.StratifiedShuffleSplit() for details.
    """
    def __init__(self, return_validate: bool = True):
        super().__init__()
        self.return_validate = return_validate
        if self.return_validate:
            self.validate_spliter = LeaveOneGroupOut()

    def split(self, X, y=None, groups=None):
        """Returns the training, validation,
        and test set index subscript (return_validate is True) or the training,
        test set data (return_validate is False).

        author:Swolf <swolfforever@gmail.com>

        Created on:2021-11-29

        update log:
            2023-12-26 by sunchang<18822197631@163.com>

            Parameters
            ----------
            X: array-like, shape(n_samples, n_features)
                Training data. n_samples indicates the number of samples, and n_features indicates the number of features.
            y: array-like, shape(n_samples,)
                Category label.Further adjustment is required by _generate_sequential_groups(y).
            groups: None
                The grouping label of the sample used when the data set is split into training,
                validation (return_validate is True), and test sets.
                The number of groups (the number of validation breaks) is calculated by this parameter.
                The number of groups here actually determines the sample size of the "one" part of the leave-one method.
                For example, a set composed of 6 samples with the group number
                [1,1,2,3,3] means that the set is divided into three parts,
                with the number of samples being 2, 1 and 3 respectively.
                In the reserve-one method, the set composed of 2 samples,1 samples and 3 samples is regarded as a test set,
                and the remaining part is regarded as a training set.
                groups can be entered externally or computed by an internal function based on the category label.

            Yields
            -------
            train: ndarray
                Training set sample index subscript or training set data.
            validate: ndarray
                Validate set sample index index subscript (return_validate is True).
            test: ndarray
                Test set sample index subscript or test set data.

            See Also:
            -------
            get_n_splits：Returns the number of packet iterators, that is, the number of packets.
            _generate_sequential_groups：The sample group tag “groups” is generated.
        """

        if groups is None and y is not None:
            groups = self._generate_sequential_groups(y)
        n_splits = super().get_n_splits(groups=groups)
        for train, test in super().split(X, y, groups):
            if self.return_validate:
                n_repeat = np.random.randint(1, n_splits)
                validate_iter = self.validate_spliter.split(
                    X[train], y[train], groups[train]
                )
                for i in range(n_repeat):
                    train_ind, validate_ind = next(validate_iter)
                yield train[train_ind], train[validate_ind], test
            else:
                yield train, test

    def _generate_sequential_groups(self, y):
        labels = np.unique(y)
        groups = np.zeros((len(y)))
        inds = [y == label for label in labels]
        n_labels = [np.sum(ind) for ind in inds]
        if len(np.unique(n_labels)) > 1:
            warnings.warn(
                "y is not balanced, the generated groups is not balanced as well.",
                RuntimeWarning,
            )
        for ind, n_label in zip(inds, n_labels):
            groups[ind] = np.arange(n_label)
        return groups


def generate_kfold_indices(
    meta: DataFrame,
    kfold: int = 5,
    random_state: Optional[Union[int, RandomState]] = None,
):
    """The EnhancedStratifiedKFold class is invoked at the meta data structure level
    to generate cross-validation grouping subscripts.
    The subscript of K-fold cross-validation is generated based on meta class data structure.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    meta: pandas.DataFrame
        metaBCI's custom data class.
    kfold: int
        Cross validation fold, default is 5.
    random_state: int 或 numpy.random.RandomState
        Random initial state, defaults to None.

    Returns
    -------
    indices: dict, {‘subject id’: classes_indices}
        The index subscript of the double-nested dictionary structure,
        the key of the outer dictionary is "subject name",
        the corresponding value classes_indices is dict format,
        and the content is {' e_name ': k_indices}.
        The key of the inner dictionary is the event class name
        and the value is the attempt index subscript k_indices for K-fold cross-validation.
        The variable is a list,
        and the internal elements are tuples (ix_train, ix_val, ix_test)
        composed of the indexes of the corresponding data sets.


    """
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    indices = {}

    for sub_id in subjects:
        sub_ix = meta["subject"] == sub_id
        classes_indices = {}
        for e_name in event_names:
            k_indices = []
            ix = sub_ix & (meta["event"] == e_name)
            spliter = EnhancedStratifiedKFold(
                n_splits=kfold, shuffle=True, random_state=random_state
            )
            for ix_train, ix_val, ix_test in spliter.split(
                np.ones((np.sum(ix))), np.ones((np.sum(ix)))
            ):
                k_indices.append((ix_train, ix_val, ix_test))
            classes_indices[e_name] = k_indices
        indices[sub_id] = classes_indices
    return indices


def match_kfold_indices(k: int, meta: DataFrame, indices):
    """At the level of meta data structure,
    hierarchical K-fold cross-validation packet subscripts are matched to generate specific indexes.
    Based on meta class data structure and combined with the output results of generate_kfold_indices(),
    the specific index is generated.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    k: int
        Cross-verify the index of folds.
    meta: pandas.DataFrame
        metaBCI's custom data class.
    indices: dict, {‘subject id’: classes_indices}
        Subscript dictionary generated by generate_kfold_indices().

    Returns
    -------
    train_ix: ndarray, ‘subject id’: classes_indices
        The index of the training set trials required for k-fold verification
        of the full class data of all subjects (i.e., meta-class data).
    val_ix: ndarray, ‘subject id’: classes_indices
        The validation set trial index required for validation of the meta-class data at k-fold validation.
    test_ix: ndarray, ‘subject id’: classes_indices
        The test set trial index required for validation of the meta-class data at the k-fold.
    """
    train_ix, val_ix, test_ix = [], [], []
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    for sub_id in subjects:
        for e_name in event_names:
            sub_meta = meta[(meta["subject"] == sub_id) & (meta["event"] == e_name)]
            train_ix.append(
                sub_meta.iloc[indices[sub_id][e_name][k][0]].index.to_numpy()
            )
            val_ix.append(sub_meta.iloc[indices[sub_id][e_name][k][1]].index.to_numpy())
            test_ix.append(
                sub_meta.iloc[indices[sub_id][e_name][k][2]].index.to_numpy()
            )
    train_ix = np.concatenate(train_ix)
    val_ix = np.concatenate(val_ix)
    test_ix = np.concatenate(test_ix)
    return train_ix, val_ix, test_ix


def generate_loo_indices(meta: DataFrame):
    """
    The EnhancedLeaveOneGroupOut class is invoked at the meta data structure level
    to generate cross-validation grouping subscripts.
    The subscript of leave-one method cross-validation is generated based on meta class data structure.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    meta: pandas.DataFrame
        metaBCI's custom data class.

    Returns
    -------
    indices: dict, {‘subject id’: classes_indices}
        The index subscript of the double-nested dictionary structure,
        the key of the outer dictionary is "subject name",
        the corresponding value classes_indices is dict format,
        and the content is {' e_name ': k_indices}.
        The key of the inner dictionary is the event class name
        and the value is the attempt index subscript k_indices for K-fold cross-validation.
        The variable is a list,
        and the internal elements are tuples (ix_train, ix_val, ix_test)
        composed of the indexes of the corresponding data sets.
    """
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    indices = {}

    for sub_id in subjects:
        sub_ix = meta["subject"] == sub_id
        classes_indices = {}
        for e_name in event_names:
            k_indices = []
            ix = sub_ix & (meta["event"] == e_name)
            spliter = EnhancedLeaveOneGroupOut()
            groups = np.arange(np.sum(ix))
            for ix_train, ix_val, ix_test in spliter.split(
                np.ones((np.sum(ix))), np.ones((np.sum(ix))), groups=groups
            ):
                k_indices.append((ix_train, ix_val, ix_test))
            classes_indices[e_name] = k_indices
        indices[sub_id] = classes_indices
    return indices


def match_loo_indices(k: int, meta: DataFrame, indices):
    """
    At the meta data structure level, a method is matched
    to cross-validate the grouping subscript and generate the specific index.
    Based on the meta class data structure and combined with the output of generate_loo_indices(),
    the specific index is generated.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    k: int
        Cross-verify the index of folds.
    meta: pandas.DataFrame
        metaBCI's custom data class.
    indices: dict, {‘subject id’: classes_indices}
        Subscript dictionary generated by generate_loo_indices().

    Returns
    -------
    train_ix: ndarray, ‘subject id’: classes_indices
        The index of the training set trial required by the k-fold verification of meta class data.
    val_ix: ndarray, ‘subject id’: classes_indices
        The validation set trial index required for validation of the meta-class data at k-fold validation.
    test_ix: ndarray, ‘subject id’: classes_indices
        The test set trial index required for validation of the meta-class data at the k-fold.

    """
    train_ix, val_ix, test_ix = [], [], []
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    for sub_id in subjects:
        for e_name in event_names:
            sub_meta = meta[(meta["subject"] == sub_id) & (meta["event"] == e_name)]
            train_ix.append(
                sub_meta.iloc[indices[sub_id][e_name][k][0]].index.to_numpy()
            )
            val_ix.append(sub_meta.iloc[indices[sub_id][e_name][k][1]].index.to_numpy())
            test_ix.append(
                sub_meta.iloc[indices[sub_id][e_name][k][2]].index.to_numpy()
            )
    train_ix = np.concatenate(train_ix)
    val_ix = np.concatenate(val_ix)
    test_ix = np.concatenate(test_ix)
    return train_ix, val_ix, test_ix


def match_loo_indices_dict(
        X: Dict,
        y: Dict,
        meta: DataFrame,
        indices,
        k: int
):
    train_X, dev_X, test_X = defaultdict(list), defaultdict(list), defaultdict(list)
    train_y, dev_y, test_y = defaultdict(list), defaultdict(list), defaultdict(list)
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    for sub_index, sub_id in enumerate(subjects):
        for e_name in event_names:
            train_idx = list(indices[sub_id][e_name][k][0])
            dev_idx = list(indices[sub_id][e_name][k][1])
            test_idx = list(indices[sub_id][e_name][k][2])
            train_X[e_name].extend([X[e_name][sub_index][i] for i in train_idx])
            dev_X[e_name].extend([X[e_name][sub_index][i] for i in dev_idx])
            test_X[e_name].extend([X[e_name][sub_index][i] for i in test_idx])
            train_y[e_name].extend([y[e_name][sub_index][i] for i in train_idx])
            dev_y[e_name].extend([y[e_name][sub_index][i] for i in dev_idx])
            test_y[e_name].extend([y[e_name][sub_index][i] for i in test_idx])

    return dict(train_X), dict(train_y), dict(dev_X), \
        dict(dev_y), dict(test_X), dict(test_y)


def generate_shuffle_indices(
    meta: DataFrame,
    n_splits: int = 5,
    test_size: float = 0.1,
    validate_size: float = 0.1,
    train_size: float = 0.8,
    random_state: Optional[Union[int, RandomState]] = None,
):
    """
    Level in the meta data structure called EnhancedStratifiedShuffleSplit class,
    generating cross validation grouping subscript.
    Generate hierarchical random cross-validation subscripts based on meta-class data structures.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    meta: pandas.DataFrame
        metaBCI's custom data class.
    n_splits: int
        Random verification fold, default is 5.
    test_size: float
        The default value is 0.1.
    validate_size: int
        The default value is 0.1, which is the same as that of the test set.
    train_size: int
        The proportion of the number of training sets is 0.8 by default
        (the sum of the proportion of test sets and verification sets is 1).
    random_state: int 或 numpy.random.RandomState
        Random initial state, defaults to None.

    Returns
    -------
    indices: dict, {‘subject id’: classes_indices}
        The index subscript of the double-nested dictionary structure,
        the key of the outer dictionary is "subject name",
        the corresponding value classes_indices is dict format, and the content is {' e_name ': k_indices}.
        The key of the inner dictionary is the event class name and the value is the attempt index subscript k_indices
        for K-fold cross-validation.
        The variable is a list,
        and the internal elements are tuples (ix_train, ix_val, ix_test) composed of the indexes of the corresponding
        data sets.

    """
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    indices = {}

    for sub_id in subjects:
        sub_ix = meta["subject"] == sub_id
        classes_indices = {}
        for e_name in event_names:
            k_indices = []
            ix = sub_ix & (meta["event"] == e_name)
            spliter = EnhancedStratifiedShuffleSplit(
                n_splits=n_splits,
                train_size=train_size,
                test_size=test_size,
                validate_size=validate_size,
                return_validate=True,
                random_state=random_state,
            )
            for ix_train, ix_val, ix_test in spliter.split(
                np.ones((np.sum(ix))), np.ones((np.sum(ix)))
            ):
                k_indices.append((ix_train, ix_val, ix_test))
            classes_indices[e_name] = k_indices
        indices[sub_id] = classes_indices
    return indices


def match_shuffle_indices(k: int, meta: DataFrame, indices):
    """
    Random cross-validation grouping subscripts are matched at the meta data structure level
    to generate specific indexes.
    Based on the meta class data structure and combined with the output of generate_shuffle_indices(),
    a specific index is generated.

    author:Swolf <swolfforever@gmail.com>

    Created on:2021-11-29

    update log:
       2023-12-26 by sunchang<18822197631@163.com>

    Parameters
    ----------
    k: int
        Cross-verify the index of folds.
    meta: pandas.DataFrame
        metaBCI's custom data class.
    indices: dict, {‘subject id’: classes_indices}
        A subscript dictionary generated by generate_shuffle_indices().

    Returns
    -------
    train_ix: ndarray, ‘subject id’: classes_indices
        The index of the training set trial required by the k-fold verification of meta class data.
    val_ix: ndarray, ‘subject id’: classes_indices
        The validation set trial index required for validation of the meta-class data at k-fold validation.
    test_ix: ndarray, ‘subject id’: classes_indices
        The test set trial index required for validation of the meta-class data at the k-fold.

    """
    train_ix, val_ix, test_ix = [], [], []
    subjects = meta["subject"].unique()
    event_names = meta["event"].unique()
    for sub_id in subjects:
        for e_name in event_names:
            sub_meta = meta[(meta["subject"] == sub_id) & (meta["event"] == e_name)]
            train_ix.append(
                sub_meta.iloc[indices[sub_id][e_name][k][0]].index.to_numpy()
            )
            val_ix.append(sub_meta.iloc[indices[sub_id][e_name][k][1]].index.to_numpy())
            test_ix.append(
                sub_meta.iloc[indices[sub_id][e_name][k][2]].index.to_numpy()
            )
    train_ix = np.concatenate(train_ix)
    val_ix = np.concatenate(val_ix)
    test_ix = np.concatenate(test_ix)
    return train_ix, val_ix, test_ix


def generate_char_indices(
    meta: DataFrame,
    kfold: int = 6,
    random_state: Optional[Union[int, RandomState]] = None,
):
    """ Generate the trail index of train set, validation set and test set.
        This method directly manipulate characters

        author: WuJieYu

        Created on: 2023-03-17

        update log:2023-12-26 by sunchang<18822197631@163.com>

        Parameters
        ----------
            meta: DataFrame
                meta of all trials.
            kfold: int
                Number of folds for cross validation.
            random_state: Optional[Union[int, RandomState]]
                State of random, default: None.
        Returns
        ----------
            indices: list
                Trial index for train set, validation set and test set.
                Ensemble in a tuple.
        """
    subjects = meta["subject"].unique()
    indices = {}

    for sub_id in subjects:
        sub_ix = meta["subject"] == sub_id
        # classes_indices = {}
        # char_total = meta.event.__len__()
        k_indices = []
        ix = sub_ix
        spliter = EnhancedStratifiedKFold(
            n_splits=kfold, shuffle=True, random_state=random_state
        )
        for ix_train, ix_val, ix_test in spliter.split(
                np.ones((np.sum(ix))), np.ones((np.sum(ix)))
        ):
            k_indices.append((ix_train, ix_val, ix_test))
        classes_indices = k_indices

        indices[sub_id] = classes_indices
    return indices


def match_char_kfold_indices(k: int, meta: DataFrame, indices):
    """ Divide train set, validation set and test set.
        This method directly manipulate characters

        author: WuJieYu

        Created on: 2023-03-17

        update log:2023-12-26 by sunchang<18822197631@163.com>

        Parameters
        ----------
            k: int
                Number of folds for cross validation.
            meta: DataFrame
                meta of all trials.
            indices: list
                indices of trial index.
        Returns
        ----------
            train_ix, val_ix, test_ix: list
                trial index for train set, validation set and test set.
        """
    train_ix, val_ix, test_ix = [], [], []
    subjects = meta["subject"].unique()
    for sub_id in subjects:
        sub_meta = meta[(meta["subject"] == sub_id)]
        train_ix.append(
            sub_meta.iloc[indices[sub_id][k][0]].index.to_numpy()
        )
        val_ix.append(sub_meta.iloc[indices[sub_id][k][1]].index.to_numpy())
        test_ix.append(
            sub_meta.iloc[indices[sub_id][k][2]].index.to_numpy()
        )
    train_ix = np.concatenate(train_ix)
    val_ix = np.concatenate(val_ix)
    test_ix = np.concatenate(test_ix)
    return train_ix, val_ix, test_ix



### ==============================添加内容=============================== ###
# %% Model parameter reset
def reset_parameters(model):
    """Reset the parameters of a PyTorch model if it has a reset_parameters method.

    This function checks if the provided model has a reset_parameters method and calls
    it to reinitialize the model's parameters. It is useful for resetting neural network
    weights to their initial state before training or inference.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-05-21

    update log:
        2025-05-21 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model whose parameters need to be reset.

    Returns
    -------
    None
        Resets the model parameters in-place if the method exists.
    """
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


# TODO: 这里应该使用官方提供的方案
def cross_validate(x_data, y_label, kfold, data_seed=20230520):
    '''
    This version dosen't use early stoping.
    Arg:
        sub:Subject number.
        data_path:The data path of all subjects.
        augment(bool):Data augment.Take care that this operation will change the size in the temporal dimension.
        validation_rate:The percentage of validation data in the data to be divided.
        data_seed:The random seed for shuffle the data.
    @author:Guangjin Liang
    '''

    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=data_seed)
    for split_train_index, split_validation_index in skf.split(x_data, y_label):
        split_train_x = x_data[split_train_index]
        split_train_y = y_label[split_train_index]
        split_validation_x = x_data[split_validation_index]
        split_validation_y = y_label[split_validation_index]

        split_train_x, split_train_y = torch.FloatTensor(split_train_x), torch.LongTensor(split_train_y).reshape(-1)
        split_validation_x, split_validation_y = torch.FloatTensor(split_validation_x), torch.LongTensor(split_validation_y).reshape(-1)

        split_train_dataset = TensorDataset(split_train_x, split_train_y)
        split_validation_dataset = TensorDataset(split_validation_x, split_validation_y)

        yield split_train_dataset, split_validation_dataset


def validate_model(model, dataset, device, losser, batch_size=128):
    """Evaluate a PyTorch model on a validation dataset.

    This function computes the average loss and accuracy of the model on the provided
    dataset using the specified loss function. The model is set to evaluation mode, and
    gradients are disabled during inference to save memory and improve performance.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-05-21

    update log:
        2025-05-21 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to evaluate.
    dataset : torch.utils.data.Dataset
        The validation dataset to evaluate the model on.
    device : torch.device
        The device (CPU or GPU) to perform computations on.
    losser : callable
        The loss function to compute the model's loss (e.g., torch.nn.CrossEntropyLoss).
    batch_size : int, optional
        Number of samples per batch for the DataLoader (default: 128).

    Returns
    -------
    loss_val : float
        Average loss over the validation dataset.
    accuracy_val : float
        Average accuracy over the validation dataset, computed as the fraction of
        correctly classified samples.
    """
    # Create DataLoader for batch processing
    loader = DataLoader(dataset, batch_size=batch_size)
    # Initialize accumulators for loss and accuracy
    loss_val = 0.0
    accuracy_val = 0.0
    model.eval() # Set model to evaluation mode
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for inputs, target in loader:
            # Move inputs and targets to the specified device
            inputs = inputs.to(device)
            target = target.to(device)

            probs = model(inputs) # Forward pass: compute model predictions
            loss = losser(probs, target) # Compute loss

            loss_val += loss.detach().item() # Accumulate loss (detached to avoid gradient tracking)
            accuracy_val += torch.sum(torch.argmax(probs, dim=1) == target, dtype=torch.float32) # Accumulate accuracy (count correct predictions)

        loss_val = loss_val / len(loader) # Compute average loss
        accuracy_val = accuracy_val / len(dataset) # Compute average accuracy

    return loss_val, accuracy_val


# TODO: 输入的数据应该是meta的类型
def model_training_two_stage(model, criterion, optimizer, lr_scheduler, frist_epochs, eary_stop_epoch, second_epochs, batch_size,
                             X_train, Y_train, kfolds, device,
                             model_name, subject, model_savePath):
    """Train a PyTorch model using a two-stage training strategy with K-fold cross-validation.

    This function implements a two-stage training process for a neural network model, typically used in
    brain-computer interface (BCI) tasks. In the first stage, the model is trained on K-fold cross-validation
    splits with early stopping based on validation loss or accuracy. In the second stage, the best model from
    the first stage is fine-tuned using both training and validation data. The trained model is saved for each fold.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-05-21

    update log:
        2025-05-21 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to train (e.g., TCNet_Fusion or EEGNet).
    criterion : callable
        The loss function for training (e.g., torch.nn.CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        The optimizer for updating model parameters (e.g., torch.optim.Adam).
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler or None
        Learning rate scheduler for adjusting learning rate during training (e.g., CosineAnnealingLR).
        If None, no scheduler is applied.
    frist_epochs : int
        Maximum number of epochs for the first training stage.
    eary_stop_epoch : int
        Number of epochs to wait for early stopping if no improvement in validation accuracy.
    second_epochs : int
        Maximum number of epochs for the second training stage.
    batch_size : int
        Number of samples per batch for DataLoader.
    X_train : torch.Tensor or np.ndarray
        Training data with shape [n_samples, n_channels, n_samples].
    Y_train : torch.Tensor or np.ndarray
        Training labels with shape [n_samples].
    kfolds : int
        Number of folds for K-fold cross-validation.
    device : torch.device
        The device (CPU or GPU) to perform computations on.
    model_name : str
        Name of the model for saving purposes (e.g., 'TCNet_Fusion').
    subject : str or int
        Subject identifier for saving model files.
    model_savePath : str
        Directory path to save trained model files.

    Returns
    -------
    None
        Trains the model in-place and saves the model state for each fold to model_savePath.
        Prints training progress and average evaluation accuracy across folds.

    References
    ----------
    .. [1] Schirrmeister RT, Springenberg JT, Fiederer LDJ, Glasstetter M, Eggensperger K, Tangermann M, Hutter F, Burgard W, Ball T.
           Deep learning with convolutional neural networks for EEG decoding and visualization. Hum Brain Mapp. 2017 Nov;38(11):5391-5420.
    """

    # vis = Visdom(env='main')  # 设置环境窗口的名称,如果不设置名称就默认为main
    # opt_train_acc = {'xlabel':'epochs', 'ylabel':'acc_value', 'title':model_name+'_train_acc'}
    # opt_train_loss = {'xlabel':'epochs', 'ylabel':'loss_value', 'title':model_name+'_train_loss'}
    # opt_eval_acc = {'xlabel':'epochs', 'ylabel':'acc_value', 'title':model_name+'_eval_acc'}
    # opt_eval_loss = {'xlabel':'epochs', 'ylabel':'loss_value', 'title':model_name+'_eval_loss'}

    avg_eval_acc = 0
    for kfold, (train_dataset, valid_dataset) in enumerate(cross_validate(X_train, Y_train, kfolds)):

        # train_acc_window = vis.line(X=[0], Y=[0], opts=opt_train_acc)
        # train_loss_window = vis.line(X=[0], Y=[0], opts=opt_train_loss)
        # eval_acc_window = vis.line(X=[0], Y=[0], opts=opt_eval_acc)
        # eval_loss_window = vis.line(X=[0], Y=[0], opts=opt_eval_loss)

        model.apply(reset_parameters)
        print(len(train_dataset), len(valid_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        ### First step
        best_acc_kfold = 0
        best_acc_kfold_loss = np.inf
        best_loss_kfold = np.inf
        best_loss_kfold_acc = 0
        mini_loss = None
        remaining_epoch = eary_stop_epoch
        for iter in range(frist_epochs):
            loss_train = 0
            accuracy_train = 0

            model.train()
            for inputs, target in train_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad()  # 清空梯度
                output = model(inputs)  # 前向传播和计算损失
                loss = criterion(output, target)
                loss.backward()  # 反向传播和计算梯度
                optimizer.step()  # 更新参数
                accuracy_train += torch.sum(torch.argmax(output, dim=1) == target, dtype=torch.float32) / len(train_dataset)
                loss_train += loss.detach().item() / len(train_dataloader)

            loss_val, accuracy_val = validate_model(model, valid_dataset, device, criterion)

            # vis.line(X=[iter], Y=[accuracy_train.cpu()], win=train_acc_window,  opts=opt_train_acc,  update='append')
            # vis.line(X=[iter], Y=[loss_train],           win=train_loss_window, opts=opt_train_loss, update='append')
            # vis.line(X=[iter], Y=[accuracy_val.cpu()],   win=eval_acc_window,  opts=opt_eval_acc,  update='append')
            # vis.line(X=[iter], Y=[loss_val],             win=eval_loss_window,   opts=opt_eval_loss,   update='append')

            remaining_epoch = remaining_epoch - 1

            if lr_scheduler:
                lr_scheduler.step()  # 调整学习率

            if remaining_epoch <= 0:
                avg_eval_acc += best_acc_kfold
                break
            if mini_loss is None or loss_train < mini_loss:
                mini_loss = loss_train

            if accuracy_val > best_acc_kfold:
                best_model = copy.deepcopy(model.state_dict())
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_acc_kfold = accuracy_val
                best_acc_kfold_loss = loss_val
                remaining_epoch = eary_stop_epoch

            # if loss_val < best_loss_kfold:
            #     best_model = copy.deepcopy(model.state_dict())
            #     optimizer_state = copy.deepcopy(optimizer.state_dict())
            #     best_loss_kfold = loss_val
            #     best_loss_kfold_acc = accuracy_val
            #     remaining_epoch = eary_stop_epoch

            info = '\tKfold:{0:1}\tEpoch:{1:3}\tTra_Loss:{2:.3}\tTr_acc:{3:.3}\tVa_Loss:{4:.3}\tVa_acc:{5:.3}\tMaxVacc:{6:.3}\tToloss:{7:.3}\tramainingEpoch:{8:3}' \
                   .format(kfold + 1, iter, loss_train, accuracy_train, loss_val, accuracy_val, best_acc_kfold, best_acc_kfold_loss, remaining_epoch)
            print(info)

            # info = '\tKfold:{0:1}\tEpoch:{1:3}\tTra_Loss:{2:.3}\tTr_acc:{3:.3}\tVa_Loss:{4:.3}\tVa_acc:{5:.3}\tMinVloss:{6:.3}\tToacc:{7:.3}\tramainingEpoch:{8:3}' \
            #        .format(kfold + 1, iter, loss_train, accuracy_train, loss_val, accuracy_val, best_loss_kfold, best_loss_kfold_acc, remaining_epoch)
            # print(info)

        info = f'Earyly stopping at Epoch {iter},and retrain the Net using both the training data and validation data.'
        print(info)

        ### Second step
        model.load_state_dict(best_model)
        optimizer.load_state_dict(optimizer_state)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        for iter in range(second_epochs):
            model.train()
            for inputs, target in train_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad()  # 清空梯度
                output = model(inputs)  # 前向传播和计算损失
                loss = criterion(output, target)
                loss.backward()  # 反向传播和计算梯度
                optimizer.step()  # 更新参数

            for inputs, target in valid_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad()  # 清空梯度
                output = model(inputs)  # 前向传播和计算损失
                loss = criterion(output, target)
                loss.backward()  # 反向传播和计算梯度
                optimizer.step()  # 更新参数

            loss_val, accuracy_val = validate_model(model, valid_dataset, device, criterion)

            info = '\tKfold:{0:1}\tEpoch:{1:3}\tVa_Loss:{2:.3}\tVa_acc:{3:.3}'.format(kfold + 1, iter, loss_val, accuracy_val)
            print(info)
            if loss_val < mini_loss:
                break

        file_name = '{}_sub{}_fold{}_acc{:.4}.pth'.format(model_name, subject, kfold, best_acc_kfold)
        print(file_name)
        torch.save(model.state_dict(), os.path.join(model_savePath, file_name))

        info = 'The model was saved successfully!'
        print(info)

    info = f"Avg_eval_Acc : {avg_eval_acc * 100 / kfolds:4f}"
    print(info)


# TODO: 输入的数据应该是meta的类型
def model_training_two_stage_up(model, criterion, optimizer, lr_scheduler, frist_epochs, eary_stop_epoch, second_epochs, batch_size,
                                X_train, Y_train, kfolds, device,
                                model_name, subject, model_savePath):
    """Train a PyTorch model using an enhanced two-stage training strategy with K-fold cross-validation.

    This function implements a two-stage training process for a neural network model, optimized for
    brain-computer interface (BCI) tasks. In the first stage, the model is trained on K-fold cross-validation
    splits, with early stopping based on both validation loss and accuracy to select the best model.
    In the second stage, the best model is fine-tuned using both training and validation data. The trained
    model is saved for each fold. Compared to standard two-stage training, this version prioritizes models
    with lower validation loss while ensuring high accuracy.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-05-21

    update log:
        2025-05-21 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to train (e.g., TCNet_Fusion or EEGNet).
    criterion : callable
        The loss function for training (e.g., torch.nn.CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        The optimizer for updating model parameters (e.g., torch.optim.Adam).
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler or None
        Learning rate scheduler for adjusting learning rate during training (e.g., CosineAnnealingLR).
        If None, no scheduler is applied.
    frist_epochs : int
        Maximum number of epochs for the first training stage.
    eary_stop_epoch : int
        Number of epochs to wait for early stopping if no improvement in validation loss or accuracy.
    second_epochs : int
        Maximum number of epochs for the second training stage.
    batch_size : int
        Number of samples per batch for DataLoader.
    X_train : torch.Tensor or np.ndarray
        Training data with shape [n_samples, n_channels, n_samples].
    Y_train : torch.Tensor or np.ndarray
        Training labels with shape [n_samples].
    kfolds : int
        Number of folds for K-fold cross-validation.
    device : torch.device
        The device (CPU or GPU) to perform computations on.
    model_name : str
        Name of the model for saving purposes (e.g., 'TCNet_Fusion').
    subject : str or int
        Subject identifier for saving model files.
    model_savePath : str
        Directory path to save trained model files.

    Returns
    -------
    None
        Trains the model in-place and saves the model state for each fold to model_savePath.
        Prints training progress and average evaluation accuracy across folds.

    References
    ----------
    .. [1] Liang G, Cao D, Wang J, Zhang Z, Wu Y.
           EISATC-Fusion: Inception Self-Attention Temporal Convolutional Network Fusion for Motor Imagery EEG Decoding.
           IEEE Trans Neural Syst Rehabil Eng. 2024;32:1535-1545.
    """
    avg_eval_acc = 0
    for kfold, (train_dataset, valid_dataset) in enumerate(cross_validate(X_train, Y_train, kfolds)):
        model.apply(reset_parameters)
        print(len(train_dataset), len(valid_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        ### First step
        best_loss_kfold = np.inf
        best_loss_kfold_acc = 0
        best_acc_kfold = 0
        best_acc_kfold_loss = np.inf
        mini_loss = None
        remaining_epoch = eary_stop_epoch
        for iter in range(frist_epochs):
            loss_train = 0
            accuracy_train = 0

            model.train()
            for inputs, target in train_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad()  # 清空梯度
                output = model(inputs)  # 前向传播和计算损失
                loss = criterion(output, target)
                loss.backward()  # 反向传播和计算梯度
                optimizer.step()  # 更新参数
                accuracy_train += torch.sum(torch.argmax(output, dim=1) == target, dtype=torch.float32) / len(train_dataset)
                loss_train += loss.detach().item() / len(train_dataloader)

            loss_val, accuracy_val = validate_model(model, valid_dataset, device, criterion)

            remaining_epoch = remaining_epoch - 1

            if lr_scheduler:
                lr_scheduler.step()  # 调整学习率

            if remaining_epoch <= 0:
                avg_eval_acc += best_acc_kfold
                break
            if mini_loss is None or loss_train < mini_loss:
                mini_loss = loss_train

            if loss_val < best_loss_kfold:
                if accuracy_val >= best_acc_kfold:
                    best_model = copy.deepcopy(model.state_dict())
                    optimizer_state = copy.deepcopy(optimizer.state_dict())
                    best_acc_kfold = accuracy_val
                    best_acc_kfold_loss = loss_val
                remaining_epoch = eary_stop_epoch
                best_loss_kfold = loss_val
                best_loss_kfold_acc = accuracy_val

            if accuracy_val > best_acc_kfold:
                best_model = copy.deepcopy(model.state_dict())
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_acc_kfold = accuracy_val
                best_acc_kfold_loss = loss_val
                remaining_epoch = eary_stop_epoch

            info = '\tKfold:{0:1}\tEpoch:{1:3}\tTra_Loss:{2:.3}\tTr_acc:{3:.3}\tVa_Loss:{4:.3}\tVa_acc:{5:.3}\tMinVloss:{6:.3}\tToacc:{7:.3}\tMaxVacc:{8:.3}\tToloss:{9:.3}\tramainingEpoch:{10:3}' \
                   .format(kfold + 1, iter, loss_train, accuracy_train, loss_val, accuracy_val, best_loss_kfold,
                           best_loss_kfold_acc, best_acc_kfold, best_acc_kfold_loss, remaining_epoch)
            print(info)

        info = f'Earyly stopping at Epoch {iter},and retrain the Net using both the training data and validation data.'
        print(info)

        ### Second step
        model.load_state_dict(best_model)
        optimizer.load_state_dict(optimizer_state)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        for iter in range(second_epochs):
            model.train()
            for inputs, target in train_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad()  # 清空梯度
                output = model(inputs)  # 前向传播和计算损失
                loss = criterion(output, target)
                loss.backward()  # 反向传播和计算梯度
                optimizer.step()  # 更新参数

            for inputs, target in valid_dataloader:
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad()  # 清空梯度
                output = model(inputs)  # 前向传播和计算损失
                loss = criterion(output, target)
                loss.backward()  # 反向传播和计算梯度
                optimizer.step()  # 更新参数

            loss_val, accuracy_val = validate_model(model, valid_dataset, device, criterion)

            info = '\tKfold:{0:1}\tEpoch:{1:3}\tVa_Loss:{2:.3}\tVa_acc:{3:.3}'.format(kfold + 1, iter, loss_val, accuracy_val)
            print(info)
            if loss_val < mini_loss:
                break

        file_name = '{}_sub{}_fold{}_acc{:.4}.pth'.format(model_name, subject, kfold, best_acc_kfold)
        print(file_name)
        torch.save(model.state_dict(), os.path.join(model_savePath, file_name))

        info = 'The model was saved successfully!'
        print(info)

    info = f"Avg_eval_Acc : {avg_eval_acc * 100 / kfolds:4f}"
    print(info)


# TODO: 输入的数据应该是meta的类型
def test_with_cross_validate(model, device, X_test, Y_test, model_path, kfolds, subject, visual=False, features_path=None):
    """Evaluate a PyTorch model on a test dataset using K-fold cross-validated models.

    This function loads pre-trained model weights from K-fold cross-validation, evaluates
    the model on the test dataset, and computes performance metrics including accuracy,
    precision, recall, F1 score, and Cohen's Kappa score. It aggregates results across folds
    to compute average accuracy and Kappa score, suitable for brain-computer interface (BCI)
    tasks such as EEG signal classification.

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-05-21

    update log:
        2025-05-21 by Guangjin Liang <3330635482@qq.com>: Initial implementation.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to evaluate (e.g., TCNet_Fusion or EEGNet).
    device : torch.device
        The device (CPU or GPU) to perform computations on.
    X_test : torch.Tensor or np.ndarray
        Test data with shape [n_samples, n_channels, n_samples].
    Y_test : array-like
        Test labels with shape [n_samples].
    model_path : str
        Directory path containing saved model files for each fold.
    kfolds : int
        Number of folds used in cross-validation.
    subject : str or int
        Subject identifier for logging purposes.
    visual : bool, optional
        Whether to save the features of the test data. The default is False.
    features_path : str, optional
        Directory path to save the features of the test data. The default is None.

    Returns
    -------
    avg_acc : float
        Average classification accuracy across all folds (in percentage).
    avg_Kscore : float
        Average Cohen's Kappa score across all folds.
    """
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.to(device).to(torch.float32).requires_grad_()
    else:
        X_test = torch.from_numpy(X_test).to(device).to(torch.float32).requires_grad_()
    files = os.listdir(model_path)

    avg_acc = 0
    avg_Kscore = 0
    performance = Performance(
        estimators_list=["Acc", "Precision", "Recall", "F1", "Kappa"],
    )
    for kfold in range(0, kfolds):
        for filename in files:
            if 'fold{}_'.format(kfold) in filename:
                file_name = filename
                break
        file_path = os.path.join(model_path, file_name)
        state_dict = torch.load(file_path)

        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            if visual:
                probs, embed_feature, transformer_feature = model(X_test, visual=True)
                features_path_1 = features_path + 'embed_feature.pkl'
                with open(features_path_1, 'wb') as f:
                    pickle.dump(embed_feature.to('cpu').numpy(), f)
                features_path_2 = features_path + 'transformer_feature.pkl'
                with open(features_path_2, 'wb') as f:
                    pickle.dump(transformer_feature.to('cpu').numpy(), f)
            else:
                probs = model(X_test)
            probs = softmax(probs, dim=-1).argmax(dim=-1).to('cpu').numpy()
            results = performance.evaluate(y_true=Y_test, y_pred=probs)
            confusion = confusion_matrix(Y_test, probs)

            avg_acc += results["Acc"]
            avg_Kscore += results["Kappa"]

            print(f"subject: {subject}, kfold: {kfold}, Classification accuracy: {results['Acc'] * 100:.4f}")
            print(f'precision:\t{results["Precision"]}\nrecall:\t\t{results["Recall"]}\nf1:\t\t{results["F1"]}\nk_score:\t{results["Kappa"]}\nconfusion:\n{confusion}\n')

    print(f"Average accuracy:\t{avg_acc / kfolds * 100:.4f}\nAverage kappa score:\t{avg_Kscore / kfolds:.4f}")

    return avg_acc / kfolds * 100, avg_Kscore / kfolds
### ==============================添加内容=============================== ###