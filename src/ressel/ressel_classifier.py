"""
Author: Sjoerd de Vries
This module contains the RESSEL classifier class.

Based loosely on the scikit-learn project-template
https://github.com/scikit-learn-contrib/project-template/
"""

import math

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .semi_supervised import self_learning
from .utils import get_remaining_indices, majority_vote


# Private method to train the RESSEL classifier using the self_learning method
# from semi_supervised.py
def _train_ensemble_parallel(
    ensemble, X_labeled, y_labeled, X_unlabeled, labeled_size, unlabeled_size
):
    """Private method for training a base classifier and enriching it
    via the RESSEL method

    Parameters
    ----------
    ensemble : ResselClassifier object
        Contains all of the relevant settings used in the training procedure
    X_labeled : array-like
        2-dimensional array containing the labeled independent data used for training
    y_labeled : array-like
        1-dimensional array containing the labeled dependent data used for training
    X_unlabeled : array-like
        2-dimensional array containing the unlabeled independent data used for training
    labeled_size : int
        The number of datapoints in the labeled dataset
    unlabeled_size : int
        The number of datapoints in the unlabeled dataset

    Returns
    -------
    ResselClassifier object
        A classifier trained by using self-training
    """

    # Checking if there is more than one class in the labeled data
    if len(np.unique(y_labeled)) == 1:
        raise Exception("There is only one class present in the labeled data")

    estimator = clone(ensemble.base_estimator)

    # Selecting the labeled data used for training
    if ensemble.stratify:
        labeled_indices_used = np.array(
            resample(
                range(labeled_size),
                n_samples=math.floor(ensemble.labeled_fraction_used * labeled_size),
                stratify=y_labeled,
                replace=ensemble.replace,
            )
        )
    else:
        # To prevent there from being samples with only 1 class used to train the
        # ensemble members
        multiple_classes = False
        while not multiple_classes:
            labeled_indices_used = np.array(
                resample(
                    range(labeled_size),
                    n_samples=math.floor(ensemble.labeled_fraction_used * labeled_size),
                    replace=ensemble.replace,
                )
            )

            if len(np.unique(y_labeled[labeled_indices_used])) != 1:
                multiple_classes = True

    # Selecting the unlabeled data used for training
    unlabeled_indices_used = np.array(
        resample(
            range(unlabeled_size),
            n_samples=math.floor(ensemble.unlabeled_fraction_used * unlabeled_size),
            replace=False,
        )
    )

    # Calculating the indices of the out-of-bag data for used for error-checking
    labeled_indices_unused = get_remaining_indices(labeled_size, labeled_indices_used)

    # Obtaining the data points from the selected indices
    X_labeled_used = X_labeled[labeled_indices_used, :]
    X_labeled_unused = X_labeled[labeled_indices_unused, :]
    y_labeled_used = y_labeled[labeled_indices_used]
    y_labeled_unused = y_labeled[labeled_indices_unused]
    X_unlabeled_used = X_unlabeled[unlabeled_indices_used, :]

    # Applying the self-learning procedure to the selected base classifier
    self_learning_result = self_learning(
        ensemble,
        estimator,
        X_labeled_used,
        y_labeled_used,
        X_labeled_unused,
        y_labeled_unused,
        X_unlabeled_used,
    )

    return self_learning_result


class ResselClassifier(ClassifierMixin, BaseEstimator):
    """The base class for the RESSEL classifier.

    Parameters
    ----------
    base_estimator : object, optional
        The base estimator, by default SVC()
    labeled_fraction_used : float, optional
        The fraction of the labeled data used, by default 1.0
    unlabeled_fraction_used : float, optional
        The fraction of the unlabeled data used, by default 0.35
    replace : bool, optional
        Whether or not sample with replacement, by default True
    ensemble_size : int, optional
        The number of estimators in the ensemble, by default 25
    batch_size : int, optional
        The number of instances to add to the enriched training set during
        self-training. Overwritten if auto_self_training_size = True, by default 10
    sl_iterations : int, optional
        The number of self-training iterations. Overwritten if
        auto_self_training_size = True, by default 10
    auto_self_training_size : bool, optional
        Whether or not to automatically determine the batch_size and sl_iterations
        parameters based on the data. If True, they are set equal to
        floor(sqrt(len(unlabeled_size_))), by default True
    verbose : bool, optional
        Whether or not to print detailed information, by default True
    sl_method : str, optional
        The configuration for self-training. Can be "proportional", "one_each",
        "true_proportion", "no_probability" or "random", by default "true_proportion"
    stratify : bool, optional
        Whether to apply stratification when sampling different labeled sets
        for the different ensemble members to be trained on, by default False
    n_jobs : int, optional
        The number of ensemble members to be trained in parallel, by default 10

    Attributes
    ----------
    self.classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`
    self.X_ : ndarray, shape (n_samples, n_features)
        The input data passed at :meth:`fit`
    self.y_ : ndarray, shape (n_samples,)
        The labels passed at :meth:`fit`
    self.X_labeled_ : ndarray, shape (n_labeled_samples, n_features)
        The input data with labels passed at :meth:`fit`
    self.y_labeled_ : ndarray, shape (n_labeled_samples,)
        The known labels passed at :meth:`fit`
    self.X_unlabeled_ : ndarray, shape (n_unlabeled_samples, n_features)
        The input data without labels, that is labeled -1, passed at :meth:`fit`
    self.labeled_size_ : int
        The number of labeled samples passed at :meth:`fit`
    self.unlabeled_size_ : int
        The number of unlabeled samples passed at :meth:`fit`
    self.fit_ensemble_ : list
        The ensemble trained via the RESSEL method
    self.is_fitted_ : bool
        Whether the classifier has been fitted
    """

    def __init__(
        self,
        base_estimator=SVC(),
        labeled_fraction_used=1.0,
        unlabeled_fraction_used=0.35,
        replace=True,
        ensemble_size=25,
        batch_size=10,
        sl_iterations=10,
        auto_self_training_size=True,
        verbose=True,
        sl_method="true_proportion",
        stratify=False,
        n_jobs=10,
    ):
        self.base_estimator = base_estimator
        self.labeled_fraction_used = labeled_fraction_used
        self.unlabeled_fraction_used = unlabeled_fraction_used
        self.replace = replace
        self.ensemble_size = ensemble_size
        self.batch_size = batch_size
        self.sl_iterations = sl_iterations
        self.auto_self_training_size = auto_self_training_size
        self.verbose = verbose
        self.sl_method = sl_method
        self.stratify = stratify
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """The fitting function for the RESSEL classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples
        y : array-like, shape (n_samples,)
            The target values. An array of int.
            unlabeled data should be labeled with -1

        Returns
        -------
        self : object
            A fitted ResselClassifier object
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        labeled_indices = np.where(y != -1)[0]
        unlabeled_indices = np.where(y == -1)[0]

        self.X_labeled_ = X[labeled_indices]
        self.y_labeled_ = y[labeled_indices]
        self.X_unlabeled_ = X[unlabeled_indices]
        self.labeled_size_ = len(labeled_indices)
        self.unlabeled_size_ = len(unlabeled_indices)

        # Fitting the ensemble members (in parallel)
        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(_train_ensemble_parallel)(
                self,
                self.X_labeled_,
                self.y_labeled_,
                self.X_unlabeled_,
                self.labeled_size_,
                self.unlabeled_size_,
            )
            for _ in range(self.ensemble_size)
        )

        self.fit_ensemble_ = results
        self.is_fitted_ = True

        # Return the classifier object
        return self

    def predict(self, X):
        """The prediction method.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted label for each sample
        """
        # Check is fit had been called
        check_is_fitted(self, "is_fitted_")

        # Input validation
        X = check_array(X)

        # Actual prediction
        result = majority_vote(self.fit_ensemble_, X)

        return result
