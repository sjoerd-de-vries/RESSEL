import sys
from statistics import multimode

import numpy as np


def get_remaining_indices(size, indices_used):
    """Returns the indices within range(size) which are not contained in indices_used

    Parameters
    ----------
    size : int
        Number of total indices
    indices_used : array-like
        The indices already used.

    Returns
    -------
    ndarray,
        Array including all indices not contained in indices_used
    """
    remaining_indices = [x for x in range(size) if x not in indices_used]
    remaining_indices = np.array(remaining_indices)

    return remaining_indices


def majority_vote(ensemble, X_test):
    """Performs a majority vote on the predictions of an ensemble

    Parameters
    ----------
    ensemble : list
        a list containing the ensemble classifiers
    X_test : ndarray
        2-dimensional array containing the independent test data

    Returns
    -------
    majority_prediction : ndarray
        1-dimensional array containing the majority predictions of the ensemble
    """

    predictions = np.full((len(ensemble), len(X_test)), sys.maxsize)
    majority_prediction = np.full(len(X_test), sys.maxsize)

    for i in range(len(ensemble)):
        predictions[i, :] = ensemble[i].predict(X_test)

    for i in range(len(X_test)):
        majority_prediction[i] = np.random.choice(multimode(predictions[:, i]))

    return majority_prediction
