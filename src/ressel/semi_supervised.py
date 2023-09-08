"""
This module is used by the RESSEL classifier
to perform self-learning
"""

import math
import random
from copy import deepcopy

import numpy as np
from sklearn.metrics import accuracy_score

from .utils import get_remaining_indices


def class_counts(y, class_order):
    """Calculates the number of examples corresponding to
    the different classes in class_order.

    Parameters
    ----------
    y : array-like
        1-dimensional array of labels
    class_order : array-like
        1-dimensional array with the different classes in order

    Returns
    -------
    count_array : array-like
        number of occurences for each of the labels in class_order
    """
    count_array = []
    y_list = y.tolist()

    for label in class_order:
        value = y_list.count(label)
        count_array.append(value)

    return count_array


def self_learning(
    ressel_clf,
    model,
    X_labeled_used,
    y_labeled_used,
    X_labeled_unused,
    y_labeled_unused,
    X_unlabeled_used,
):
    """Enriches a single base classifier as part of the ressel_clf ensemble,
    using RESSEL's semi-supervised self-learning method.

    Parameters
    ----------
    ressel_clf : ResselClassifier object
        Class in which all parameters needed in the semi-supervised
        learning step are contained
    model : Classifier object
        Needs to implement the sci-kit learn model API.
        e.g. sklearn.neighbors.KNeighborsClassifier
    X_labeled_used : array-like
        2-dimensional array containing the labeled independent data
        used for training
    y_labeled_used : array-like
        1-dimensional array containing labeled dependent data used
        for training
    X_labeled_unused : array-like
        2-dimensional array containing all of the labeled independent
        data not used for training,
        but for error measurement
    y_labeled_unused : array-like
        1-dimensional array containing all of the labeled dependent
        data not used for training,
        but for error measurement
    X_unlabeled_used : array-like
        2-dimensional array containing all of the unlabeled data used

    Returns
    -------
    Classifier object
        A trained classifier object, enriched using the RESSEL method.

    Raises
    ------
    Exception
        No valid SL method selected
    Exception
        Sadly, different class order occur. Debugging, remove.
    """

    # Setting parameters for self-training
    if ressel_clf.auto_self_training_size:
        calculated_size = math.floor(math.sqrt(len(X_unlabeled_used)))
        iterations = batch_size = calculated_size
    else:
        iterations = ressel_clf.sl_iterations
        batch_size = ressel_clf.batch_size

    # All arrays to be modified must be copies,
    X_extended = np.copy(X_labeled_used)
    y_extended = np.copy(y_labeled_used)
    X_unlabeled = np.copy(X_unlabeled_used)

    # Fitting the model on just the labeled data
    model.fit(X_extended, y_extended)

    # Set the initial model as the best encountered model so far
    elite_model = deepcopy(model)
    elite_iteration = 0

    # Calculate the inital out-of-bag accuracy
    current_oob_acc = accuracy_score(model.predict(X_labeled_unused), y_labeled_unused)
    initial_oob_accuracy = elite_oob_acc = current_oob_acc

    # Initializing class counts and frequencies for proportional selection
    class_order = model.classes_
    n_classes = class_order.shape[0]
    class_prior_counts = class_counts(y_labeled_used, class_order)
    class_prior_counts = np.array([float(i) for i in class_prior_counts])
    class_prior_frequencies = class_prior_counts / (len(y_labeled_used) / batch_size)
    remainder = np.zeros(n_classes)

    for i in range(iterations):
        # Preparation for adding the appropriate data later on
        if ressel_clf.sl_method == "proportional":
            class_counts_to_add = np.ceil(class_prior_frequencies).astype(int)
        elif ressel_clf.sl_method == "one_each":
            class_counts_to_add = np.ones(len(class_order)).astype(int)
        elif ressel_clf.sl_method == "true_proportion":
            class_counts_to_add = np.floor(class_prior_frequencies).astype(int)
            remainder = remainder + np.remainder(class_prior_frequencies, 1)
            rest = batch_size - sum(class_counts_to_add)

            for _ in range(rest):
                index_max = np.where(remainder == np.amax(remainder))[0][0]
                class_counts_to_add[index_max] = class_counts_to_add[index_max] + 1
                remainder[index_max] = remainder[index_max] - 1

        elif ressel_clf.sl_method == "no_probability":
            # Purely for the next validation step
            class_counts_to_add = np.ones(batch_size)
        elif ressel_clf.sl_method == "random":
            # Purely for the next validation step
            class_counts_to_add = np.ones(batch_size)
        else:
            raise Exception("No valid SL method selected")

        # Check to see if there is enough data left to perform another
        # iteration of self training
        if sum(class_counts_to_add) > X_unlabeled.shape[0]:
            print("Not enough labeled data left to increment")
            print(f"Step: {i})")
            print(f"class_counts_to_add: {class_counts_to_add}")
            print(f"iterations: {iterations}")
            print(f"batch_size: {batch_size}")
            print()
            break

        y_unlabeled_probabilities = model.predict_proba(X_unlabeled)
        class_order_new = model.classes_

        # Sanity check, should never occur
        if any(class_order != class_order_new):
            raise Exception("Sadly, different a class order occurred, write solution")

        indices_moved = []

        # Actually performing the self-learning step, based on class_counts_to_add
        if ressel_clf.sl_method == "no_probability":
            y_unlabeled_predicted = model.predict(X_unlabeled)
            random_indices = np.array(
                random.sample(range(X_unlabeled.shape[0]), k=batch_size)
            )
            X_new_data = np.copy(X_unlabeled[random_indices, :])
            y_new_data = np.copy(y_unlabeled_predicted[random_indices])
            indices_moved = random_indices
        elif ressel_clf.sl_method == "random":
            random_indices = np.array(
                random.sample(range(X_unlabeled.shape[0]), k=batch_size)
            )
            X_new_data = np.copy(X_unlabeled[random_indices, :])
            y_new_data = np.array(random.choices(class_order, k=batch_size))
            indices_moved = random_indices
        else:
            # when sl_method = proportional, one_each or true_proportion

            # To initialize array with new data on the first addition
            first_addition = True

            for j in range(n_classes):
                to_add = class_counts_to_add[j]

                if to_add != 0:
                    sorted_indices = np.argsort(y_unlabeled_probabilities[:, j])
                    top_indices = sorted_indices[-1 : -(to_add + 1) : -1]
                    X_to_add = np.copy(X_unlabeled[top_indices, :])
                    y_to_add = np.full(to_add, class_order[j])

                    # All probabilities set to zero to prevent an index from
                    # being added twice as two different classes within an iteration
                    for temp_index in top_indices:
                        y_unlabeled_probabilities[temp_index, :] = np.zeros(
                            y_unlabeled_probabilities.shape[1]
                        )

                    # To add data points for each class
                    if first_addition:
                        X_new_data = np.copy(X_to_add)
                        y_new_data = np.copy(y_to_add)
                        first_addition = False
                    else:
                        X_new_data = np.append(X_new_data, X_to_add, axis=0)
                        y_new_data = np.append(y_new_data, y_to_add, axis=0)

                    indices_moved.extend(list(top_indices))

        # Add the selected data points to the train set
        X_extended = np.append(X_extended, X_new_data, axis=0)
        y_extended = np.append(y_extended, y_new_data, axis=0)

        remaining_indices = get_remaining_indices(X_unlabeled.shape[0], indices_moved)

        if len(remaining_indices) != 0:
            X_unlabeled = X_unlabeled[remaining_indices, :]

        model.fit(X_extended, y_extended)
        current_oob_acc = accuracy_score(
            model.predict(X_labeled_unused), y_labeled_unused
        )

        # Elitism
        if current_oob_acc > elite_oob_acc:
            elite_oob_acc = current_oob_acc
            elite_model = deepcopy(model)
            elite_iteration = i + 1

    elite_accuracy_difference = elite_oob_acc - initial_oob_accuracy
    extended_set_accuracy = accuracy_score(model.predict(X_extended), y_extended)

    # Out-of-bag accuracy metrics
    if ressel_clf.verbose:
        print()
        print("-----")
        print()
        print(f"Initial oob accuracy: {round(100*initial_oob_accuracy, ndigits=2)}")
        print(f"Best encountered oob accuracy: {round(100*elite_oob_acc, ndigits=2)}")
        print(f"Final oob accuracy: {round(100*current_oob_acc, ndigits=2)}")
        print(
            f"Self learning difference: \
            {round(100*elite_accuracy_difference, ndigits=2)}"
        )
        print(f"Elite iteration: {elite_iteration}")

    train_used_accuracy = accuracy_score(model.predict(X_labeled_used), y_labeled_used)

    # Training accuracy metrics
    if ressel_clf.verbose:
        print()
        print(f"Training used accuracy: {round(100*train_used_accuracy, ndigits=2)}")
        print(f"Extended data accuracy: {round(100*extended_set_accuracy, ndigits=2)}")

    return elite_model
