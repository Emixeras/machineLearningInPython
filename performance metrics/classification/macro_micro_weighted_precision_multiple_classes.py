import numpy as np
from machineLearningInPython import accuracy_v2 as acc
from collections import Counter


def macro_precision(y_true, y_pred):
    """
    Function to calculate macro averaged precisions
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: macro precision score
    """
    # find the number of classes by taking
    # length of unique values in true list
    num_classes = len(np.unique(y_true))

    # initialize precision to 0
    precision = 0

    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        tp = acc.true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        fp = acc.false_positive(temp_true, temp_pred)

        temp_precision = tp / (tp + fp)

        # keep adding precision
        precision += temp_precision
    # calculate and return average precision over all classes
    precision /= num_classes
    return precision


def micro_precision(y_true, y_pred):
    """
    Function to calculate micro averaged precisions
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: micro precision score
    """
    # find the number of classes by taking
    # length of unique values in true list
    num_classes = len(np.unique(y_true))

    # initialize tp and fp to 0
    tp = 0
    fp = 0

    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        # and update overall tp
        tp += acc.true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        # and update overall fp
        fp += acc.false_positive(temp_true, temp_pred)

    # calculate and return overall precision
    precision = tp / (tp + fp)
    return precision


def weighted_precision(y_true, y_pred):
    """
    Function to calculate weighted averaged precision
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: weighted precision score
    """

    # find number of classes by taking
    # length of unique values
    num_classes = len(np.unique(y_true))

    # create class:sample count dictionary
    # it looks something like this
    # {0: 20, 1:15, 2:21}
    class_counts = Counter(y_true)

    #initialize precision to 0
    precision = 0

    #loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate tp and fp for class
        tp = acc.true_positive(temp_true, temp_pred)
        fp = acc.false_positive(temp_true, temp_pred)

        #calculate precision of class

        temp_precision = tp / (tp+fp)
        weighted_precision = class_counts[class_] * temp_precision

        #add to overall precision
        precision += weighted_precision

    # calculate overall precision by dividing by
    # total number of samples
    overall_precision = precision / len(y_true)
    return overall_precision

