import numpy as np
import accuracy_v2 as acc
from collections import Counter


def macro_recall(y_true, y_pred):
    """
    Function to calculate macro averaged recalls
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: macro recall score
    """
    # find the number of classes by taking
    # length of unique values in true list
    num_classes = len(np.unique(y_true))

    # initialize precision to 0
    recall = 0

    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        tp = acc.true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        fn = acc.false_negative(temp_true, temp_pred)

        temp_recall = tp / (tp + fn)

        # keep adding precision
        recall += temp_recall
    # calculate and return average precision over all classes
    recall /= num_classes
    return recall

def micro_recall(y_true, y_pred):
    """
    Function to calculate micro recall precisions
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: micro precision score
    """
    # find the number of classes by taking
    # length of unique values in true list
    num_classes = len(np.unique(y_true))

    # initialize tp and fn to 0
    tp = 0
    fn = 0

    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        # and update overall tp
        tp += acc.true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        # and update overall fn
        fn += acc.false_negative(temp_true, temp_pred)

    # calculate and return overall precision
    recall = tp / (tp + fn)
    return recall

def weighted_recall(y_true, y_pred):
    """
    Function to calculate weighted averaged recall
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: weighted recall score
    """

    # find number of classes by taking
    # length of unique values
    num_classes = len(np.unique(y_true))

    # create class:sample count dictionary
    # it looks something like this
    # {0: 20, 1:15, 2:21}
    class_counts = Counter(y_true)

    #initialize recall to 0
    recall = 0

    #loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate tp and fn for class
        tp = acc.true_positive(temp_true, temp_pred)
        fn = acc.false_negative(temp_true, temp_pred)

        #calculate recall of class

        temp_recall = tp / (tp+fn)
        weighted_recall = class_counts[class_] * temp_recall

        #add to overall recall
        recall += weighted_recall

    # calculate overall recall by dividing by
    # total number of samples
    overall_recall = recall / len(y_true)
    return overall_recall
