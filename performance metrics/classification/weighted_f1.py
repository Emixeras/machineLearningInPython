from collections import Counter
import numpy as np
from sklearn import metrics


def weighted_f1(y_true, y_pred):
    """
    function to calculate weighted f1 score
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: weighted f1 score
    """

    # find number of classes by taking
    # length of unique values
    num_classes = len(np.unique(y_true))

    # create class:sample count dictionary
    # it looks something like this
    # {0: 20, 1:15, 2:21}
    class_counts = Counter(y_true)

    # initialize f1 to 0
    f1 = 0

    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate precision and recall for class
        p = metrics.precision_score(temp_true, temp_pred)
        r = metrics.recall_score(temp_true, temp_pred)

        # calculate f1 of class
        if p + r != 0:
            temp_f1 = 2 * p * r / (p + r)
        else:
            temp_f1 = 0

        # multiply f1 with count of samples in class
        weighted_f1 = class_counts[class_] * temp_f1
        f1 += weighted_f1

    overall_f1 = f1 / len(y_true)
    return overall_f1


