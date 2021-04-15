from machineLearningInPython import accuracy_v2 as acc


def mcc(y_true, y_pred):
    """
     This function calculates Matthew's Correlation Coefficient
     for binary classification.
     :param y_true: list of true values
     :param y_pred: list of predicted values
     :return: mcc score
     """
    tp = acc.true_positive(y_true, y_pred)
    tn = acc.true_negative(y_true, y_pred)
    fp = acc.false_positive(y_true, y_pred)
    fn = acc.false_negative(y_true, y_pred)

    numerator = (tp*tn) - (fp*fn)

    denominator = (
        (tp+fp)*
        (fn+tn)*
        (fp+tn)*
        (tp+fn)
    )
    denominator = denominator ** 0,5

    return numerator/denominator

