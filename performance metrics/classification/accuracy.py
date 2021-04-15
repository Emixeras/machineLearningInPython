def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score
    """
    correct_counter = 0
    for yt, yp in zip(y_true, y_pred):
        if yt==yp:
            correct_counter+=1
    return correct_counter / len(y_true)

#so gehts mit scikit learn
from sklearn import metrics
l1=[0,1,1,1,0,0,0,1]
l2=[0,1,0,1,0,1,0,0]
metrics.accuracy_score(l1,l2)
print(metrics.accuracy_score(l1,l2))

print(accuracy(l1,l2))
