import numpy as np
def log_loss(y_true, y_proba):
    """
    Function to calculate log loss
    :param y_true: list of true values
    :param y_proba: list of probabilites for 1
    :return: overall log loss
    """
    #define an epsilon value
    #this can also be an input
    #this  value is used to clip probabilities
    epsilon = 1e-15
    #initialize empty list to store
    #individual losses
    loss = []
    #loop over all true and predicted probability values

    for yt, yp in zip(y_true, y_proba):
        #adjust probability
        #0 gets converted to 1e-15
        # 1 gets converted to 1-1e-15
        yp = np.clip(yp, epsilon, 1-epsilon)
        #calculate loss for one sample
        temp_loss = -1.0 * (
            yt * np.log(yp) + (1-yt) * np.log(1-yp)
        )
        #add to loss list
        loss.append(temp_loss)
    return np.mean(loss)
