from machineLearningInPython import accuracy_v2


def tpr(y_true, y_pred):
    """
    true positive rate also called recall
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: tpr/recall
    """
    return accuracy_v2.recall(y_true, y_pred)


def fpr(y_true, y_pred):
    """
    false positive rate
    :param y_true:
    :param y_pred:
    :return:
    """
    fp = accuracy_v2.false_positive(y_true, y_pred)
    tn = accuracy_v2.true_negative(y_true, y_pred)
    return fp / (tn + fp)


# empty lists to store tpr and fpr values
tpr_list = []
fpr_list = []

# actual targets
y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]

# predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

# handmade threshholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]

# loop over all thresholds

for thresh in thresholds:
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
    temp_tpr = tpr(y_true, temp_pred)
    temp_fpr = fpr(y_true, temp_pred)
    tpr_list.append(temp_tpr)
    fpr_list.append(temp_fpr)

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 7))
plt.fill_between(fpr_list, tpr_list, alpha=0.4)
plt.plot(fpr_list, tpr_list, lw=3)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.xlabel("FPR", fontsize=15)
plt.ylabel("TPR", fontsize=15)
plt.show()

from sklearn import metrics
metrics.roc_auc_score(y_true, y_pred)

tp_list = []
fp_list = []

for thresh in thresholds:
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
    temp_tp = accuracy_v2.true_positive(y_true, temp_pred)
    temp_fp = accuracy_v2.false_positive(y_true, temp_pred)
    tp_list.append(temp_tp)
    fp_list.append(temp_fp)

import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(
    header=dict(values=['thresholds',"tp_list", "fp_list"],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[thresholds, # 1st column
                       tp_list, fp_list], # 2nd column
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])

fig.update_layout(width=500, height=300)
fig.show()
