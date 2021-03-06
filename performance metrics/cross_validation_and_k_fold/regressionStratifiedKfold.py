# stratified k-fold for regression
import numpy as np
import pandas as pd

from sklearn import datasets, model_selection


def create_folds(data):
    data["kfold"] = -1

    data = data.sample(frac=1.).reset_index(drop=True)
    num_bins = int(np.floor(1 + np.log2(len(data))))
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )

    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, "kfold"] = f

    data = data.drop("bins", axis=1)
    return data


X, y = datasets.make_regression(n_samples=15000, n_features=100, n_targets=1)
df = pd.DataFrame(
    X,
    columns=[f"f_{i}" for i in range(X.shape[1])]
)
df.loc[:, "target"] = y
df = create_folds(df)
print(df)
