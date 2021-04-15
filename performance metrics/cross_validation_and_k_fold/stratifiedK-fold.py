import pandas as pd
from sklearn import model_selection
import seaborn as sns

df = pd.read_csv("/home/felix/PycharmProjects/pythonProject/data/winequality-red.csv")
df["kfold"] = -1

df = df.sample(frac=1).reset_index(drop=True)

kf = model_selection.StratifiedKFold(n_splits=5)

for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.quality)):
    df.loc[val_, "kfold"] = fold

df.to_csv("train_Stratifiedfolds.csv", index=False)
kfold1 = df[df["kfold"] == 2] #for testing kfold validity

b = sns.countplot(x="quality", data=df)
b.set_xlabel("quality", fontsize=20)
b.set_ylabel("count", fontsize=20)

import matplotlib.pyplot as plt

plt.show()#shows plots dunnow why
