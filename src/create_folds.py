import os

import pandas as pd 
from sklearn import model_selection

import config


def create_folds(target):
    df = pd.read_csv(config.RAW_CSV)
    df["kfold"] = -1
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    y = df[target].values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (train, valid) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid, "kfold"] = fold

    df.to_csv(
        f"{os.path.splitext(config.RAW_CSV)[0]}_folds.csv", 
        index=False
    )


if __name__ == "__main__":
    create_folds("label")


    