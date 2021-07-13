import os
import argparse

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

import config
import model_dispatcher


def run(fold, model):

    # read the data that has been processed
    # with create_folds.py
    df = pd.read_csv(config.TRAINING_FILE)
    
    # split the data into train and valid, using the fold 
    # variable passed into this function
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # separate the features from the target
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    # initialize the classifier
    clf = model_dispatcher.models[model]

    # fit the model on the train set 
    clf.fit(x_train, y_train)

    # predict on the validation set
    preds = clf.predict(x_valid)

    # calculate accurracy on the validation set
    accuracy = metrics.accuracy_score(y_valid, preds)

    print(f"Fold={fold}, Accuracy={accuracy}")

    # save the model 
    joblib.dump(
        clf, 
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )

if __name__ == "__main__":

    # initialize an ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the neccessary arguments
    parser.add_argument(
        "--fold",
        type=int 
    )

    parser.add_argument(
        "--model",
        type=str
    )

    #read the arguement from the command line
    args = parser.parse_args()

    #run the fold supplied in the command line argument
    run(
        fold=args.fold,
        model=args.model
    )



