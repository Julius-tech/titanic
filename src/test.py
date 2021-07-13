import os
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
import joblib
import numpy as np

import config
import model_dispatcher


def predict():
    df = pd.read_csv(config.TESTING_FILE)
    y_test = df["label"].values

    Accuracy = []

    for FOLD in range(5):
        df = pd.read_csv(config.TESTING_FILE)
        clf = joblib.load(os.path.join(config.MODEL_OUTPUT, f"dt_{FOLD}.bin"))
        
        preds = clf.predict(df)

        accuracy = metrics.accuracy_score(y_test, preds)

        Accuracy.append(accuracy)

    return np.mean(Accuracy)
    

if __name__ == "__main__":
    avg_acc = predict()
    print("Average Test Accuracy: ", avg_acc)
    #submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    #submission.to_csv(f"models/rf_submission.csv", index=False)