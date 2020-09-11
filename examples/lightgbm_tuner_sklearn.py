"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM tuner.

In this example, we optimize the validation log loss of cancer detection.

"""

import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import optuna.integration.lightgbm as lgb


if __name__ == "__main__":
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.25)

    clf = lgb.LGBMClassifier(
        objective="binary", metric="binary_logloss"
    )
    clf.fit(train_x, train_y, eval_set=[(val_x, val_y)])

    prediction = clf.predict(val_x)
    accuracy = accuracy_score(val_y, prediction)

    print("Accuracy = {}".format(accuracy))
