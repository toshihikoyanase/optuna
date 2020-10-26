"""
This configurable Optuna example demonstrates the use of pruning when using
custom cross-validation functions based on LightGBM classifiers.

In this example, we optimize cross-validation accuracy of a classification model
of breast cancer probability with a custom CV function custom_cv_fun() applying
LightGBM's .fit() method to each cross-validation fold separately, instead of using
the standard built-in .cv() function.

The example emphasizes reproducibility, giving the user precise control over all seed
values that are being used to let her arrive at the same results over repeated studies.

You can run this example as follows:
    $ python lightgbm_with_custom_cv_function.py

"""

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold

import optuna


def custom_cv_fun(
    lgbm_params,
    X,
    y,
    eval_metric="auc",
    eval_name="valid",
    n_estimators=100,
    early_stopping_rounds=10,
    nfold=5,
    random_state=123,
    callbacks=[],
    verbose=False,
):

    # create placeholders for results
    fold_best_iterations = []
    fold_best_scores = []

    # get feature names
    feature_names = list(X.columns)

    # split data into k folds
    kf = KFold(n_splits=nfold, shuffle=True, random_state=random_state)

    # iterate over folds
    for train_index, valid_index in kf.split(X, y):

        # subset train and valid (out-of-fold) parts of the fold
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model = LGBMClassifier(n_estimators=n_estimators, random_state=random_state)

        # pass hyperparameters dict to the estimator
        model.set_params(**lgbm_params)

        # train the model
        model.fit(
            X_train,
            y_train.values.ravel(),
            eval_set=(X_valid, y_valid.values.ravel()),
            eval_metric=[eval_metric],  # note: a list required
            eval_names=[eval_name],  # note: a list required
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            feature_name=feature_names,
            callbacks=callbacks,
        )

        # collect current fold data
        fold_best_iterations.append(model.best_iteration_)
        fold_best_scores.append(model.best_score_[eval_name])

    # average folds iterations numbers
    folds_data = {}
    folds_data["best_iterations_mean"] = int(np.mean(fold_best_iterations))

    # collect metrics for best scores in each fold
    fold_best_score = {}
    for metric in fold_best_scores[0].keys():
        fold_best_score[metric] = [fold[metric] for fold in fold_best_scores]

    # avearage folds metrics (for all metrics)
    for metric in fold_best_scores[0].keys():
        folds_data["eval_mean-" + metric] = np.mean(fold_best_score[metric])

    return {
        "folds_mean_data": folds_data,
        "feature_names": feature_names,
        "fold_best_iter": fold_best_iterations,
        "fold_best_score": fold_best_score,
    }


def load_data():
    data, target = datasets.load_breast_cancer(return_X_y=True)

    # Convert numpy arrays to DataFrames.
    train_x_df = pd.DataFrame(data)
    train_x_df.columns = ["col_{}".format(i) for i in train_x_df.columns]
    train_y_df = pd.DataFrame({"y": target})

    return train_x_df, train_y_df


class ObjectiveCustom:
    def __init__(
        self,
        train_x_df,
        train_y_df,
        objective="binary",
        eval_metric="auc",
        eval_name="valid",
        folds=5,
        n_jobs=1,
        seed=123,
        verbosity=-1,
    ):

        self.train_x_df = train_x_df
        self.train_y_df = train_y_df
        self.objective = objective
        self.eval_metric = eval_metric
        self.eval_name = eval_name
        self.folds = folds
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbosity = verbosity

    def __call__(self, trial):
        params = {
            "bagging_fraction": float(trial.suggest_float("bagging_fraction", 0.1, 1, log=False)),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 100, log=True),
            "feature_fraction": float(trial.suggest_float("feature_fraction", 0.1, 1, log=False)),
            "lambda_l1": float(trial.suggest_float("lambda_l1", 1e-06, 100, log=True)),
            "lambda_l2": float(trial.suggest_float("lambda_l2", 1e-06, 100, log=True)),
            "learning_rate": float(trial.suggest_float("learning_rate", 0.0001, 0.3, log=True)),
            "num_leaves": trial.suggest_int("num_leaves", 3, 100, log=True),
        }
        params.update(self.static_params)

        # add a LightGBM callback for pruning
        lgbm_pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, metric=self.eval_metric, valid_name=self.eval_name
        )

        # random boosting and stopping rounds will be passed as arguments
        num_boost_round = trial.suggest_int("num_boost_round", 100, 2000, log=False)
        early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 2, 500, log=True)

        # train the model using custom function
        cv_results_dict = custom_cv_fun(
            lgbm_params=params,
            X=self.train_x_df,
            y=self.train_y_df,
            eval_metric=self.eval_metric,
            eval_name=self.eval_name,
            n_estimators=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            nfold=self.folds,
            random_state=self.seed,
            callbacks=[lgbm_pruning_callback],
            verbose=self.verbosity,
        )

        # get mean CV metric from the appropriate key
        # of the dict returned by the custom CV function
        eval_mean_metric = cv_results_dict["folds_mean_data"]["eval_mean-" + self.eval_metric]

        print("Mean CV metric: %.5f" % eval_mean_metric)

        return eval_mean_metric

    @property
    def static_params(self):
        return {
            "boosting_type": "gbdt",
            "objective": self.objective,
            "metric": self.eval_metric,
            "n_jobs": self.n_jobs,
            "seed": self.seed,
            "verbose": self.verbosity,
        }


if __name__ == "__main__":
    train_x_df, train_y_df = load_data()

    objective = ObjectiveCustom(
        train_x_df,
        train_y_df,
        objective="binary",
        eval_metric="auc",
        eval_name="valid",
        folds=5,
        # The number of threads to be used by lightgbm in each of the workers.
        n_jobs=2,
        # Fix seed model training (lgbm.fit()).
        seed=123,
        verbosity=-1,
    )
    # Note that we fix sampler seed to make the sampling process deterministic.
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=123), direction="maximize")
    study.optimize(objective, n_trials=20)

    # Append static parameters not returned by Optuna.
    all_best_params = {**study.best_trial.params, **objective.static_params}

    print("\nBest mean {}: {:.5f}\n".format(objective.eval_metric, study.best_value))

    print("Optuna-optimized best hyperparameters: ")
    for name, value in all_best_params.items():
        print("    {}: {}".format(name, value))
