"""
Optuna example that demonstrates a pruner for LightGBM.

In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters. Throughout
training of models, a pruner observes intermediate results and stop unpromising trials.

You can run this example as follows:
    $ python lightgbm_integration.py

"""
from collections import defaultdict
import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)

    dtrain = lgb.Dataset(train_x, label=train_y)
    dtest = lgb.Dataset(test_x, label=test_y)

    param = {
        'objective': 'binary',
        'metric': 'binary_error',
        'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
        'num_leaves': trial.suggest_int('num_leaves', 2, 5000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
        'max_bin': trial.suggest_int('max_bin', 2, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 1000),
    }

    if param['boosting_type'] == 'dart':
        param['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
        param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
    if param['boosting_type'] == 'goss':
        param['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
        param['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - param['top_rate'])

    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'binary_error')
    gbm = lgb.train(
        param, dtrain, valid_sets=[dtest], verbose_eval=False, callbacks=[pruning_callback])

    preds = gbm.predict(test_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
    return 1.0 - accuracy


def run_once(label, n_trials, sampler, pruner):
    study = optuna.create_study(pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    print('=== {} ==='.format(label))
    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    steps = defaultdict(int)
    losses = defaultdict(float)
    for t in study.trials:
        if t.state is not optuna.structs.TrialState.PRUNED:
            continue

        last_step = max(t.intermediate_values.keys())
        steps[last_step] += 1
        losses[last_step] += t.intermediate_values[last_step]

    print('Pruned statistics:')
    for step, count in sorted(list(steps.items())):
        print("  step[{}]: \tpruned_count={}, \tpruned_loss_avg={}".format(
            step, count, losses[step] / count))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    # print('  Params: ')
    # for key, value in trial.params.items():
    #     print('    {}: {}'.format(key, value))
    print('')


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = 500
    xs = [
        ("TPE/ASHA (PRUNED+COMPLETE, startup=2)",
         optuna.samplers.TPESampler(include_pruned=True, n_startup_trials=2),
         optuna.pruners.SuccessiveHalvingPruner()),
        ("TPE/ASHA (COMPLETE)", optuna.samplers.TPESampler(
            include_pruned=False, n_startup_trials=2), optuna.pruners.SuccessiveHalvingPruner()),
        ("Random/ASHA", optuna.samplers.RandomSampler(), optuna.pruners.SuccessiveHalvingPruner()),
        ("TPE/ASHA (PRUNED+COMPLETE, startup_trials=50)",
         optuna.samplers.TPESampler(include_pruned=True, n_startup_trials=50),
         optuna.pruners.SuccessiveHalvingPruner()),
    ]
    for (label, sampler, pruner) in xs:
        run_once(label, n_trials, sampler, pruner)
