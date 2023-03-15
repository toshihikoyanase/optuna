from collections import OrderedDict

from optuna import create_study
from optuna import TrialPruned
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.search_space import intersection_search_space
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import Trial


def test_intersection_search_space() -> None:
    search_space = IntersectionSearchSpace()
    study = create_study()

    # No trial.
    assert search_space.calculate(study.trials) == {}
    assert search_space.calculate(study.trials) == intersection_search_space(study.trials)

    # Waiting trial.
    study.enqueue_trial(
        {"y": 0, "x": 5}, {"y": FloatDistribution(-3, 3), "x": IntDistribution(0, 10)}
    )
    assert search_space.calculate(study.trials) == {}
    assert search_space.calculate(study.trials) == intersection_search_space(study.trials)

    # First trial.
    study.optimize(lambda t: t.suggest_float("y", -3, 3) + t.suggest_int("x", 0, 10), n_trials=1)
    assert search_space.calculate(study.trials) == {
        "x": IntDistribution(low=0, high=10),
        "y": FloatDistribution(low=-3, high=3),
    }
    assert search_space.calculate(study.trials) == intersection_search_space(study.trials)

    # Returning sorted `OrderedDict` instead of `dict`.
    assert search_space.calculate(study.trials, ordered_dict=True) == OrderedDict(
        [
            ("x", IntDistribution(low=0, high=10)),
            ("y", FloatDistribution(low=-3, high=3)),
        ]
    )
    assert search_space.calculate(study.trials, ordered_dict=True) == intersection_search_space(
        study.trials, ordered_dict=True
    )

    # Second trial (only 'y' parameter is suggested in this trial).
    study.optimize(lambda t: t.suggest_float("y", -3, 3), n_trials=1)
    assert search_space.calculate(study.trials) == {"y": FloatDistribution(low=-3, high=3)}
    assert search_space.calculate(study.trials) == intersection_search_space(study.trials)

    # Failed or pruned trials are not considered in the calculation of
    # an intersection search space.
    def objective(trial: Trial, exception: Exception) -> float:
        trial.suggest_float("z", 0, 1)
        raise exception

    study.optimize(lambda t: objective(t, RuntimeError()), n_trials=1, catch=(RuntimeError,))
    study.optimize(lambda t: objective(t, TrialPruned()), n_trials=1)
    assert search_space.calculate(study.trials) == {"y": FloatDistribution(low=-3, high=3)}
    assert search_space.calculate(study.trials) == intersection_search_space(study.trials)

    # If two parameters have the same name but different distributions,
    # those are regarded as different parameters.
    study.optimize(lambda t: t.suggest_float("y", -1, 1), n_trials=1)
    assert search_space.calculate(study.trials) == {}
    assert search_space.calculate(study.trials) == intersection_search_space(study.trials)

    # The search space remains empty once it is empty.
    study.optimize(lambda t: t.suggest_float("y", -3, 3) + t.suggest_int("x", 0, 10), n_trials=1)
    assert search_space.calculate(study.trials) == {}
    assert search_space.calculate(study.trials) == intersection_search_space(study.trials)
