from typing import Callable
from typing import Optional
from typing import Sequence

import numpy as np

from optuna._study_direction import StudyDirection
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


def plot_optimization_history(
    study: Study,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
) -> "go.Figure":
    """Plot optimization history of all trials in a study.

    Example:

        The following code snippet shows how to plot optimization history.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)

            fig = optuna.visualization.plot_optimization_history(study)
            fig.show()

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label and the legend.
        constraints_func:
            An optional function that computes the objective constraints. It returns a sequence
            of :obj:`float` values, and a value strictly larger than 0 means that a constraint
            is violated. The trials violate the constraints, they are excluded from the best
            trials and they will be plotted as gray points. See also the references of the
            samplers which support constraints optimization such as :class:`~optuna.samplers.
            NSGAIISampler` and :class:`~optuna.integration.BoTorchSampler`.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.

    Raises:
        :exc:`ValueError`:
            If ``target`` is :obj:`None` and ``study`` is being used for multi-objective
            optimization.
    """

    _imports.check()
    _check_plot_args(study, target, target_name)
    return _get_optimization_history_plot(study, target, target_name, constraints_func)


def _get_optimization_history_plot(
    study: Study,
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]],
) -> "go.Figure":

    layout = go.Layout(
        title="Optimization History Plot",
        xaxis={"title": "#Trials"},
        yaxis={"title": target_name},
    )

    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Study instance does not contain trials.")
        return go.Figure(data=[], layout=layout)

    target_func = (lambda t: t.value) if target is None else target

    traces = []
    if constraints_func is not None:
        feasible_trials = []
        infeasible_trials = []
        for trial in trials:
            if all(c <=0 for c in constraints_func(trial)):
                feasible_trials.append(trial)
            else:
                infeasible_trials.append(trial)
        trials = feasible_trials
        if len(infeasible_trials) > 0:
            traces.append(go.Scatter(
                x=[t.number for t in infeasible_trials],
                y=[target_func(t) for t in infeasible_trials],
                mode="markers",
                name=f"{target_name} (infeasible)",
                marker_color='LightGray'
            ))

    if len(trials) > 0:
        traces.insert(0,
            go.Scatter(
                x=[t.number for t in trials],
                y=[target_func(t) for t in trials],
                mode="markers",
                name=target_name,
            )
        )

    if target is None:
        if study.direction == StudyDirection.MINIMIZE:
            best_values = np.minimum.accumulate([t.value for t in trials])
        else:
            best_values = np.maximum.accumulate([t.value for t in trials])
        if len(best_values) > 0:
            traces.insert(1,
                go.Scatter(x=[t.number for t in trials], y=best_values, name="Best Value"),
            )

    figure = go.Figure(data=traces, layout=layout)

    return figure
