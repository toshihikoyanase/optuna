from collections import OrderedDict
import copy
from typing import Dict
from typing import List
from typing import Optional

import optuna
from optuna.distributions import BaseDistribution


class IntersectionSearchSpace:
    """A class to calculate the intersection search space of a :class:`~optuna.study.Study`.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    Args:
        include_pruned:
            Whether pruned trials should be included in the search space.
    """

    def __init__(self, include_pruned: bool = False) -> None:
        self._cursor: int = -1
        self._search_space: Optional[Dict[str, BaseDistribution]] = None

        self._include_pruned = include_pruned

    def calculate(
        self, trials: List[optuna.trial.FrozenTrial], ordered_dict: bool = False
    ) -> Dict[str, BaseDistribution]:
        """Returns the intersection search space of the given trials.

        Args:
            trials:
                A list of trials.
            ordered_dict:
                A boolean flag determining the return type.
                If :obj:`False`, the returned object will be a :obj:`dict`.
                If :obj:`True`, the returned object will be an :obj:`collections.OrderedDict`
                sorted by keys, i.e. parameter names.

        Returns:
            A dictionary containing the parameter names and parameter's distributions.

        """

        states_of_interest = [
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.WAITING,
            optuna.trial.TrialState.RUNNING,
        ]

        if self._include_pruned:
            states_of_interest.append(optuna.trial.TrialState.PRUNED)

        trials_interest = [trial for trial in trials if trial.state in states_of_interest]

        next_cursor = trials_interest[-1].number + 1 if len(trials_interest) > 0 else -1
        for trial in reversed(trials_interest):
            if self._cursor > trial.number:
                break

            if not trial.state.is_finished():
                next_cursor = trial.number
                continue

            if self._search_space is None:
                self._search_space = copy.copy(trial.distributions)
                continue

            self._search_space = {
                name: distribution
                for name, distribution in self._search_space.items()
                if trial.distributions.get(name) == distribution
            }

        self._cursor = next_cursor
        search_space = self._search_space or {}

        if ordered_dict:
            search_space = OrderedDict(sorted(search_space.items(), key=lambda x: x[0]))

        return copy.deepcopy(search_space)


def intersection_search_space(
    trials: List[optuna.trial.FrozenTrial],
    ordered_dict: bool = False,
    include_pruned: bool = False,
) -> Dict[str, BaseDistribution]:
    """Return the intersection search space of the given trials.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    .. note::
        :class:`~optuna.search_space.IntersectionSearchSpace` provides the same functionality with
        a much faster way. Please consider using it if you want to reduce execution time
        as much as possible.

    Args:
        trials:
            A list of trials.
        ordered_dict:
            A boolean flag determining the return type.
            If :obj:`False`, the returned object will be a :obj:`dict`.
            If :obj:`True`, the returned object will be an :obj:`collections.OrderedDict` sorted by
            keys, i.e. parameter names.
        include_pruned:
            Whether pruned trials should be included in the search space.

    Returns:
        A dictionary containing the parameter names and parameter's distributions.
    """

    return IntersectionSearchSpace(include_pruned=include_pruned).calculate(
        trials, ordered_dict=ordered_dict
    )
