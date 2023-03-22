from collections import OrderedDict
import copy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import optuna
from optuna.distributions import BaseDistribution
from optuna.study import Study


class IntersectionSearchSpace:
    """A class to calculate the intersection search space of a :class:`~optuna.study.Study`.

    Intersection search space contains the intersection of parameter distributions that have been
    suggested in the completed trials of the study so far.
    If there are multiple parameters that have the same name but different distributions,
    neither is included in the resulting search space
    (i.e., the parameters with dynamic value ranges are excluded).

    Note that an instance of this class is supposed to be used for only one study.
    If different studies are passed to
    :func:`~optuna.search_space.IntersectionSearchSpace.calculate`,
    a :obj:`ValueError` is raised.

    Args:
        include_pruned:
            Whether pruned trials should be included in the search space.
    """

    def __init__(self, include_pruned: bool = False) -> None:
        self._cursor: int = -1
        self._search_space: Optional[Dict[str, BaseDistribution]] = None
        self._study_id: Optional[int] = None

        self._include_pruned = include_pruned

    def calculate(
        self, study: Study, ordered_dict: bool = False
    ) -> Dict[str, BaseDistribution]:
        """Returns the intersection search space of the given trials.

        Args:
            study:
                A study with completed trials. The same study must be passed for one instance
                of this class through its lifetime.
            ordered_dict:
                A boolean flag determining the return type.
                If :obj:`False`, the returned object will be a :obj:`dict`.
                If :obj:`True`, the returned object will be an :obj:`collections.OrderedDict`
                sorted by keys, i.e. parameter names.

        Returns:
            A dictionary containing the parameter names and parameter's distributions.

        """

        if self._study_id is None:
            self._study_id = study._study_id
        else:
            # Note that the check below is meaningless when `InMemoryStorage` is used
            # because `InMemoryStorage.create_new_study` always returns the same study ID.
            if self._study_id != study._study_id:
                raise ValueError("`IntersectionSearchSpace` cannot handle multiple studies.")

        search_space, self._search_space, self._cursor = _calculate(
            study.trials, ordered_dict, self._include_pruned, self._cursor, self._search_space
        )
        return search_space


def _calculate(
    trials: List[optuna.trial.FrozenTrial],
    ordered_dict: bool = False,
    include_pruned: bool = False,
    cursor: int = -1,
    search_space: Optional[Dict[str, BaseDistribution]] = None,
) -> Tuple[Dict[str, BaseDistribution], Optional[Dict[str, BaseDistribution]], int]:

    states_of_interest = [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.WAITING,
        optuna.trial.TrialState.RUNNING,
    ]

    _search_space = copy.deepcopy(search_space)

    if include_pruned:
        states_of_interest.append(optuna.trial.TrialState.PRUNED)

    trials_of_interest = [trial for trial in trials if trial.state in states_of_interest]

    next_cursor = trials[-1].number + 1 if len(trials) > 0 else -1
    for trial in reversed(trials_of_interest):
        if cursor > trial.number:
            break

        if not trial.state.is_finished():
            next_cursor = trial.number
            continue

        if _search_space is None:
            _search_space = copy.copy(trial.distributions)
            continue

        _search_space = {
            name: distribution
            for name, distribution in _search_space.items()
            if trial.distributions.get(name) == distribution
        }

    search_space = _search_space or {}

    if ordered_dict:
        search_space = OrderedDict(sorted(search_space.items(), key=lambda x: x[0]))

    return search_space, _search_space, next_cursor


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

    search_space, _, _ = _calculate(trials, ordered_dict, include_pruned)
    return search_space
