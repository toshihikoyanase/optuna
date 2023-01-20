import copy
from typing import Any
from typing import cast
from typing import Container
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

from optuna import logging
from optuna.study._multi_objective import _get_pareto_front_trials_by_trials
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = logging.get_logger(__name__)


class FrozenStudy:
    """Basic attributes of a :class:`~optuna.study.Study`.

    This class is private and not referenced by Optuna users.

    Attributes:
        study_name:
            Name of the :class:`~optuna.study.Study`.
        direction:
            :class:`~optuna.study.StudyDirection` of the :class:`~optuna.study.Study`.

            .. note::
                This attribute is only available during single-objective optimization.
        directions:
            A list of :class:`~optuna.study.StudyDirection` objects.
        user_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.study.Study` set with
            :func:`optuna.study.Study.set_user_attr`.
        system_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.study.Study` internally
            set by Optuna.

    """

    def __init__(
        self,
        study_name: str,
        direction: Optional[StudyDirection],
        user_attrs: Dict[str, Any],
        system_attrs: Dict[str, Any],
        study_id: int,
        *,
        directions: Optional[Sequence[StudyDirection]] = None,
        trials: Optional[List[FrozenTrial]] = None,
    ):
        self.study_name = study_name
        if direction is None and directions is None:
            raise ValueError("Specify one of `direction` and `directions`.")
        elif directions is not None:
            self._directions = list(directions)
        elif direction is not None:
            self._directions = [direction]
        else:
            raise ValueError("Specify only one of `direction` and `directions`.")
        self.user_attrs = user_attrs
        self.system_attrs = system_attrs
        self._study_id = study_id
        self.trials = trials

    def __eq__(self, other: Any) -> bool:

        if not isinstance(other, FrozenStudy):
            return NotImplemented

        return other.__dict__ == self.__dict__

    def __lt__(self, other: Any) -> bool:

        if not isinstance(other, FrozenStudy):
            return NotImplemented

        return self._study_id < other._study_id

    def __le__(self, other: Any) -> bool:

        if not isinstance(other, FrozenStudy):
            return NotImplemented

        return self._study_id <= other._study_id

    @property
    def direction(self) -> StudyDirection:

        if len(self._directions) > 1:
            raise RuntimeError(
                "This attribute is not available during multi-objective optimization."
            )

        return self._directions[0]

    @property
    def directions(self) -> List[StudyDirection]:

        return self._directions

    def _is_multi_objective(self) -> bool:

        return len(self.directions) > 1

    def get_trials(
        self,
        deepcopy: bool = True,
        states: Optional[Container[TrialState]] = None,
    ) -> List[FrozenTrial]:
        if states:
            _ts = [t for t in self.trials if t.state in states]
        else:
            _ts = self.trials
        if deepcopy:
            _ts = copy.deepcopy(_ts)
        return _ts

    @property
    def best_trial(self) -> FrozenTrial:
        directions = self.directions
        all_trials = self.trials
        if len(directions) > 1:
            raise RuntimeError(
                "Best trial can be obtained only for single-objective optimization."
            )
        direction = directions[0]

        if direction == StudyDirection.MAXIMIZE:
            best_trial = max(all_trials, key=lambda t: cast(float, t.value))
        else:
            best_trial = min(all_trials, key=lambda t: cast(float, t.value))

        return best_trial

    @property
    def best_trials(self) -> List[FrozenTrial]:
        _get_pareto_front_trials_by_trials(
            self.trials, self.directions
        )
