import copy
from typing import Any, cast, Callable, Container, Dict, Iterable, List, Optional, Sequence, Type, Union

from optuna import logging
from optuna import pruners
from optuna import samplers
from optuna import storages
from optuna.distributions import BaseDistribution
from optuna.study.study import Study
from optuna.study.study import create_study
from optuna.study.study import ObjectiveFuncType
from optuna.study._multi_objective import _get_pareto_front_trials_by_trials
from optuna.study import StudyDirection
from optuna import trial as trial_module

_logger = logging.get_logger(__name__)


class StudyView(Study):
    def __init__(self, study: Study) -> None:
        self._directions = study.directions
        self._trials = study.trials
        self._user_attrs = study.user_attrs
        self._system_attrs = study.system_attrs
    
    def __getstate__(self) -> Dict[Any, Any]:
        pass

    def __setstate__(self, state: Dict[Any, Any]) -> None:
        pass

    @property
    def trials(self) -> List[trial_module.FrozenTrial]:
        return self._trials
    
    @trials.setter
    def trials(self, value: List[trial_module.FrozenTrial]) -> None:
        self._trials = value
    
    @property
    def directions(self) -> List[StudyDirection]:
        return self._directions
    
    @directions.setter
    def directions(self, value: List[StudyDirection]) -> None:
        self._directions = value
    
    @property
    def user_attrs(self) -> Dict[str, Any]:
        return self._user_attrs
    
    @user_attrs.setter
    def user_attrs(self, value: Dict[str, Any]) -> None:
        self._user_attrs = value
    
    @property
    def system_attrs(self) -> Dict[str, Any]:
        return self._system_attrs

    @system_attrs.setter
    def system_attrs(self, value: Dict[str, Any]) -> None:
        self._system_attrs = value

    @property
    def best_trial(self) -> trial_module.FrozenTrial:
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
    def best_trials(self) -> List[trial_module.FrozenTrial]:
        _get_pareto_front_trials_by_trials(
            self.trials, self.directions
        )
    
    def get_trials(
        self,
        deepcopy: bool = True,
        states: Optional[Container[trial_module.TrialState]] = None,
    ) -> List[trial_module.FrozenTrial]:
        if states:
            _ts = [t for t in self.trials if t.state in states]
        else:
            _ts = self.trials
        if deepcopy:
            _ts = copy.deepcopy(_ts)
        return _ts
    
    def optimize(
        self,
        func: ObjectiveFuncType,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        catch: Union[Iterable[Type[Exception]], Type[Exception]] = (),
        callbacks: Optional[List[Callable[["Study", trial_module.FrozenTrial], None]]] = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        raise NotImplementedError()

    def ask(
        self, fixed_distributions: Optional[Dict[str, BaseDistribution]] = None
    ) -> trial_module.Trial:
        raise NotImplementedError()

    def tell(
        self,
        trial: Union[trial_module.Trial, int],
        values: Optional[Union[float, Sequence[float]]] = None,
        state: Optional[trial_module.TrialState] = None,
        skip_if_finished: bool = False,
    ) -> trial_module.FrozenTrial:
        raise NotImplementedError()

    def stop(self) -> None:
        raise NotImplementedError()

    def enqueue_trial(
        self,
        params: Dict[str, Any],
        user_attrs: Optional[Dict[str, Any]] = None,
        skip_if_exists: bool = False,
    ) -> None:
        raise NotImplementedError()

    def add_trial(self, trial: trial_module.FrozenTrial) -> None:
        self.trials.append(trial)
    
    def save_as_study(
        self,
        *,
        study_name: Optional[str] = None,
        storage: Optional[Union[str, storages.BaseStorage]] = None,
        sampler: Optional["samplers.BaseSampler"] = None,
        pruner: Optional[pruners.BasePruner] = None,
    ) -> Study:
        study = create_study(
            storage=storage,
            study_name=study_name,
            directions=self.directions,
            sampler=sampler,
            pruner=pruner,
        )
        study.add_trials(self.trials)
        return study