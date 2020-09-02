import math
import types
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np

import optuna
from optuna._experimental import experimental
import optuna.logging
from optuna.multi_objective.trial import FrozenMultiObjectiveTrial

ObjectiveFuncType = Callable[
    ["optuna.batch.multi_objective.trial.BatchMultiObjectiveTrial"], Sequence[np.ndarray]
]
CallbackFuncType = Callable[
    [
        "optuna.multi_objective.study.MultiObjectiveStudy",
        "optuna.multi_objective.trial.FrozenMultiObjectiveTrial",
    ],
    None,
]

_logger = optuna.logging.get_logger(__name__)


class _ObjectiveCallbackWrapper(object):
    def __init__(
        self,
        study: "optuna.multi_objective.study.MultiObjectiveStudy",
        objective: ObjectiveFuncType,
        batch_size: int,
    ):
        self._study = study
        self._batch_size = batch_size
        self._objective = objective
        self._members = {}  # type: Dict[int, List[int]]

    def batch_objective(
        self, trial: "optuna.multi_objective.trial.MultiObjectiveTrial"
    ) -> Sequence[float]:
        trials = [trial]
        # Assume storage has already been synchronized.
        self._members[trial._trial._trial_id] = []
        for _ in range(self._batch_size - 1):
            trial_id = self._study._study._pop_waiting_trial_id()
            if trial_id is None:
                trial_id = self._study._storage.create_new_trial(self._study._study_id)
            self._members[trial._trial._trial_id].append(trial_id)
            new_trial = optuna.trial.Trial(self._study._study, trial_id)
            trials.append(optuna.multi_objective.trial.MultiObjectiveTrial(new_trial))
        batch_trial = optuna.batch.multi_objective.trial.BatchMultiObjectiveTrial(trials)
        try:
            results = self._objective(batch_trial)
            transposed_results = np.array(results).transpose()
        except optuna.exceptions.TrialPruned as e:
            for trial in trials:
                trial_id = trial._trial._trial_id
                message = "Trial {} pruned. {}".format(trial.number, str(e))
                _logger.info(message)

                # Register the last intermediate value if present as the value of the trial.
                # TODO(hvy): Whether a pruned trials should have an actual value can be
                # discussed.
                frozen_trial = self._study._storage.get_trial(trial_id)
                last_step = frozen_trial.last_step
                if last_step is not None:
                    self._study._storage.set_trial_value(
                        trial_id, frozen_trial.intermediate_values[last_step]
                    )
                self._study._storage.set_trial_state(trial_id, optuna.trial.TrialState.PRUNED)
            raise
        except Exception as e:
            for trial in trials:
                message = "Trial {} failed because of the following error: {}".format(
                    trial.number, repr(e)
                )
                _logger.warning(message, exc_info=True)
                trial_id = trial._trial._trial_id
                self._study._storage.set_trial_system_attr(trial_id, "fail_reason", message)
                self._study._storage.set_trial_state(trial_id, optuna.trial.TrialState.FAIL)
            raise

        for trial, result in zip(trials, transposed_results):
            trial_id = trial._trial._trial_id
            trial._report_complete_values(result)
            _logger.info(
                "Trial {} finished with values: {} with parameters: {}.".format(
                    trial._trial.number, result, trial._trial.params
                )
            )
            # Set dummy objective value.
            self._study._storage.set_trial_value(trial_id, 0)
            self._study._storage.set_trial_state(trial_id, optuna.trial.TrialState.COMPLETE)
        return transposed_results[0]

    def wrap_callback(self, callback: CallbackFuncType) -> CallbackFuncType:
        def _callback(
            study: "optuna.multi_objective.study.MultiObjectiveStudy",
            trial: "optuna.multi_objective.trial.FrozenMultiObjectiveTrial",
        ) -> None:
            callback(study, trial)
            for member_id in self._members[trial._trial_id]:
                _trial = study._study._storage.get_trial(member_id)
                mo_trial = FrozenMultiObjectiveTrial(study.n_objectives, _trial)
                callback(study, mo_trial)

        return _callback


class BatchMultiObjectiveStudy(object):
    def __init__(self, study: "optuna.multi_objective.study.MultiObjectiveStudy", batch_size: int):
        self._study = study
        self._batch_size = batch_size

    @property
    def n_objectives(self) -> int:
        """Return the number of objectives.

        Returns:
            Number of objectives.
        """

        return self._study.n_objectives

    @property
    def directions(self) -> List["optuna.study.StudyDirection"]:
        """Return the optimization direction list.

        Returns:
            A list that contains the optimization direction for each objective value.
        """

        return self._study.directions

    @property
    def sampler(self) -> "optuna.multi_objective.samplers.BaseMultiObjectiveSampler":
        """Return the sampler.

        Returns:
            A :class:`~multi_objective.samplers.BaseMultiObjectiveSampler` object.
        """

        return self._study.sampler

    def optimize(
        self,
        objective: ObjectiveFuncType,
        timeout: Optional[int] = None,
        n_batches: Optional[int] = None,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[CallbackFuncType]] = None,
        gc_after_trial: bool = True,
        show_progress_bar: bool = False,
    ) -> None:

        wrapper = _ObjectiveCallbackWrapper(self._study, objective, self._batch_size)

        if callbacks is None:
            wrapped_callbacks = None
        else:
            wrapped_callbacks = [wrapper.wrap_callback(callback) for callback in callbacks]

        n_trials = math.ceil(n_batches / n_jobs) if n_batches is not None else None

        try:
            self._study._study._org_run_trial = self._study._study._run_trial  # type: ignore
            self._study._study._run_trial = types.MethodType(  # type: ignore
                _run_trial,
                self._study._study,
            )
            self._study.optimize(
                wrapper.batch_objective,
                timeout=timeout,
                n_trials=n_trials,
                n_jobs=n_jobs,
                catch=catch,
                callbacks=wrapped_callbacks,
                gc_after_trial=gc_after_trial,
                show_progress_bar=show_progress_bar,
            )
        finally:
            self._study._study._run_trial = self._study._study._org_run_trial  # type: ignore
            pass

    def get_pareto_front_trials(
        self,
    ) -> List["optuna.multi_objective.trial.FrozenMultiObjectiveTrial"]:
        """Return trials located at the pareto front in the study.

        A trial is located at the pareto front if there are no trials that dominate the trial.
        It's called that a trial ``t0`` dominates another trial ``t1`` if
        ``all(v0 <= v1) for v0, v1 in zip(t0.values, t1.values)`` and
        ``any(v0 < v1) for v0, v1 in zip(t0.values, t1.values)`` are held.

        Returns:
            A list of :class:`~optuna.multi_objective.trial.FrozenMultiObjectiveTrial` objects.
        """
        return self._study.get_pareto_front_trials()


def _run_trial(
    self: "optuna.study.Study",
    func: "optuna.study.ObjectiveFuncType",
    catch: Tuple[Type[Exception], ...],
    gc_after_trial: bool,
) -> "optuna.trial.Trial":

    # Sync storage once at the beginning of the objective evaluation.
    self._storage.read_trials_from_remote_storage(self._study_id)

    trial_id = self._pop_waiting_trial_id()
    if trial_id is None:
        trial_id = self._storage.create_new_trial(self._study_id)
    trial = optuna.trial.Trial(self, trial_id)
    func(trial)
    return trial


@experimental("2.1.0")
def create_study(
    directions: List[str],
    study_name: Optional[str] = None,
    storage: Optional[Union[str, "optuna.storages.BaseStorage"]] = None,
    sampler: Optional["optuna.multi_objective.samplers.BaseMultiObjectiveSampler"] = None,
    load_if_exists: bool = False,
    batch_size: int = 1,
) -> BatchMultiObjectiveStudy:

    study = optuna.multi_objective.create_study(
        directions, study_name, storage, sampler, load_if_exists
    )
    return BatchMultiObjectiveStudy(study, batch_size)


@experimental("2.1.0")
def load_study(
    study_name: str,
    storage: Optional[Union[str, "optuna.storages.BaseStorage"]] = None,
    sampler: Optional["optuna.multi_objective.samplers.BaseMultiObjectiveSampler"] = None,
    batch_size: int = 1,
) -> BatchMultiObjectiveStudy:

    study = optuna.multi_objective.load_study(study_name, storage, sampler)
    return BatchMultiObjectiveStudy(study, batch_size)