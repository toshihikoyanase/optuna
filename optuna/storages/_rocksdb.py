import copy
from datetime import datetime
import pickle
import threading
from typing import Any
from typing import Container
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

from optuna import distributions
from optuna import exceptions
from optuna import logging
from optuna._imports import try_import
from optuna.storages import BaseStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.distributions import BaseDistribution
from optuna.study import StudyDirection
from optuna.study import StudySummary
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = logging.get_logger(__name__)


with try_import() as _imports:
    import rocksdb


class RocksDBStorage(BaseStorage):
    def __init__(self, path: str) -> None:
        _imports.check()

        options = rocksdb.Options(
            create_if_missing=True,
        )
        self._db = rocksdb.DB(path, options)
        self._lock = threading.RLock()

    @staticmethod
    def _key_study_name(study_name: str) -> bytes:

        return f"study_id:study_name:{study_name}".encode()

    @staticmethod
    def _key_study_summary(study_id: int) -> bytes:

        return f"study_summary:study_id:{study_id:010d}".encode()

    @staticmethod
    def _key_study_param_distribution(study_id: int) -> bytes:

        return f"param_distribution:study_id:{study_id:010d}".encode()

    @staticmethod
    def _key_trial(trial_id: int) -> bytes:

        return f"frozentrial:trial_id:{trial_id:010d}".encode()

    @staticmethod
    def _key_study_ids() -> bytes:

        return f"study_ids".encode()

    @staticmethod
    def _key_trial_ids(study_id: int) -> bytes:

        return f"trial_ids:study_id:{study_id:010d}".encode()


    def _check_study_id(self, study_id: int) -> None:
        exist, _ = self._db.key_may_exist(self._key_study_summary(study_id))
        if not exist:
            raise KeyError(f"study_id {study_id} does not exist.")

    def _check_trial_id(self, trial_id: int) -> None:
        exist, _ = self._db.key_may_exist(self._key_trial(trial_id))
        if not exist:
            raise KeyError(f"trial_id {trial_id} does not exist.")

    def create_new_study(self, study_name: Optional[str] = None) -> int:
        with self._lock:
            if study_name is not None and self._db.get(self._key_study_name(study_name)) is not None:
                raise exceptions.DuplicatedStudyError

            study_counter_pkl = self._db.get(b"study_counter")
            study_id = 0 if study_counter_pkl is None else pickle.loads(study_counter_pkl)
            self._db.put(b"study_counter", pickle.dumps(study_id + 1))

            if study_name is None:
                study_name = f"{DEFAULT_STUDY_NAME_PREFIX}{study_id:010d}"
            self._db.put(self._key_study_name(study_name), pickle.dumps(study_id))

            study_summary = StudySummary(
                study_name=study_name,
                direction=StudyDirection.NOT_SET,
                best_trial=None,
                user_attrs={},
                system_attrs={},
                n_trials=0,
                datetime_start=None,
                study_id=study_id,
            )
            self._db.put(self._key_study_summary(study_id), pickle.dumps(study_summary))

            study_ids_pkl = self._db.get(self._key_study_ids())
            study_ids = [] if study_ids_pkl is None else pickle.loads(study_ids_pkl)
            study_ids.append(study_id)
            self._db.put(self._key_study_ids(), pickle.dumps(study_ids))
            self._db.put(self._key_trial_ids(study_id), pickle.dumps([]))
            self._db.put(self._key_study_param_distribution(study_id), pickle.dumps({}))

            _logger.info(f"A new study created in RocksDB with name: {study_name}")
            return study_id

    def delete_study(self, study_id: int) -> None:

        self._check_study_id(study_id)

        with self._lock:
            # Delete trials.
            for trial in self.get_all_trials(study_id, deepcopy=False):
                self._db.delete(self._key_trial(trial._trial_id))
            self._db.delete(self._key_trial_ids(study_id))

            # Delete study.
            study_name = self.get_study_name_from_id(study_id)
            self._db.delete(self._key_study_name(study_name))
            self._db.delete(self._key_study_summary(study_id))
            self._db.delete(self._key_study_param_distribution(study_id))

            study_ids_pkl = self._db.get(self._key_study_ids())
            study_ids = [] if study_ids_pkl is None else pickle.loads(study_ids_pkl)
            study_ids.remove(study_id)
            self._db.put(self._key_study_ids(), pickle.dumps(study_ids))

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        self._check_study_id(study_id)

        with self._lock:
            summary = pickle.loads(self._db.get(self._key_study_summary(study_id)))
            summary.user_attrs[key] = value
            self._db.put(self._key_study_summary(study_id), pickle.dumps(summary))

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        self._check_study_id(study_id)

        with self._lock:
            summary = pickle.loads(self._db.get(self._key_study_summary(study_id)))
            summary.system_attrs[key] = value
            self._db.put(self._key_study_summary(study_id), pickle.dumps(summary))

    def set_study_directions(self, study_id: int, directions: Sequence[StudyDirection]) -> None:
        self._check_study_id(study_id)

        with self._lock:
            summary = pickle.loads(self._db.get(self._key_study_summary(study_id)))
            current_directions = summary.directions
            if (
                current_directions[0] != StudyDirection.NOT_SET
                and current_directions != directions
            ):
                raise ValueError(
                    f"Cannot overwrite study direction from {current_directions} to {directions}."
                )
            summary._directions = directions
            self._db.put(self._key_study_summary(study_id), pickle.dumps(summary))

    def get_study_id_from_name(self, study_name: str) -> int:

        with self._lock:
            study_id_pkl = self._db.get(self._key_study_name(study_name))
            if study_id_pkl is None:
                raise KeyError(f"No such study {study_name}.")
            return pickle.loads(study_id_pkl)

    def get_study_id_from_trial_id(self, trial_id: int) -> int:

        with self._lock:
            study_id_pkl = self._db.get(f"study_id:trial_id:{trial_id:010d}".encode())
            if study_id_pkl is None:
                raise KeyError(f"No such trial {trial_id}.")
            return pickle.loads(study_id_pkl)

    def get_study_name_from_id(self, study_id: int) -> str:
        self._check_study_id(study_id)

        with self._lock:
            summary: StudySummary = pickle.loads(self._db.get(self._key_study_summary(study_id)))
            return summary.study_name

    def get_study_directions(self, study_id: int) -> List[StudyDirection]:
        self._check_study_id(study_id)

        with self._lock:
            summary: StudySummary = pickle.loads(self._db.get(self._key_study_summary(study_id)))
            return summary.directions

    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:
        self._check_study_id(study_id)

        with self._lock:
            summary: StudySummary = pickle.loads(self._db.get(self._key_study_summary(study_id)))
            return summary.user_attrs

    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:
        self._check_study_id(study_id)

        with self._lock:
            summary: StudySummary = pickle.loads(self._db.get(self._key_study_summary(study_id)))
            return summary.system_attrs

    def get_all_study_summaries(self, include_best_trial: bool) -> List[StudySummary]:
        with self._lock:
            study_ids_obj = self._db.get(self._key_study_ids())
            assert study_ids_obj is not None
            study_ids = pickle.loads(study_ids_obj)
            study_summaries: List[StudySummary] = []
            for study_id in study_ids:
                summary: StudySummary = pickle.loads(self._db.get(self._key_study_summary(study_id)))
                summary.n_trials = len(pickle.loads(self._db.get(self._key_trial_ids(study_id))))
                if include_best_trial and len(summary.directions) == 1:
                    trials = self.get_all_trials(study_id, deepcopy=False, states=(TrialState.COMPLETE, ))
                    if summary.direction == StudyDirection.MINIMIZE:
                        best_trial = min(trials, key=lambda t: t.value)
                    else:
                        best_trial = max(trials, key=lambda t: t.value)
                    summary.best_trial = best_trial
                study_summaries.append(summary)
            return study_summaries

    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:
        self._check_study_id(study_id)

        with self._lock:
            if template_trial is None:
                trial = self._create_running_trial()
            else:
                trial = copy.deepcopy(template_trial)

            trial_counter_pkl = self._db.get(b"trial_counter")
            trial_id = 0 if trial_counter_pkl is None else pickle.loads(trial_counter_pkl)
            self._db.put(b"trial_counter", pickle.dumps(trial_id + 1))
            trial._trial_id = trial_id

            trial_number_counter_pkl = self._db.get(f"trial_number:study_id:{study_id:010d}".encode())
            trial_number = 0 if trial_number_counter_pkl is None else pickle.loads(trial_number_counter_pkl)
            self._db.put(f"trial_number:study_id:{study_id:010d}".encode(), pickle.dumps(trial_number + 1))
            trial.number = trial_number

            self._db.put(self._key_trial(trial_id), pickle.dumps(trial))
            self._db.put(f"study_id:trial_id:{trial_id:010d}".encode(), pickle.dumps(study_id))
            self._db.put(f"trial_id:study_id:{study_id:010d}:trial_number:{trial_number}".encode(), pickle.dumps(trial_id))

            trial_ids = pickle.loads(self._db.get(self._key_trial_ids(study_id)))
            trial_ids.append(trial_id)
            self._db.put(self._key_trial_ids(study_id), pickle.dumps(trial_ids))

        return trial._trial_id

    @staticmethod
    def _create_running_trial() -> FrozenTrial:

        return FrozenTrial(
            trial_id=-1,  # dummy value.
            number=-1,  # dummy value.
            state=TrialState.RUNNING,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            value=None,
            intermediate_values={},
            datetime_start=datetime.now(),
            datetime_complete=None,
        )

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:
        return pickle.loads(self._db.get(f"trial_id:study_id:{study_id:010d}:trial_number:{trial_number}".encode()))

    def set_trial_state(self, trial_id: int, state: TrialState) -> bool:
        self._check_trial_id(trial_id)

        with self._lock:
            trial: FrozenTrial = pickle.loads(self._db.get(self._key_trial(trial_id)))
            self.check_trial_is_updatable(trial_id, trial.state)

            if state == TrialState.RUNNING and trial.state != TrialState.WAITING:
                return False

            trial.state = state

            if state == TrialState.RUNNING:
                trial.datetime_start = datetime.now()

            if state.is_finished():
                trial.datetime_complete = datetime.now()

            self._db.put(self._key_trial(trial_id), pickle.dumps(trial))

            return True

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        self._check_trial_id(trial_id)

        with self._lock:
            trial: FrozenTrial = pickle.loads(self._db.get(self._key_trial(trial_id)))
            self.check_trial_is_updatable(trial_id, trial.state)

            study_id = self.get_study_id_from_trial_id(trial_id)
            param_distribution = pickle.loads(self._db.get(self._key_study_param_distribution(study_id)))
            if param_name in param_distribution:
                distributions.check_distribution_compatibility(
                    param_distribution[param_name], distribution
                )

            param_distribution[param_name] = distribution
            trial.params[param_name] = param_value_internal
            trial.distributions[param_name] = distribution
            self._db.put(self._key_study_param_distribution(study_id), pickle.dumps(param_distribution))
            self._db.put(self._key_trial(trial_id), pickle.dumps(trial))

    def set_trial_values(self, trial_id: int, values: Sequence[float]) -> None:
        self._check_trial_id(trial_id)

        with self._lock:
            trial: FrozenTrial = pickle.loads(self._db.get(self._key_trial(trial_id)))
            self.check_trial_is_updatable(trial_id, trial.state)
            trial.values = values
            self._db.put(self._key_trial(trial_id), pickle.dumps(trial))

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        self._check_trial_id(trial_id)

        with self._lock:
            trial: FrozenTrial = pickle.loads(self._db.get(self._key_trial(trial_id)))
            self.check_trial_is_updatable(trial_id, trial.state)
            trial.intermediate_values[step] = intermediate_value
            self._db.put(self._key_trial(trial_id), pickle.dumps(trial))

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        self._check_trial_id(trial_id)

        with self._lock:
            trial: FrozenTrial = pickle.loads(self._db.get(self._key_trial(trial_id)))
            self.check_trial_is_updatable(trial_id, trial.state)
            trial.user_attrs[key] = value
            self._db.put(self._key_trial(trial_id), pickle.dumps(trial))

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        self._check_trial_id(trial_id)

        with self._lock:
            trial: FrozenTrial = pickle.loads(self._db.get(self._key_trial(trial_id)))
            self.check_trial_is_updatable(trial_id, trial.state)
            trial.user_attrs[key] = value
            self._db.put(self._key_trial(trial_id), pickle.dumps(trial))

    def get_trial(self, trial_id: int) -> FrozenTrial:
        self._check_trial_id(trial_id)

        with self._lock:
            trial: FrozenTrial = pickle.loads(self._db.get(self._key_trial(trial_id)))
            return trial

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Optional[Container[TrialState]] = None,
    ) -> List[FrozenTrial]:
        self._check_study_id(study_id)
        trial_ids = pickle.loads(self._db.get(self._key_trial_ids(study_id)))
        trials = []
        for trial_id in trial_ids:
            trial = self.get_trial(trial_id)
            if states is not None and trial.state not in states:
                continue
            trials.append(trial)
        return trials

    def read_trials_from_remote_storage(self, study_id: int) -> None:
        pass

