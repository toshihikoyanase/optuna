from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
import warnings

import numpy

from optuna import logging
from optuna._experimental import experimental
from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import IntersectionSearchSpace
from optuna.samplers import RandomSampler
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

with try_import() as _imports:
    from botorch.acquisition.monte_carlo import qExpectedImprovement
    from botorch.fit import fit_gpytorch_model
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from botorch.sampling.samplers import SobolQMCNormalSampler
    from botorch.models.transforms.input import InputTransform
    from gpytorch.kernels.scale_kernel import ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import torch

_logger = logging.get_logger(__name__)


class T(InputTransform):
    def __init__(self, max_indices_for_categorical, dim):
        self.transform_on_eval = True
        self.transform_on_train = True
        self.transform_on_preprocess = True
        self._max_indices_for_categorical = max_indices_for_categorical
        self._dim = dim

    def transform(self, X) -> torch.Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """

        if len(X.shape) == 2:
            if X.shape[-1] != self._dim:
                return X

            x = numpy.rint(X.numpy()).astype(int)
            transformed_sample = []
            for sample in X:
                transformed_elems = []
                for elem, max_index in zip(sample, self._max_indices_for_categorical):
                    one_hot = numpy.zeros(max_index + 1)
                    one_hot[int(elem)] = 1.
                    transformed_elems.append(one_hot)
                transformed_sample.append(numpy.concatenate(transformed_elems))
            y = torch.DoubleTensor(numpy.array(transformed_sample))

            return y
        elif len(X.shape) == 3:
            transformed_x = []
            if X.shape[-1] != self._dim:
                return X
            x = numpy.rint(X.numpy()).astype(int)
            for samples in x:
                transformed_sample = []
                for sample in samples:
                    transformed_elems = []
                    for elem, max_index in zip(sample, self._max_indices_for_categorical):
                        one_hot = numpy.zeros(max_index + 1)
                        one_hot[elem] = 1.
                        transformed_elems.append(one_hot)
                    transformed_sample.append(numpy.concatenate(transformed_elems))
                transformed_x.append(transformed_sample)
            y = torch.DoubleTensor(numpy.array(transformed_x))

            return y
        else:
            raise ValueError(f"Unsupported tensor shape: {X.shape}.")

    def untransform(self, X) -> torch.Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """


        if len(X.shape) == 2:
            untransformed_sample = []
            for sample in X:
                untransformed_elems = []
                start = 0
                for max_index in self._max_indices_for_categorical:
                    end = start + max_index + 1
                    idx = numpy.argmax(sample[start:end])
                    untransformed_elems.append(idx)
                untransformed_sample.append(untransformed_elems)
            y = torch.DoubleTensor(numpy.array(untransformed_sample))
            return y
        elif len(X.shape) == 3:
            untransformed_x = []
            for samples in X:
                untransformed_sample = []
                for sample in samples:
                    untransformed_elems = []
                    start = 0
                    for max_index in self._max_indices_for_categorical:
                        end = start + max_index + 1
                        idx = numpy.argmax(sample[start:end])
                        untransformed_elems.append(idx)
                    untransformed_sample.append(untransformed_elems)
                untransformed_x.append(untransformed_sample)
            y = torch.DoubleTensor(numpy.array(untransformed_x))

            return y
        else:
            raise ValueError(f"Unsupported tensor shape: {X.shape}.")


    def __call__(self, X):
        return self.transform(X)

    def to(self, X):
        pass

    def transform_bounds(self):
        dim = sum(self._max_indices_for_categorical) + len(self._max_indices_for_categorical)
        return torch.DoubleTensor([[0 for _ in range(dim)], [1 for _ in range(dim)]])


@experimental("2.4.0")
def qei_candidates_func(
        train_x: "torch.Tensor",
        train_obj: "torch.Tensor",
        train_con: Optional["torch.Tensor"],
        bounds: "torch.Tensor",
        max_indices_for_categorical: List[int],
        # one_hot_ranges: List[Tuple[int, int]],
        # one_hot_categorical: bool
) -> "torch.Tensor":
    """Quasi MC-based batch Expected Improvement (qEI).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with single-objective optimization.

    Args:
        train_x:
            Previous parameter configurations. A ``torch.Tensor`` of shape
            ``(n_trials, n_params)``. ``n_trials`` is the number of already observed trials
            and ``n_params`` is the number of parameters. ``n_params`` may be larger than the
            actual number of parameters if categorical parameters are included in the search
            space, since these parameters are one-hot encoded.
            Values are not normalized.
        train_obj:
            Previously observed objectives. A ``torch.Tensor`` of shape
            ``(n_trials, n_objectives)``. ``n_trials`` is identical to that of ``train_x``.
            ``n_objectives`` is the number of objectives. Observations are not normalized.
        train_con:
            Objective constraints. A ``torch.Tensor`` of shape ``(n_trials, n_constraints)``.
            ``n_trials`` is identical to that of ``train_x``. ``n_constraints`` is the number of
            constraints. A constraint is violated if strictly larger than 0. If no constraints are
            involved in the optimization, this argument will be :obj:`None`.
        bounds:
            Search space bounds. A ``torch.Tensor`` of shape ``(n_params, 2)``. ``n_params`` is
            identical to that of ``train_x``. The first and the second column correspond to the
            lower and upper bounds for each parameter respectively.

    Returns:
        Next set of candidates. Usually the return value of BoTorch's ``optimize_acqf``.

    """

    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qEI.")
    if train_con is not None:
        pass
    else:
        train_y = train_obj

        best_f = train_obj.max()

        objective = None  # Using the default identity objective.

    _input_transform = T(
        max_indices_for_categorical, dim=len(max_indices_for_categorical)
    )
    model = SingleTaskGP(
        train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)),
        input_transform=_input_transform
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    acqf = qExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(num_samples=256),
        objective=objective,
    )

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=_input_transform.transform_bounds(),
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    candidates = _input_transform.untransform(candidates)

    candidates = candidates.detach()

    return candidates


def _get_categorical_domains(search_space: Dict[str, BaseDistribution]) -> List[int]:
    return [len(d.choices) - 1 for d in search_space.values()]


def _categorical_param_2_index(params, search_space: Dict[str, BaseDistribution]):
    trans_params = numpy.zeros(len(search_space), dtype=numpy.float64)
    for i, (name, distribution) in enumerate(search_space.items()):
        assert name in params, "Parameter configuration must contain all distributions."
        param = params[name]

        if isinstance(distribution, CategoricalDistribution):
            choice_idx = distribution.to_internal_repr(param)
            trans_params[i] = choice_idx
        else:
            ValueError
    return trans_params


# TODO(hvy): Allow utilizing GPUs via some parameter, not having to rewrite the callback
# functions.
@experimental("2.4.0")
class BoTorchCategoricalSampler(BaseSampler):
    """A sampler that uses BoTorch, a Bayesian optimization library built on top of PyTorch.

    This sampler allows using BoTorch's optimization algorithms from Optuna to suggest parameter
    configurations. Parameters are transformed to continuous space and passed to BoTorch, and then
    transformed back to Optuna's representations. Categorical parameters are one-hot encoded.

    .. seealso::
        See an `example <https://github.com/optuna/optuna/blob/master/examples/
        multi_objective/botorch_simple.py>`_ how to use the sampler.

    .. seealso::
        See the `BoTorch <https://botorch.org/>`_ homepage for details and for how to implement
        your own ``candidates_func``.

    .. note::
        An instance of this sampler *should be not used with different studies* when used with
        constraints. Instead, a new instance should be created for each new study. The reason for
        this is that the sampler is stateful keeping all the computed constraints.

    Args:
        candidates_func:
            An optional function that suggests the next candidates. It must take the training
            data, the objectives, the constraints, the search space bounds and return the next
            candidates. The arguments are of type ``torch.Tensor``. The return value must be a
            ``torch.Tensor``. However, if ``constraints_func`` is omitted, constraints will be
            :obj:`None`. For any constraints that failed to compute, the tensor will contain
            NaN.

            If omitted, is determined automatically based on the number of objectives. If the
            number of objectives is one, Quasi MC-based batch Expected Improvement (qEI) is used.
            If the number of objectives is larger than one but smaller than four, Quasi MC-based
            batch Expected Hypervolume Improvement (qEHVI) is used. Otherwise, for larger number
            of objectives, the faster Quasi MC-based extended ParEGO (qParEGO) is used.

            The function should assume *maximization* of the objective.

            .. seealso::
                See :func:`optuna.integration.botorch.qei_candidates_func` for an example.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.

            If omitted, no constraints will be passed to ``candidates_func`` nor taken into
            account during suggestion if ``candidates_func`` is omitted.
        n_startup_trials:
            Number of initial trials, that is the number of trials to resort to independent
            sampling.
        independent_sampler:
            An independent sampler to use for the initial trials and for parameters that are
            conditional.
    """

    def __init__(
            self,
            *,
            candidates_func: Callable[
                [
                    "torch.Tensor",
                    "torch.Tensor",
                    Optional["torch.Tensor"],
                    "torch.Tensor",
                ],
                "torch.Tensor",
            ] = None,
            constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
            n_startup_trials: int = 10,
            independent_sampler: Optional[BaseSampler] = None,
    ):
        _imports.check()

        self._candidates_func = candidates_func
        self._constraints_func = constraints_func
        self._independent_sampler = independent_sampler or RandomSampler()
        self._n_startup_trials = n_startup_trials

        self._study_id: Optional[int] = None
        self._search_space = IntersectionSearchSpace()

    def infer_relative_search_space(
            self,
            study: Study,
            trial: FrozenTrial,
    ) -> Dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            # Note that the check below is meaningless when `InMemoryStorage` is used
            # because `InMemoryStorage.create_new_study` always returns the same study ID.
            raise RuntimeError("BoTorchSampler cannot handle multiple studies.")

        return self._search_space.calculate(study, ordered_dict=True)  # type: ignore

    def sample_relative(
            self,
            study: Study,
            trial: FrozenTrial,
            search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        assert isinstance(search_space, OrderedDict)

        if len(search_space) == 0:
            return {}

        trials = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]

        n_trials = len(trials)
        if n_trials < self._n_startup_trials:
            return {}

        # get dimensionality
        n_params = len(search_space)

        # TODO(nzw): remove max_indices_for_categorical by using bounds' values for simplicity.
        max_indices_for_categorical = _get_categorical_domains(search_space)

        n_objectives = len(study.directions)
        values: Union[numpy.ndarray, torch.Tensor] = numpy.empty(
            (n_trials, n_objectives), dtype=numpy.float64
        )
        params: Union[numpy.ndarray, torch.Tensor]
        con: Optional[Union[numpy.ndarray, torch.Tensor]] = None
        bounds: Union[numpy.ndarray, torch.Tensor] = numpy.array([[0., float(m)] for m in max_indices_for_categorical])
        params = numpy.empty((n_trials, n_params), dtype=numpy.float64)

        for trial_idx, trial in enumerate(trials):
            params[trial_idx] = _categorical_param_2_index(trial.params, search_space)

            assert len(study.directions) == len(trial.values)

            for obj_idx, (direction, value) in enumerate(zip(study.directions, trial.values)):
                assert value is not None
                if direction == StudyDirection.MINIMIZE:  # BoTorch always assumes maximization.
                    value *= -1
                values[trial_idx, obj_idx] = value

        params = torch.from_numpy(params.astype(numpy.float64))
        values = torch.from_numpy(values.astype(numpy.float64))
        bounds = torch.from_numpy(bounds)

        if con is not None:
            if con.dim() == 1:
                con.unsqueeze_(-1)

        bounds.transpose_(0, 1)
        candidates = qei_candidates_func(params, values, con, bounds, max_indices_for_categorical)

        if not isinstance(candidates, torch.Tensor):
            raise TypeError("Candidates must be a torch.Tensor.")
        if candidates.dim() == 2:
            if candidates.size(0) != 1:
                raise ValueError(
                    "Candidates batch optimization is not supported and the first dimension must "
                    "have size 1 if candidates is a two-dimensional tensor. Actual: "
                    f"{candidates.size()}."
                )
            # Batch size is one. Get rid of the batch dimension.
            candidates = candidates.squeeze(0)
        if candidates.dim() != 1:
            raise ValueError("Candidates must be one or two-dimensional.")
        if candidates.size(0) != bounds.size(1):
            raise ValueError(
                "Candidates size must match with the given bounds. Actual candidates: "
                f"{candidates.size(0)}, bounds: {bounds.size(1)}."
            )

        # untransform
        candidates = numpy.rint(candidates.numpy()).astype(int)
        params = {}

        for i, (name, distribution) in enumerate(search_space.items()):
            params[name] = distribution.to_external_repr(candidates[i])

        return params

    def sample_independent(
            self,
            study: Study,
            trial: FrozenTrial,
            param_name: str,
            param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()

    def after_trial(
            self,
            study: Study,
            trial: FrozenTrial,
            state: TrialState,
            values: Optional[Sequence[float]],
    ) -> None:
        if self._constraints_func is not None:
            constraints = None

            try:
                con = self._constraints_func(trial)
                if not isinstance(con, (tuple, list)):
                    warnings.warn(
                        f"Constraints should be a sequence of floats but got {type(con).__name__}."
                    )
                constraints = tuple(con)
            except Exception:
                raise
            finally:
                assert constraints is None or isinstance(constraints, tuple)

                study._storage.set_trial_system_attr(
                    trial._trial_id,
                    "botorch:constraints",
                    constraints,
                )
        self._independent_sampler.after_trial(study, trial, state, values)
