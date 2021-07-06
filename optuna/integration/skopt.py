import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import warnings

import numpy as np

import optuna
from optuna import distributions
from optuna import samplers
from optuna._imports import try_import
from optuna._study_direction import StudyDirection
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


with try_import() as _imports:
    import skopt
    from skopt.learning import GaussianProcessRegressor
    from skopt.learning.gaussian_process.kernels import ConstantKernel
    from skopt.learning.gaussian_process.kernels import Matern
    from skopt.space import space
    from skopt.space import Space
    from skopt.utils import normalize_dimensions


class SkoptSampler(BaseSampler):
    """Sampler using Scikit-Optimize as the backend.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.integration.SkoptSampler`.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                y = trial.suggest_int("y", 0, 10)
                return x ** 2 + y


            sampler = optuna.integration.SkoptSampler()
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)

    Args:
        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.integration.SkoptSampler` is determined by
            :func:`~optuna.samplers.intersection_search_space()`.

            If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler` is used
            as the default.

            .. seealso::
                :class:`optuna.samplers` module provides built-in independent samplers
                such as :class:`~optuna.samplers.RandomSampler` and
                :class:`~optuna.samplers.TPESampler`.

        warn_independent_sampling:
            If this is :obj:`True`, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.

            Note that the parameters of the first trial in a study are always sampled
            via an independent sampler, so no warning messages are emitted in this case.

        skopt_kwargs:
            Keyword arguments passed to the constructor of
            `skopt.Optimizer <https://scikit-optimize.github.io/#skopt.Optimizer>`_
            class.

            Note that ``dimensions`` argument in ``skopt_kwargs`` will be ignored
            because it is added by :class:`~optuna.integration.SkoptSampler` automatically.

        n_startup_trials:
            The independent sampling is used until the given number of trials finish in the
            same study.

        consider_pruned_trials:
            If this is :obj:`True`, the PRUNED trials are considered for sampling.

            .. note::
                Added in v2.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.0.0.

            .. note::
                As the number of trials :math:`n` increases, each sampling takes longer and longer
                on a scale of :math:`O(n^3)`. And, if this is :obj:`True`, the number of trials
                will increase. So, it is suggested to set this flag :obj:`False` when each
                evaluation of the objective function is relatively faster than each sampling. On
                the other hand, it is suggested to set this flag :obj:`True` when each evaluation
                of the objective function is relatively slower than each sampling.
    """

    def __init__(
        self,
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        skopt_kwargs: Optional[Dict[str, Any]] = None,
        n_startup_trials: int = 1,
        *,
        consider_pruned_trials: bool = False,
        experimental_categorical: bool = False,
    ) -> None:

        _imports.check()

        self._skopt_kwargs = skopt_kwargs or {}
        if "dimensions" in self._skopt_kwargs:
            del self._skopt_kwargs["dimensions"]

        self._independent_sampler = independent_sampler or samplers.RandomSampler()
        self._warn_independent_sampling = warn_independent_sampling
        self._n_startup_trials = n_startup_trials
        self._search_space = samplers.IntersectionSearchSpace()
        self._consider_pruned_trials = consider_pruned_trials
        self._experimental_categorical = experimental_categorical

        if self._consider_pruned_trials:
            warnings.warn(
                "`consider_pruned_trials` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

    def reseed_rng(self) -> None:

        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, distributions.BaseDistribution]:

        search_space = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                if not isinstance(distribution, distributions.CategoricalDistribution):
                    # `skopt` cannot handle non-categorical distributions that contain just
                    # a single value, so we skip this distribution.
                    #
                    # Note that `Trial` takes care of this distribution during suggestion.
                    continue

            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, distributions.BaseDistribution],
    ) -> Dict[str, Any]:

        self._raise_error_if_multi_objective(study)

        if len(search_space) == 0:
            return {}

        complete_trials = self._get_trials(study)
        if len(complete_trials) < self._n_startup_trials:
            return {}

        optimizer = _Optimizer(search_space, self._skopt_kwargs, self._experimental_categorical)
        optimizer.tell(study, complete_trials)
        return optimizer.ask()

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:

        self._raise_error_if_multi_objective(study)

        if self._warn_independent_sampling:
            complete_trials = self._get_trials(study)
            if len(complete_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:

        logger = optuna.logging.get_logger(__name__)
        logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "by using `{}` instead of `SkoptSampler` "
            "(optimization performance may be degraded). "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `SkoptSampler`, "
            "if this independent sampling is intended behavior.".format(
                param_name, trial.number, self._independent_sampler.__class__.__name__
            )
        )

    def _get_trials(self, study: Study) -> List[FrozenTrial]:
        complete_trials = []
        for t in study.get_trials(deepcopy=False):
            if t.state == TrialState.COMPLETE:
                complete_trials.append(t)
            elif (
                t.state == TrialState.PRUNED
                and len(t.intermediate_values) > 0
                and self._consider_pruned_trials
            ):
                _, value = max(t.intermediate_values.items())
                if value is None:
                    continue
                # We rewrite the value of the trial `t` for sampling, so we need a deepcopy.
                copied_t = copy.deepcopy(t)
                copied_t.value = value
                complete_trials.append(copied_t)
        return complete_trials

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:

        self._independent_sampler.after_trial(study, trial, state, values)


class _Optimizer(object):
    def __init__(
        self,
        search_space: Dict[str, distributions.BaseDistribution],
        skopt_kwargs: Dict[str, Any],
        experimental_categorical: bool = False,
        force_one_hot: bool = False,
    ) -> None:

        self._search_space = search_space

        dimensions = []
        for name, distribution in sorted(self._search_space.items()):
            if isinstance(distribution, distributions.UniformDistribution):
                # Convert the upper bound from exclusive (optuna) to inclusive (skopt).
                high = np.nextafter(distribution.high, float("-inf"))
                dimension = space.Real(distribution.low, high)
            elif isinstance(distribution, distributions.LogUniformDistribution):
                # Convert the upper bound from exclusive (optuna) to inclusive (skopt).
                high = np.nextafter(distribution.high, float("-inf"))
                dimension = space.Real(distribution.low, high, prior="log-uniform")
            elif isinstance(distribution, distributions.IntUniformDistribution):
                count = (distribution.high - distribution.low) // distribution.step
                dimension = space.Integer(0, count)
            elif isinstance(distribution, distributions.IntLogUniformDistribution):
                low = distribution.low - 0.5
                high = distribution.high + 0.5
                dimension = space.Real(low, high, prior="log-uniform")
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                count = int((distribution.high - distribution.low) // distribution.q)
                dimension = space.Integer(0, count)
            elif isinstance(distribution, distributions.CategoricalDistribution):
                dimension = space.Categorical(distribution.choices)
            else:
                raise NotImplementedError(
                    "The distribution {} is not implemented.".format(distribution)
                )

            dimensions.append(dimension)

        if experimental_categorical:

            # copy from sklearn's definition
            _space = Space(dimensions)
            _space = Space(normalize_dimensions(_space.dimensions))
            n_dims = _space.transformed_n_dims

            cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
            if not force_one_hot:
                other_kernel = _WrappedMatern(
                    dimensions=_space,
                    length_scale=np.ones(n_dims),
                    length_scale_bounds=[(0.01, 100)] * n_dims,
                    nu=2.5,
                )
            else:
                other_kernel = Matern(
                    length_scale=np.ones(n_dims),
                    length_scale_bounds=[(0.01, 100)] * n_dims, nu=2.5)
            base_estimator = GaussianProcessRegressor(
                kernel=cov_amplitude * other_kernel,
                normalize_y=True,
                noise="gaussian",
                n_restarts_optimizer=2,
            )
            self._optimizer = skopt.Optimizer(
                dimensions, base_estimator=base_estimator, acq_optimizer="sampling", **skopt_kwargs
            )
        else:
            # original skopt implementation
            self._optimizer = skopt.Optimizer(dimensions, **skopt_kwargs)

    def tell(self, study: Study, complete_trials: List[FrozenTrial]) -> None:

        xs = []
        ys = []

        for trial in complete_trials:
            if not self._is_compatible(trial):
                continue

            x, y = self._complete_trial_to_skopt_observation(study, trial)
            xs.append(x)
            ys.append(y)

        self._optimizer.tell(xs, ys)

    def ask(self) -> Dict[str, Any]:

        params = {}
        param_values = self._optimizer.ask()
        for (name, distribution), value in zip(sorted(self._search_space.items()), param_values):
            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                value = value * distribution.q + distribution.low
            if isinstance(distribution, distributions.IntUniformDistribution):
                value = value * distribution.step + distribution.low
            if isinstance(distribution, distributions.IntLogUniformDistribution):
                value = int(np.round(value))
                value = min(max(value, distribution.low), distribution.high)

            params[name] = value

        return params

    def _is_compatible(self, trial: FrozenTrial) -> bool:

        # Thanks to `intersection_search_space()` function, in sequential optimization,
        # the parameters of complete trials are always compatible with the search space.
        #
        # However, in distributed optimization, incompatible trials may complete on a worker
        # just after an intersection search space is calculated on another worker.

        for name, distribution in self._search_space.items():
            if name not in trial.params:
                return False

            distributions.check_distribution_compatibility(distribution, trial.distributions[name])
            param_value = trial.params[name]
            param_internal_value = distribution.to_internal_repr(param_value)
            if not distribution._contains(param_internal_value):
                return False

        return True

    def _complete_trial_to_skopt_observation(
        self, study: Study, trial: FrozenTrial
    ) -> Tuple[List[Any], float]:

        param_values = []
        for name, distribution in sorted(self._search_space.items()):
            param_value = trial.params[name]

            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                param_value = (param_value - distribution.low) // distribution.q
            if isinstance(distribution, distributions.IntUniformDistribution):
                param_value = (param_value - distribution.low) // distribution.step

            param_values.append(param_value)

        value = trial.value
        assert value is not None

        if study.direction == StudyDirection.MAXIMIZE:
            value = -value

        return param_values, value


class _WrappedMatern(Matern):
    def __init__(
        self, dimensions: skopt.Space, length_scale=1.0, length_scale_bounds=(0.01, 100), nu=2.5
    ):
        self.dimensions = dimensions
        super(_WrappedMatern, self).__init__(length_scale, length_scale_bounds, nu)

    def _transform(self, X):
        logger = optuna.logging.get_logger(__name__)

        transformed_X = []
        num_samples = len(X)
        logger.info("before", X.shape)
        for i, skopt_space in enumerate(self.dimensions):
            if isinstance(skopt_space, space.Categorical):
                # todo(nzw): implement `if isinstance(skopt_space, space.Integer)`
                n_choice = len(skopt_space.categories)
                one_hot_dims = []
                for x in X[:, i]:
                    # convert normalized domain to integer
                    one_hot_dims.append(
                        [np.where(np.arange(n_choice) / (n_choice - 1) == x)[0][0]]
                    )

                one_hot = np.zeros((num_samples, n_choice))
                np.put_along_axis(one_hot, np.array(one_hot_dims), 1.0, axis=1)

                transformed_X.append(one_hot)
            else:
                transformed_X.append(X[:, i])

        transformed_X = np.hstack(transformed_X)
        logger.info("after", transformed_X.shape)

        return transformed_X
