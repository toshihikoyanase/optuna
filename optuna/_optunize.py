import functools
import inspect
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

from optuna.distributions import BaseDistribution
from optuna.trial import Trial


def _extract_args(func: Callable[[Any], Any]) -> Tuple[List[str], Dict[str, Any]]:
    """Extract arguments of the function.

    It returns required a list of required argument names and a dictionary of the
    default values of the arguments.
    """
    s = inspect.signature(func)
    required_arg_names = []
    default_values = {}
    for param in s.parameters.values():
        if param.default == param.empty:
            required_arg_names.append(param.name)
        else:
            default_values[param.name] = param.default
    return required_arg_names, default_values


def optunize(search_space: Dict[str, BaseDistribution]) -> Any:
    """Decorate a function to convert Optuna's objective function.

    It expects the function that takes parameters as the arguments and returns a float value as
    an objective value.

    Example:

        .. testcode::

            import optuna
            from optuna.distributions import UniformDistribution

            @optunize({"x": UniformDistribution(0, 10)})
            def func(x, y=5):
                return (x - 2) ** 2 + (y - 8) ** 2

            study = optuna.create_study()
            study.optimize(func, n_trials=10)

    Args:
        search_space: A dictionary to define search space. The key is the parameter name and
        the value is the corresponding distribution.

        .. note::
            Note that ``search_space`` assumes that the parameter names are the same as the
            corresponding argument names of the given function.
    """

    def _optunize_wrapper(func: Callable[[Any], float]) -> Callable[[Trial], float]:
        @functools.wraps(func)
        def new_func(trial: Trial) -> float:
            _required, _args = _extract_args(func)
            for name, distribution in search_space.items():
                _args[name] = trial._suggest(name, distribution)
            if not (set(_required) <= set(_args.keys())):
                not_found = set(_required) - set(_args.keys())
                raise ValueError(
                    "Parameter {} not found. Please add them to the search space.".format(
                        not_found
                    )
                )
            return func(*[], **_args)

        return new_func

    return _optunize_wrapper
