import abc
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


_NUMERICAL_DISTRIBUTIONS = (
    UniformDistribution,
    LogUniformDistribution,
    DiscreteUniformDistribution,
    IntUniformDistribution,
    IntLogUniformDistribution,
)


class BaseCrossover(object, metaclass=abc.ABCMeta):
    def create_child(
        self,
        study: Study,
        parent_population: Sequence[FrozenTrial],
        search_space: Dict[str, BaseDistribution],
        rng: np.random.RandomState,
        swapping_prob: float,
        dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
    ) -> Dict[str, Any]:

        while True:  # Repeat while parameters lie outside search space boundaries.
            parents = self._selection(study, parent_population, rng, dominates)
            child = {}
            transes = []
            distributions = []
            param_names = []
            parents_not_categorical_params: List[List[np.float64]] = [
                [] for _ in range(len(parents))
            ]
            for param_name in search_space.keys():
                param_distribution = search_space[param_name]
                parents_param = [p.params[param_name] for p in parents]

                # categorical data operates on uniform crossover
                if isinstance(search_space[param_name], CategoricalDistribution):
                    param = _swap(parents_param[0], parents_param[-1], rng.rand(), swapping_prob)
                    child[param_name] = param
                    continue

                trans = _SearchSpaceTransform({param_name: param_distribution})
                transes.append(trans)
                distributions.append(param_distribution)
                param_names.append(param_name)
                for parent_index, trial in enumerate(parents):
                    param = trans.transform({param_name: trial.params[param_name]})[0]
                    parents_not_categorical_params[parent_index].append(param)

            xs = np.array(parents_not_categorical_params)

            params_array = self.numerical_crossover(xs, rng, distributions, study)

            _params = [
                trans.untransform(np.array([param])) for trans, param in zip(transes, params_array)
            ]
            child = {}
            for param in _params:
                for param_name in param.keys():
                    child[param_name] = param[param_name]

            if _is_constrained(child, search_space):
                break

        return child

    def _selection(
        self,
        study: Study,
        parent_population: Sequence[FrozenTrial],
        rng: np.random.RandomState,
        dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
    ) -> List[FrozenTrial]:
        n_select = self.get_n_select()
        parents = []
        for _ in range(n_select):
            parent = _select_parent(
                study, [t for t in parent_population if t not in parents], rng, dominates
            )
            parents.append(parent)
        return parents

    @abc.abstractmethod
    def numerical_crossover(
        self,
        xs: np.ndarray,
        rng: np.random.RandomState,
        distributions: Sequence[BaseDistribution],
        study: Study,
    ) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def get_n_select(self) -> int:
        raise NotImplementedError


def _swap(p0_i: Any, p1_i: Any, rand: float, swapping_prob: float) -> Any:
    if rand < swapping_prob:
        return p1_i
    else:
        return p0_i


class UniformCrossover(BaseCrossover):
    def __init__(self, swapping_prob: float) -> None:
        self._swapping_prob = swapping_prob

    def get_n_select(self) -> int:
        return 2

    def numerical_crossover(
        self,
        xs: np.ndarray,
        rng: np.random.RandomState,
        distributions: Sequence[BaseDistribution],
        study: Study,
    ) -> np.ndarray:
        child = []
        x0, x1 = xs[0], xs[1]
        for x0_i, x1_i in zip(x0, x1):
            param = _swap(x0_i, x1_i, rng.rand(), self._swapping_prob)
            child.append(param)
        return np.array(child)


class BlxAlpha(BaseCrossover):
    def __init__(self, alpha: float=0.5) -> None:
        self._alpha = alpha

    def get_n_select(self) -> int:
        return 2

    def numerical_crossover(
        self,
        xs: np.ndarray,
        rng: np.random.RandomState,
        distributions: Sequence[BaseDistribution],
        study: Study,
    ) -> np.ndarray:
        # https://confit.atlas.jp/guide/event-img/jsai2019/4O3-J-7-02/public/pdf?type=in
        # https://www.sciencedirect.com/science/article/abs/pii/B9780080948324500180

        x_min = xs.min(axis=0)
        x_max = xs.max(axis=0)
        diff = self._alpha * (x_max - x_min)
        low = x_min - diff
        high = x_max + diff
        r = rng.uniform(0, 1, size=len(diff))
        param = (high - low) * r + low
        return param


class Sbx(BaseCrossover):
    def __init__(self, eta: Optional[float] = None) -> None:
        self._eta = eta

    def get_n_select(self) -> int:
        return 2

    def numerical_crossover(
        self,
        xs: np.ndarray,
        rng: np.random.RandomState,
        distributions: Sequence[BaseDistribution],
        study: Study,
    ) -> np.ndarray:
        # https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1589-07.pdf
        # https://www.slideshare.net/paskorn/simulated-binary-crossover-presentation

        eta = self._eta
        if eta is None:
            eta = 2 if len(study.directions) == 1 else 20

        _xls = []
        _xus = []
        for distribution in distributions:
            assert isinstance(distribution, _NUMERICAL_DISTRIBUTIONS)
            _xls.append(distribution.low)
            _xus.append(distribution.high)
        xls = np.array(_xls)
        xus = np.array(_xus)

        xs_min = np.min(xs, axis=0)
        xs_max = np.max(xs, axis=0)

        xs_diff = np.clip(xs_max - xs_min, 1e-10, None)
        beta1 = 1 + 2 * (xs_min - xls) / xs_diff  # (6)
        beta2 = 1 + 2 * (xus - xs_max) / xs_diff  # (6)
        alpha1 = 2 - np.power(beta1, -(eta + 1))  # (5)
        alpha2 = 2 - np.power(beta2, -(eta + 1))  # (5)

        us = rng.uniform(0, 1, size=len(xs[0]))

        mask1 = us > 1 / alpha1  # (4)
        betaq1 = np.power(us * alpha1, 1 / (eta + 1))  # (4)
        betaq1[mask1] = np.power((1 / (2 - us * alpha1)), 1 / (eta + 1))[mask1]  # (4)

        mask2 = us > 1 / alpha2  # (4)
        betaq2 = np.power(us * alpha2, 1 / (eta + 1))  # (4)
        betaq2[mask2] = np.power((1 / (2 - us * alpha2)), 1 / (eta + 1))[mask2]  # (4)

        c1 = 0.5 * ((xs_min + xs_max) - betaq1 * xs_diff)  # (7)
        c2 = 0.5 * ((xs_min + xs_max) + betaq2 * xs_diff)  # (7)

        child = []
        for c1_i, c2_i, x1_i, x2_i in zip(c1, c2, xs[0], xs[1]):
            if rng.rand() < 0.5:
                if rng.rand() < 0.5:
                    child.append(c1_i)
                else:
                    child.append(c2_i)
            else:
                if rng.rand() < 0.5:
                    child.append(x1_i)
                else:
                    child.append(x2_i)
        return np.array(child)


class VSbx(BaseCrossover):
    def __init__(self, eta: Optional[float] = None) -> None:
        self._eta = eta

    def get_n_select(self) -> int:
        return 2

    def numerical_crossover(
        self,
        xs: np.ndarray,
        rng: np.random.RandomState,
        distributions: Sequence[BaseDistribution],
        study: Study,
    ) -> np.ndarray:
        # https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1589-07.pdf
        # https://www.slideshare.net/paskorn/simulated-binary-crossover-presentation

        eta = self._eta
        if eta is None:
            eta = 2 if len(study.directions) == 1 else 20

        us = rng.uniform(0, 1, size=len(xs[0]))  # (3.12)
        x0, x1 = xs[0], xs[1]
        beta_1 = np.power(1 / 2 * us, 1 / (eta + 1))  # (3.9)
        beta_2 = np.power(1 / 2 * (1 - us), 1 / (eta + 1))  # (3.11)
        mask = us > 0.5
        c1 = 0.5 * ((1 + beta_1) * x0 + (1 - beta_1) * x1)  # (3.8)
        c1[mask] = 0.5 * ((1 - beta_1) * x0 + (1 + beta_1) * x1)[mask]  # (3.8)
        c2 = 0.5 * ((3 - beta_2) * x0 - (1 - beta_2) * x1)  # (3.10)
        c2[mask] = 0.5 * (-(1 - beta_2) * x0 + (3 - beta_2) * x1)[mask]  # (3.10)

        child = []
        for c1_i, c2_i, x1_i, x2_i in zip(c1, c2, x0, x1):
            if rng.rand() < 0.5:
                if rng.rand() < 0.5:
                    child.append(c1_i)
                else:
                    child.append(c2_i)
            else:
                if rng.rand() < 0.5:
                    child.append(x1_i)
                else:
                    child.append(x2_i)
        return np.array(child)


class Undx(BaseCrossover):
    def get_n_select(self) -> int:
        return 3

    def numerical_crossover(
        self,
        xs: np.ndarray,
        rng: np.random.RandomState,
        distributions: Sequence[BaseDistribution],
        study: Study,
    ) -> np.ndarray:
        # https://www.jstage.jst.go.jp/article/sicetr1965/36/10/36_10_875/_pdf
        sigma_xi = 0.5
        sigma_eta = 0.35 / np.sqrt(len(xs[0]))

        x0, x1, x2 = xs[0], xs[1], xs[2]  # section 2 (1)
        n = len(x0)
        xp = (x0 + x1) / 2  # section 2 (2)
        d = x0 - x1  # section 2 (3)
        D = Undx._distance_from_x_to_psl(x0, x1, x2)  # section 2 (4)
        xi = rng.normal(0, sigma_xi ** 2)
        etas = rng.normal(0, sigma_eta, size=n)
        es = Undx._orthonormal_basis_vector_to_psl(
            x0, x1
        )  # Orthonormal basis vectors of the subspace orthogonal to the psl
        one = xp  # section 2 (5)
        two = xi * d  # section 2 (5)
        three = np.zeros(len(es[0]))  # section 2 (5)
        for i in range(n - 1):
            three += etas[i] * es[i]
        three *= D
        return one + two + three

    @staticmethod
    def _normalized_x1_to_x2(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        v_12 = x2 - x1
        m_12 = np.linalg.norm(v_12, ord=2)
        e_12 = v_12 / np.clip(m_12, 1e-10, None)
        return e_12

    @staticmethod
    def _distance_from_x_to_psl(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
        e_12 = Undx._normalized_x1_to_x2(x1, x2)  # Normalized vector from x1 to x2
        v_13 = x3 - x1  # Vector from x1 to x3
        v_12_3 = v_13 - np.dot(v_13, e_12) * e_12  # Vector orthogonal to v_12 through x3
        m_12_3 = np.linalg.norm(v_12_3, ord=2)  # 2-norm of v_12_3
        return m_12_3

    @staticmethod
    def _orthonormal_basis_vector_to_psl(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        n = len(x1)
        e_12 = Undx._normalized_x1_to_x2(x1, x2)  # Normalized vector from x1 to x2
        basis_matrix = np.identity(n)
        if np.count_nonzero(e_12) != 0:
            v_01 = x1 - np.zeros(len(x1))
            basis_matrix[0] = v_01 - np.dot(v_01, e_12) * e_12  # Vector orthogonal to e_12
        basis_matrix_t = basis_matrix.T
        Q, _ = np.linalg.qr(basis_matrix_t)
        return Q.T


class UndxM(BaseCrossover):
    def get_n_select(self) -> int:
        return 4

    def numerical_crossover(
        self,
        xs: np.ndarray,
        rng: np.random.RandomState,
        distributions: Sequence[BaseDistribution],
        study: Study,
    ) -> np.ndarray:
        # https://www.jstage.jst.go.jp/article/sicetr1965/36/10/36_10_875/_pdf
        _m = 2
        _n = len(xs[0])
        sigma_xi = 1 / np.sqrt(_m)
        sigma_eta = (
            0.35 * np.sqrt(_m + 1) * np.sqrt(3) / np.sqrt(_n - _m) / np.sqrt(_m + 2) / np.sqrt(2)
        )

        x_mp2, xs = xs[-1], xs[:-1]  # section 4.2 (1), (3)
        m = len(xs) - 1
        dim = len(x_mp2)
        p = np.sum(xs, axis=0) / (m + 1)  # section 4.2(2)
        ds = [x - p for x in xs]  # section 4.2 (2)
        n = UndxM._normal(
            rng, ds[:-1]
        )  # Normal to the plane that contains d_i(1,..,m) section 4.2 (4)
        d_mp2 = x_mp2 - p  # section 4.2 (4)
        D = np.dot(d_mp2, n) / np.linalg.norm(n)  # section 4.2 (4)
        es = UndxM._orthonormal_basis_vector_from_ds(
            ds[:-1]
        )  # orthonormal basis of the subspace orthogonal to d_i(1,..,m) section 4.2 (5)

        one = p  # section 4.2 (6)

        ws = rng.normal(0, sigma_xi ** 2, size=m)
        two = np.zeros(dim)  # section 4.2 (6)
        for i in range(m):
            two += ws[i] * ds[i]

        three = np.zeros(dim)  # section 4.2 (6)
        for i in range(dim - m):
            vs = rng.normal(0, sigma_eta ** 2, size=dim)
            three += vs - np.dot(vs, es[i]) * es[i]
        three *= D
        return one + two + three

    @staticmethod
    def _normal(rng: np.random.RandomState, ds: List[np.ndarray]) -> np.ndarray:
        # Create an orthonormal basis by adding one appropriate vector to ds
        # , and extract one vector from the orthonormal basis
        d = rng.normal(0, 1, size=ds[0].shape[0])
        ds.append(d)
        X = np.stack(ds)
        Q, _ = np.linalg.qr(X.T)
        return Q.T[-1]

    @staticmethod
    def _orthonormal_basis_vector_from_ds(ds: List[np.ndarray]) -> np.ndarray:
        X = np.stack(ds)
        Q, _ = np.linalg.qr(X.T)
        return Q.T[-1]


class Spx(BaseCrossover):
    def get_n_select(self) -> int:
        return 3

    def numerical_crossover(
        self,
        xs: np.ndarray,
        rng: np.random.RandomState,
        distributions: Sequence[BaseDistribution],
        study: Study,
    ) -> np.ndarray:
        # https://www.jstage.jst.go.jp/article/tjsai/16/1/16_1_147/_pdf
        epsilon = np.sqrt(len(xs[0]) + 2)

        n = xs.shape[0] - 1
        G = xs.sum(axis=0) / xs.shape[0]  # section 3.1 [2].
        rs = [np.power(rng.uniform(0, 1), 1 / (k + 1)) for k in range(n)]  # equation (5)
        xks = [G + epsilon * (pk - G) for pk in xs]  # equation (3)
        ck = 0  # equation (4)
        for k in range(1, n + 1):
            ck = rs[k - 1] * (xks[k - 1] - xks[k] + ck)

        c = xks[-1] + ck  # equation (7)
        return c


def _select_parent(
    study: Study,
    population: Sequence[FrozenTrial],
    rng: np.random.RandomState,
    dominates: Callable[[FrozenTrial, FrozenTrial, Sequence[StudyDirection]], bool],
) -> FrozenTrial:
    # TODO(ohta): Consider to allow users to specify the number of parent candidates.
    population_size = len(population)
    candidate0 = population[rng.choice(population_size)]
    candidate1 = population[rng.choice(population_size)]

    # TODO(ohta): Consider crowding distance.
    if dominates(candidate0, candidate1, study.directions):
        return candidate0
    else:
        return candidate1


def _is_constrained(params: Dict[str, Any], search_space: Dict[str, BaseDistribution]) -> bool:
    contains = True
    for param_name in params.keys():
        param, param_distribution = params[param_name], search_space[param_name]
        if isinstance(param_distribution, CategoricalDistribution):
            continue
        if not param_distribution._contains(param):
            contains = False
            break
    return contains


def _get_crossover(name: str, swapping_prob: float) -> BaseCrossover:
    if name == "uniform":
        return UniformCrossover(swapping_prob=swapping_prob)
    if name == "blxalpha":
        return BlxAlpha()
    if name == "sbx":
        return Sbx()
    if name == "vsbx":
        return VSbx()
    if name == "undx":
        return Undx()
    if name == "undxm":
        return UndxM()
    if name == "spx":
        return Spx()
    assert False, f"No such crossover: {name}"
