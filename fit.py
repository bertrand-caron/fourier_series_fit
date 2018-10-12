from math import cos, sin
from numpy import vectorize, cos as np_cos, sin as np_sin, vectorize, fft, pi, linspace, rad2deg, radians, random, isclose, inf, seterr, isinf, max as np_max, abs as np_abs, float64 as np_float # type: ignore
seterr(all='raise')
from scipy.integrate import trapz, simps #type: ignore
from scipy.optimize import curve_fit, minimize # type: ignore
from typing import Any, Union, Callable, List, Optional, Tuple, Iterable, Sequence
from functools import reduce
from operator import itemgetter

from fourier_series_fit.types_helpers import vector, Vector
from fourier_series_fit.rmsd import vector_rmsd

def convolution(z1: float, z2: float) -> float:
    return z1 * z2

convolution_v = vectorize(convolution)

def evaluate(model: Callable[[Any], float], points: Sequence[Any]) -> List[float]:
    return [model(point) for point in points]

Integration_Method = Callable[[Vector, Vector], float]

DEFAULT_INTEGRATION_METHOD = trapz

def vector_if_necessary(x: Union[List, Tuple, Vector]) -> Vector:
    if isinstance(x, Vector):
        return x
    else:
        return vector(x)

def fourier_coeff(trigonometric_function: Callable[[float], float], xs: Vector, ys: Vector, m: int, integration_method: Integration_Method = DEFAULT_INTEGRATION_METHOD) -> float:
    coeff = integration_method(
        convolution_v(trigonometric_function(m * xs), ys),
        xs,
    ) / pi
    return coeff

def a0(xs: Vector, ys: Vector, integration_method: Integration_Method = DEFAULT_INTEGRATION_METHOD) -> float:
    assert len(xs) == len(ys), (xs, ys)

    if len(ys) == 1:
        return ys[0]
    else:
        return integration_method(
            ys,
            x=xs,
        ) / (xs[-1] - xs[0])

def an(xs: Vector, ys: Vector, m: int, a0: float) -> float:
    assert m >= 1, m

    return fourier_coeff(np_cos, xs, ys - a0, m)

def bn(xs: Vector, ys: Vector, m: int, a0: float) -> float:
    assert m >= 1, m

    return fourier_coeff(np_sin, xs, ys - a0, m)

class Term(Iterable):
    def __init__(self, n: int, k_n: float, term_type: str) -> None:
        self.n, self.k_n, self.term_type = n, k_n, term_type

    def __iter__(self) -> Tuple[int, float, str]:
        return iter([self.n, self.k_n, self.term_type])

    def __str__(self) -> str:
        return '''Term(n={n}, k_n={k_n:3.2f}, type='{term_type}')'''.format(
            n=self.n if isinstance(self.n, int) else self.n * 360 / (2 * pi),
            k_n=self.k_n,
            term_type=self.term_type,
        )

    def __repr__(self) -> str:
        return str(self)

def cst(x: Any) -> Any:
    if isinstance(x, Vector):
        return vector([1.0 for _ in x])
    elif type(x) in [float, np_float]:
        return 1.0
    else:
        raise Exception(type(x))

TERM_FCT = {
    'cos': np_cos,
    'sin': np_sin,
    'cst': cst,
}

def fourier_series_fct(terms: List[Term]) -> Callable[[float], float]:
    if len(terms) == 0:
        return lambda x: 0.0 * cst(x)
    else:
        return lambda x: reduce(
            lambda acc, e: acc + e,
            [
                k_n * TERM_FCT[term_type](n * x)
                for (n, k_n, term_type) in terms
            ],
        )

def optimise_fourier_terms(terms: List[Term], xs: Vector, Es: Vector, rmsd_weights: Optional[Vector] = None) -> Tuple[List[Term], float, float]:
    assert all([isinstance(a, Vector) for a in [xs, Es]]), [type(a) for a in [xs, Es] if not isinstance(a, Vector)]

    max_abs_k = 2.0 * (np_max(np_abs([term.k_n for term in terms])) + 1.0)

    def function_to_optimise(x: Vector, *K: List[float]) -> Vector:
        try:
            return reduce(
                lambda acc, e: acc + e,
                [
                    k * TERM_FCT[term_type](n * x)
                    for ((n, _, term_type), k) in zip(terms, K)
                ],
            )
        except:
            print(x)
            print(type(x))
            print(K)
            print(
                [
                    k * TERM_FCT[term_type](n * x)
                    for ((n, _, term_type), k) in zip(terms, K)
                ]
            )
            raise

    def correct_weights(rmsd_weights: Optional[Vector]) -> Vector:
        MIN_VALUE = 0.05
        if rmsd_weights is None:
            return rmsd_weights
        else:
            return vector([max(1.0 - x, MIN_VALUE) for x in rmsd_weights])

    optimised_ks, _ = curve_fit(
        function_to_optimise,
        xs,
        Es,
        [term.k_n for term in terms],
        bounds=(-max_abs_k, max_abs_k),
        sigma=correct_weights(rmsd_weights),
    )

    return (
        [Term(term.n, k_n, term.term_type) for (term, k_n) in zip(terms, optimised_ks)],
        vector_rmsd(
            Es,
            function_to_optimise(xs, *optimised_ks),
            weights=rmsd_weights,
        ),
        vector_rmsd(
            Es,
            function_to_optimise(xs, *optimised_ks),
            weights=None,
        ),
    )

MAX_FREQUENCY = 6

def get_fourier_terms(xs: Vector, Es: Vector, Ns: List[int]) -> List[Term]:
    assert all([isinstance(a, Vector) for a in [xs, Es]]), [type(a) for a in [xs, Es] if not isinstance(a, Vector)]

    A0 = Term(0, a0(xs, Es), 'cst')

    return (
        [A0]
        +
        [Term(n, an(xs, Es, n, A0.k_n), 'cos') for n in Ns]
        +
        [Term(n, bn(xs, Es, n, A0.k_n), 'sin') for n in Ns]
    )

Penalty_Function = Callable[[List[Term]], float]

def penalty_function_for(base_scale: float, penalty_power_exponent: float) -> Penalty_Function:
    return (lambda fit_terms: base_scale * len([1 for fit_term in fit_terms if fit_term.term_type != 'cst']) ** penalty_power_exponent)

DEFAULT_PENALTY_FUNCTION = penalty_function_for(1.0, 1.5)
LINEAR_PENALTY_FUNCTION = penalty_function_for(1.0, 1.0)
QUADRATIC_PENALTY_FUNCTION = penalty_function_for(1.0, 2.0)

def rmsd_score_with_n_terms(
    xs: Union[List, Vector],
    Es: Union[List, Vector],
    keep_n: int = 1,
    should_plot: bool = False,
    max_frequency: Optional[int] = None,
    weights: Optional[Vector] = None,
    penalty_function: Penalty_Function = DEFAULT_PENALTY_FUNCTION,
    debug: Optional[Any] = None,
) -> Tuple[List[Term], float, float]:
    if isinstance(Es, Vector):
        Es_np = Es
    else:
        Es_np = vector(Es)

    if max_frequency is None:
        max_frequency = min(len(xs) // 2, MAX_FREQUENCY)

    Ns = range(1, max_frequency + 1)

    assert 2 * max_frequency <= len(xs), 'Inappropriate max_frequency 2 * {0} > {1}'.format(
        max_frequency,
        len(xs),
    )

    A0 = Term(0, a0(xs, Es_np), 'cst')
    As = [Term(n, an(xs, Es_np, n, A0.k_n), 'cos') for n in Ns]
    Bs = [Term(n, bn(xs, Es_np, n, A0.k_n), 'sin') for n in Ns]

    assert keep_n <= len(As + Bs), 'Not enough terms to keep: {0} > {1}'.format(
        keep_n,
        len(As + Bs),
    )

    coeff_threshold = sorted(
        map(
            lambda coeff: abs(coeff.k_n),
            As + Bs,
        ),
        reverse=True,
    )[keep_n - 1] if keep_n >= 1 else inf

    kept_terms = [A0] + list(
        sorted(
            filter(
                lambda coeff: abs(coeff.k_n) >= coeff_threshold,
                As + Bs,
            ),
            key=lambda coeff: abs(coeff.k_n),
            reverse=True,
        ),
    )

    assert len(kept_terms) == keep_n + 1, kept_terms # We always keep A0

    fourier_series = fourier_series_fct(
        kept_terms,
    )

    if should_plot:
        plot(xs, Es, fourier_series)

    return (
        kept_terms,
        vector_rmsd(
            Es_np,
            vector(evaluate(fourier_series, xs)),
            weights=weights,
        ),
        penalty_function(kept_terms),
    )

def in_degrees(list_of_terms: List[Term]) -> List[Term]:
    return [Term(n * (2 * pi / 360.), k_n, term_type) for (n, k_n, term_type) in list_of_terms]

def in_radians(list_of_terms: List[Term]) -> List[Term]:
    return [Term(n / (2 * pi / 360.), k_n, term_type) for (n, k_n, term_type) in list_of_terms]

MAX_NUM_TERMS = 12

WEIGHTED_RMSD, UNWEIGHTED_RMSD = float, float

def best_fit(
    xs: Sequence[float],
    Es: Sequence[float],
    unit: str = 'rad',
    should_plot: bool = False,
    optimise_final_terms: bool = True,
    debug: Optional[Any] = None,
    rmsd_weights: Optional[Vector] = None,
    penalty_function: Penalty_Function = DEFAULT_PENALTY_FUNCTION,
) -> Tuple[List[Term], WEIGHTED_RMSD, UNWEIGHTED_RMSD]:
    assert unit in ['rad', 'deg'], unit

    # Type casting to numpy arrays (vector)
    xs, Es = map(vector, (xs, Es))
    if rmsd_weights is not None:
        rmsd_weights = vector(rmsd_weights)

    assert not isinf(Es).any(), Es

    if len(xs) == 0:
        return (
            [],
            float('inf'),
            float('inf'),
        )
    else:
        max_keep_n = min(MAX_NUM_TERMS, len(xs) // 2)

        xs_in_rad = (xs if unit == 'rad' else radians(xs)) # pylint: disable=no-member

        get_weighted_rmsd, get_penalty = itemgetter(1), itemgetter(2)

        sorted_all_fits = sorted(
            [
                rmsd_score_with_n_terms(
                    xs_in_rad,
                    Es,
                    keep_n=keep_n,
                    should_plot=should_plot,
                    weights=rmsd_weights,
                    penalty_function=penalty_function,
                    debug=debug,
                )
                for keep_n in range(0, max_keep_n)
            ],
            key=lambda x: get_weighted_rmsd(x) + get_penalty(x)
        )

        if debug is not None:
            debug.write('\n'.join(map(str, [(terms, rmsd, penalty, rmsd + penalty) for (terms, rmsd, penalty) in sorted_all_fits])) + '\n')

        best_fit_terms, best_fit_rmsd, best_fit_penalty = sorted_all_fits[0]

        if optimise_final_terms:
            optimised_best_fit_terms, optimised_best_fit_weighted_rmsd, optimised_best_fit_unweighted_rmsd = optimise_fourier_terms(
                best_fit_terms,
                vector_if_necessary(xs_in_rad),
                vector_if_necessary(Es),
                rmsd_weights=rmsd_weights,
            )

            rmsd_to_compare = (optimised_best_fit_weighted_rmsd if rmsd_weights is not None else optimised_best_fit_unweighted_rmsd)
            try:
                assert rmsd_to_compare <= best_fit_rmsd, [rmsd_to_compare, best_fit_rmsd]
            except AssertionError as e:
                if debug is not None:
                    debug.write(str(e))
                else:
                    optimised_best_fit_terms = best_fit_terms
                    optimised_best_fit_weighted_rmsd, optimised_best_fit_unweighted_rmsd = map(
                        lambda weights: vector_rmsd(
                            Es,
                            fourier_series_fct(best_fit_terms)(xs),
                        ),
                        (rmsd_weights, None),
                    )
        else:
            pass

        return (
            optimised_best_fit_terms if unit == 'rad' else in_degrees(optimised_best_fit_terms),
            optimised_best_fit_weighted_rmsd,
            optimised_best_fit_unweighted_rmsd,
        )

NUMBER_POINTS_FIT = 1000

def plot(xs: Vector, Es: Vector, fitted_function: Any) -> None:
    import matplotlib.pyplot as plt # type: ignore

    fine_xs = linspace(xs[0], xs[-1], NUMBER_POINTS_FIT)

    plt.plot(xs, Es, label='Es')
    plt.plot(
        fine_xs,
        vector(evaluate(fitted_function, fine_xs)),
        label='fit',
    )
    plt.legend()
    plt.show()

