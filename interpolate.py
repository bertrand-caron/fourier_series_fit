from scipy.interpolate import interp1d, splrep, splev
from numpy import isnan, max as np_max, gradient, abs as np_abs, concatenate, linspace
from typing import Tuple, List, Any, Callable, Optional

from fourrier_series_fit.types_helpers import Vector, vector

def flatten(xs: Vector) -> Vector:
    assert isinstance(xs, Vector), xs

    return vector([x[0] for x in xs])

def cyclise(xs: Vector, Es: Vector, x_period: float = 360., flatten_xs_array: bool = False) -> Tuple[Vector, Vector]:
    assert all([isinstance(an_array, Vector) for an_array in [xs, Es]])

    if flatten_xs_array:
        flat_xs = flatten(xs)
    else:
        flat_xs = xs

    try:
        return (
            concatenate(
                [
                    vector([flat_xs[-1] - x_period]),
                    flat_xs,
                ],
            ),
            concatenate(
                [
                    vector([Es[-1]]),
                    Es,
                ],
            ),
        )
    except:
        raise Exception([flat_xs, Es, x_period])

def interpolating_fct(xs: Vector, Es: Vector) -> Callable[[Vector], Vector]:
    tck = splrep(xs, Es, k=3)
    assert not any([isnan(x) for x in tck[1]]), [xs, Es]

    return (lambda xs_vector: splev(xs_vector, tck))

def normalised_anti_gradient(xs: Vector, Es: Vector, scale: float = 1.0, use_interpolated_gradient: bool = False, max_abs_gradient: float = 100.) -> Vector:
    assert all([isinstance(an_array, Vector) for an_array in [xs, Es]])

    if use_interpolated_gradient:
        try:
            fine_xs = linspace(xs[0], xs[-1], 150)
        except:
            raise Exception(xs[0], xs[-1])
        interpolated_Es = interpolating_fct(xs, Es)(fine_xs)
        fine_d_ys = interpolating_fct(fine_xs, gradient(interpolated_Es))(fine_xs)
        fine_d_xs = gradient(fine_xs)
        fine_total_gradient = fine_d_ys / fine_d_xs
        # Necessary as the spacing can be uneven and gradient does not take a spacing argument
        total_gradient = interpolating_fct(fine_xs, fine_total_gradient)(xs)
    else:
        d_xs = gradient(xs)
        d_ys = gradient(Es)
        # Necessary as the spacing can be uneven and gradient does not take a spacing argument
        total_gradient = d_ys / d_xs

    try:
        absolute_gradient = np_abs(total_gradient)
    except:
        raise Exception([d_xs, d_ys])

    assert all([d < max_abs_gradient for d in absolute_gradient]), [absolute_gradient]

    return (
        xs,
        scale * (1.0 - (absolute_gradient / np_max(absolute_gradient))),
    )
