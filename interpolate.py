from scipy.interpolate import interp1d, splrep, splev # type: ignore
from numpy import isnan, max as np_max, gradient as np_gradient, abs as np_abs, concatenate, linspace # type: ignore
from typing import Tuple, List, Any, Callable, Optional, Union, Dict

from fourier_series_fit.types_helpers import Vector, vector
from fourier_series_fit.exceptions import Discountinuity_Error, Not_Enough_Points, Fit_Error, Duplicated_Values

def flatten(xs: Union[Vector, List[Any]]) -> Vector:
    '''
    Returns a vector containing the first element of each of its elements.

    :param xs: Vector of lists, or vector of vectors.
    :rtype: Vector
    '''
    return vector([x[0] for x in xs])

def gradient(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
    '''
    Returns the (numpy) gradient of a series of values,
    or raise a custom exception (`Not_Enough_Points`) if that fails.

    :param *args: tuple arguments
    :param *kargs: keyword arguments
    :rtype: Any
    '''
    try:
        return np_gradient(*args, **kwargs)
    except ValueError as e:
        raise Not_Enough_Points(str(e))

def cyclise(xs: Vector, ys: Vector, x_period: float = 360., flatten_xs_array: bool = False) -> Tuple[Vector, Vector]:
    '''
    Given periodic data (x and y values) with a known period, which potentially needs to be flattened,
    returns a vector of the cyclised data (that is to say a vector where the last point is also the first point).

    :param xs: (Vector) x values
    :param ys: (Vector) y values
    :param x_period: (Float) period of the data (xs, ys)
    :param flatten_xs_array: (Boolean) whether or not xs should be flattened (see `flatten`)
    :rtype: Tuple of cyclic xs, cyclic ys
    '''
    assert all([isinstance(an_array, Vector) for an_array in [xs, ys]])

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
                    vector([ys[-1]]),
                    ys,
                ],
            ),
        )
    except:
        raise Exception([flat_xs, ys, x_period])

def interpolating_fct(xs: Vector, ys: Vector) -> Callable[[Vector], Vector]:
    '''
    Returns a function that interpolates some data (x and y values) using a cubic spline.

    :param xs: (Vector) x values
    :param ys: (Vector) y values
    :rtype: Callable
    '''
    assert all(isinstance(x, float) for x in xs.tolist()), xs
    if len(set(xs)) != len(xs):
        raise Duplicated_Values('Duplicated values in xs: {0}'.format(xs))

    tck = splrep(xs, ys, k=min(3, len(xs) - 1))
    assert not any([isnan(x) for x in tck[1]]), [xs, ys]

    return (lambda xs_vector: splev(xs_vector, tck))

def normalised_anti_gradient(xs: Vector, ys: Vector, scale: float = 1.0, use_interpolated_gradient: bool = False, max_abs_gradient: Optional[float] = None) -> Vector:
    '''
    Given data (x and y values), returns the normalised anti-gradient, that is to say a number between zero and `scale` (defaults to 1)
    which is equal to one when the gradient is null, and equal to zero when the gradient is maximal.

    :param xs: (Vector) x values.
    :param ys: (Vector) y values.
    :param scale: (Float) maximum value of the normalised anti-gradient. Defaults to 1.
    :param use_interpolated_gradient: (Boolean) whether or not to interpolate the numerical gradient with a smooth (cubic) function
    :param max_abs_gradient: (Float) maximum absolute gradient that should be tolarated. Will throw an exception if the gradient exceeds this value.
    Useful for catching outliers.
    :rtype: Vector of the normalised anti-gradient.
    '''
    assert all([isinstance(an_array, Vector) for an_array in [xs, ys]])

    if use_interpolated_gradient:
        try:
            fine_xs = linspace(xs[0], xs[-1], 150)
        except:
            raise Exception(xs[0], xs[-1])
        interpolated_Es = interpolating_fct(xs, ys)(fine_xs)
        fine_d_ys = interpolating_fct(fine_xs, gradient(interpolated_Es))(fine_xs)
        fine_d_xs = gradient(fine_xs)
        fine_total_gradient = fine_d_ys / fine_d_xs
        # Necessary as the spacing can be uneven and gradient does not take a spacing argument
        total_gradient = interpolating_fct(fine_xs, fine_total_gradient)(xs)
    else:
        d_xs = gradient(xs)
        d_ys = gradient(ys)
        # Necessary as the spacing can be uneven and gradient does not take a spacing argument
        total_gradient = d_ys / d_xs

    try:
        absolute_gradient = np_abs(total_gradient)
    except:
        raise Exception([d_xs, d_ys])

    if max_abs_gradient is not None:
        if any([d > max_abs_gradient for d in absolute_gradient]):
            raise Discountinuity_Error(
                'Found possible discontinuity (rate of change > {0}) in gradient for {1}.\nIncrease the value of max_abs_gradient or add points to the fit.'.format(
                    max_abs_gradient,
                    dict(xs=xs, ys=ys, gradient=absolute_gradient),
                )
            )

    anti_gradient = (1.0 - (absolute_gradient / np_max(absolute_gradient)))

    if sum(anti_gradient) == 0.:
        anti_gradient = vector([1. for x in anti_gradient])

    assert sum(anti_gradient) != 0., anti_gradient

    return (
        xs,
        scale * anti_gradient,
    )
