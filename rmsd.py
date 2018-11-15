from numpy import vectorize, sum as sum_np, sqrt as sqrt_np, average as np_average # type: ignore
from typing import Any, Optional

from fourier_series_fit.types_helpers import vector, Vector

def d2(z1: float, z2: float) -> Any:
    '''
    Returns the squared difference between two numbers.

    :param z1: (Float) first number.
    :param z2: (Float) second number.
    :rtype: Float
    '''
    return (z2 - z1) ** 2

# Vectorize d2 to use with numpy arrays
d2_v = vectorize(d2)

def vector_rmsd(ys1: Vector, ys2: Vector, weights: Optional[Vector] = None, should_align: bool = True) -> float:
    '''
    Returns the RMSD (root mean square deviation) between two vectors, with optional weights, while potentially aligning them
    (that is to say finding the translation tha minimises the RMSD between the two vectors).

    :param ys1: (Vector) first vector.
    :param ys2: (Vector) second vector.
    :param weights: (Optional) RMSD weights.
    :param should_align: (Boolean) whether or not to align the two vectors
    (that is to say find the translation tha minimises the RMSD between them)
    :rtype: Float
    '''
    try:
        assert len(ys1) == len(ys2), [ys1, ys2]
    except TypeError:
        print([ys1, ys2])
        raise

    Es_1, Es_2 = [vector(data_set) for data_set in [ys1, ys2]]

    def potentially_aligned(Es: Vector) -> Vector:
        '''
        Returns either the vector if `should_align` is False,
        or the vector centered around 0 otherwise.

        :param Es: (Vector) vector.
        :rtype: Vector
        '''
        return Es - (np_average(Es) if should_align else 0.)

    return sqrt_np(
        np_average(
            d2(
                potentially_aligned(Es_1),
                potentially_aligned(Es_2),
            ),
            weights=weights,
        )
    )
