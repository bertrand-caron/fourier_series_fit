from numpy import vectorize, sum as sum_np, sqrt as sqrt_np, average as np_average
from typing import Any, Optional

from fourier_series_fit.types_helpers import vector, Vector

def d2(z1: Any, z2:Any) -> Any:
    return (z2 - z1)**2

d2_v = vectorize(d2)

def vector_rmsd(ys1: Vector, ys2: Vector, weights: Optional[Vector] = None, should_align: bool = True) -> float:
    try:
        assert len(ys1) == len(ys2), [ys1, ys2]
    except TypeError:
        print([ys1, ys2])
        raise

    Es_1, Es_2 = [vector(data_set) for data_set in [ys1, ys2]]

    def potentially_aligned(Es: Vector) -> Vector:
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

