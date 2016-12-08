from numpy import cos as np_cos, rad2deg, pi, random, sin as np_sin, linspace, isclose # pylint: disable=no-name-in-module
from scipy.integrate import simps

from fourrier_series_fit.fit import best_fit, plot, optimise_fourrier_terms, a0, an, bn, convolution_v, evaluate, fourrier_series_fct
from fourrier_series_fit.types_helpers import vector, Vector

def test_fourrier_coeffs(xs: Vector, Es: Vector) -> None:
    assert isclose(
        simps(
            convolution_v(np_sin(2 * xs), 10 * np_cos(xs) + np_sin(1 * xs)),
            xs,
        ),
        0.0,
        atol=1E-5,
    )
    A0 = a0(xs, np_cos(xs))
    print(bn(xs, np_cos(xs), 1, A0))
    print(bn(xs, np_cos(xs), 2, A0))

if __name__ == '__main__':
    xs_in_rad = linspace(0, 2 * pi, 120)
    Es = 24.0 + 10. * np_cos(1. * xs_in_rad - pi / 3) + 20. * np_cos(2.0 * xs_in_rad + pi / 4) + 2.0 * random.normal(0., 1.5, len(xs_in_rad)) # pylint: disable=no-member

    fit_in_rad = best_fit(xs_in_rad, Es, should_plot=False, optimise_final_terms=True)

    xs_in_deg = rad2deg(xs_in_rad)
    print(xs_in_deg)
    fit_in_deg = best_fit(xs_in_deg, Es, unit='deg', should_plot=False, optimise_final_terms=True)

    for ((fit_terms, fit_rmsd), fit_unit) in zip([fit_in_rad, fit_in_deg], ['rad', 'deg']):
        print('Optimising fit in {0}'.format(fit_unit))
        print('Original fit terms: {0} (RMSD={1})'.format(fit_terms, fit_rmsd))
        print('Optimised fit terms: {0} (RMSD={1})\n'.format(*optimise_fourrier_terms(fit_terms, xs_in_rad, Es)))

    print(fit_in_rad)
    print(fit_in_deg)

    if True:
        import matplotlib.pyplot as plt

        plt.plot(
            xs_in_deg,
            Es,
        )
        plt.plot(
            xs_in_deg,
            evaluate(
                fourrier_series_fct(
                    fit_in_deg[0],
                ),
                xs_in_deg,
            ),
        )
        plt.show()
