from numpy import linspace, pi, cos, random
from fit import best_fit, fourier_series_fct, LINEAR_PENALTY_FUNCTION

# Get 40 points linearly distributed between -pi and pi
xs = linspace(-pi, pi, 40)

# F(x) = 5 + 4.cos(x) + 6.cos(5x), with some Gaussian noise
Fs = 5.0 + 4. * cos(xs) + 6. * cos (5 * xs) + 1.5 * random.normal(0, 1.0, len(xs))

# Fit with a linear penalty function P: terms -> 1.0 * len(terms)
fit_terms, weighted_rmsd, unweighted_rmsd = best_fit(xs, Fs, penalty_function=LINEAR_PENALTY_FUNCTION)

interpolated_fct = fourier_series_fct(fit_terms)
# Evaluate the fitting function on 200 equally distributed points between -pi and pi
fine_xs = linspace(-pi, pi, 200)
interpolated_Fs = [interpolated_fct(x) for x in fine_xs]
print('Fit terms:', fit_terms)

# Plot F and its fit
import pylab as p
p.plot(xs, Fs, label='F')
p.plot(fine_xs, interpolated_Fs, label='fit')
p.xlabel('$\phi$')
p.xlim([-pi, pi])
p.legend()
p.show()

# Re-run with debugging on to see the effect of the penalty function
from sys import stderr
fit_terms, weighted_rmsd, unweighted_rmsd = best_fit(xs, Fs, penalty_function=LINEAR_PENALTY_FUNCTION, debug=stderr)
