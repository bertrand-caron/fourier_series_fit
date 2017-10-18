# Fourier Series Fit

[![DOI](https://zenodo.org/badge/75901428.svg)](https://zenodo.org/badge/latestdoi/75901428)

## Description

* `fourier_series_fit` implements the Fourier series fitting of periodic scalar functions using a series of trigonometric functions.

## Usage

* `fit.best_fit()` implements the main fitting function.
Given a series of 2D, scalar data points `(xs, Fs)` and a penalty function `p`,
`fit.best_fit(xs, Fs, penalty_function=p)` returns a list of terms
as well as a measure of the goodness of fit (weighted and unweight root mean square deviation).
The list of terms can be used to crate an interpolation function using `fourier_series_fct(list_of_terms)` used to evaluate the fitting function at any point.

## Example

### Trignonometric function

Cf `example.py`:

```
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
```

![Example Trigonometric](https://raw.githubusercontent.com/bertrand-caron/fourier_series_fit/master/figures/example_trigonometric.png)

### Polynomial function

![Example Polynomial](https://raw.githubusercontent.com/bertrand-caron/fourier_series_fit/master/figures/example_polynomial.png)