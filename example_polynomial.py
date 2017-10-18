from numpy import linspace, cos, pi
from fit import get_fourier_terms, fourier_series_fct
from rmsd import vector_rmsd
import pylab as p

xs_in_rad = linspace(-pi, pi, 500)

FUNCTION_LATEX_FORM, FUNCTION = ('$-0.025 \ x^6 + 1 \ x^4 - 2  \pi \ x^2$', lambda xs: - 0.025 * xs ** 6 + xs ** 4 - 2 * pi * xs ** 2)

Es = FUNCTION(xs_in_rad)

all_terms = get_fourier_terms(xs_in_rad, Es, range(1, 100))

sorted_terms = sorted(
    all_terms,
    key=lambda term: (term.n != 0, -abs(term.k_n)),
)

figure, axis = p.subplots()
axis.plot(xs_in_rad, Es, label=FUNCTION_LATEX_FORM, linewidth=3.0)

cmap = p.get_cmap('brg')

Ns = [0, 1, 2, 3, 4, 5, 10, 25, 100]

for (i, N) in enumerate(Ns):
    print(N, sorted_terms[:N + 1])
    fit_fct = fourier_series_fct(sorted_terms[:N + 1])
    axis.plot(
        xs_in_rad,
        fit_fct(xs_in_rad),
        label='$N={0:<3d}\ (RMSD={1:.1f})$'.format(N, vector_rmsd(Es, fit_fct(xs_in_rad))),
        marker='',
        linestyle='-',
        color=cmap(1.0 - (i / (len(Ns) * 1.))),
    )

axis.set_xlabel('$\phi\ (rad)$')
axis.set_ylabel('$F$')
axis.set_xlim((-pi, pi))

NCOLS = 1
axis.legend(loc='upper right', scatterpoints=1)

p.show()
