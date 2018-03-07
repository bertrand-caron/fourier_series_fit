from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['CMU Serif']})

from numpy import linspace, pi, cos, random

# Get 40 points linearly distributed between -pi and pi
xs = linspace(-pi, pi, 500)

N_1, N_2 = 2, 3

import pylab as p
for N in [N_1, N_2]:
    p.plot(
        xs,
        cos(N * xs),
        label='$\cos({0}x)$'.format('' if N == 1 else N)
    )
p.fill_between(
    xs,
    0,
    cos(N_1 * xs) *  cos(N_2 * xs),
    label='$\cos({0}x) . \cos({1}x)$'.format(
        *[
            '' if N == 1 else N
            for N in [N_1, N_2]
        ]
    ),
    color='#FED8B1',
    edgecolor='black',
    facecolor='black',
)
p.xlabel('$\phi$')
p.xlim([-pi, pi])
p.legend()
p.show()
