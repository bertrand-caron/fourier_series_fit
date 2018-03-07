from sys import argv
from matplotlib import rc
rc('text', usetex=True)
import pylab as p
from numpy import linspace, pi, cos, random

xs = linspace(-pi, pi, 500)

N_1, N_2 = 2, 3

PLOT_TO_FILE = True

fig, axis = p.subplots(figsize=(8, 6))

for N in [N_1, N_2]:
    axis.plot(
        xs,
        cos(N * xs),
        label='$\cos({0}x)$'.format('' if N == 1 else N)
    )
axis.fill_between(
    xs,
    0,
    cos(N_1 * xs) *  cos(N_2 * xs),
    label='$\cos({0}x) . \cos({1}x)$'.format(
        *[
            '' if N == 1 else N
            for N in [N_1, N_2]
        ]
    ),
    color='#DCDCDC',
    edgecolor='black',
    facecolor='black',
)
#axis.plot(xs, cos(N_1 * xs) *  cos(N_2 * xs), color='#808080')
axis.set_xlabel('$\phi$ (rad)')
axis.set_xlim([-pi, pi])
axis.legend()
if PLOT_TO_FILE:
    p.savefig(argv[0].replace('.py', '.pdf'))
else:
    p.show()
