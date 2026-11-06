"""This file produces an animation of the uniform advection figure in the paper (figures/uniform_advection.pdf from fig_uniform_advection in main.py),
showing all time steps up to t=1. The four panels use different time steps, so they are shown on one common clock."""


import os
import io
import string
from fractions import Fraction
from math import lcm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import schemes as sch
import analytic as an


def anim_uniform_advection(duration=16., pause=2., dpi=80):
    """This function saves an animation of the paper's uniform advection figure with all time steps up to one revolution (t=1).
    All panels follow one clock: there is a frame at every time at which at least one panel reaches a new time step, each panel shows its latest
    time step at that time (n = floor(t/dt)), and each frame lasts until the next such time, so wall-clock time is proportional to t.
    duration : float, wall-clock time (s) of one revolution. The shortest time between frames is 1/800, which lasts 20 ms (the shortest GIF frame
               that browsers play at the right speed) for duration=16. For shorter durations, frames shorter than 20 ms are merged into the next frame.
    pause    : float, wall-clock time (s) that the final time t=1 (the paper figure) is held for
    dpi      : float, resolution of the animation (the figure size and font sizes are as in the paper)
    """

    # Setup (as for the paper figure in fig_uniform_advection in main.py)
    nx, xmax = 40, 1.
    dx = xmax/nx
    dt = [0.01, 0.03125, 0.0625, 0.1]
    u = 1.
    C = [dt_i*u/dx for dt_i in dt]
    nt = [int(1./dt_i) for dt_i in dt]
    xf = np.linspace(0., xmax, nx, endpoint=False)
    xc = xf + 0.5*dx
    init = an.combi(xc, xmax, u=0., t=0.)
    theta = [sch.implicitness(C_i) for C_i in C]

    # Run schemes, keeping all time steps
    styles = {'AdHImEx': dict(color='black', linestyle='-', marker='x'),
              'AdHImEx AdImEx FCT': dict(color='blue', linestyle='-'),
              'AdHImEx NN FCT': dict(color='cyan', linestyle='-', marker='x'),
              'WKS24': dict(color='magenta', linestyle='-')}
    psi = []
    for dt_i, nt_i in zip(dt, nt):
        psi.append({'AdHImEx': sch.AdHImEx(init, nt_i, dt_i, np.full((nt_i,nx), u), np.full(nx,dx)),
                    'AdHImEx AdImEx FCT': sch.AdHImEx(init, nt_i, dt_i, np.full((nt_i,nx), u), np.full(nx,dx), FCT=True, nondivergent=True),
                    'AdHImEx NN FCT': sch.AdHImEx(init, nt_i, dt_i, np.full((nt_i,nx), u), np.full(nx,dx), FCT=True, ymin=0., ymax=10000., nondivergent=True),
                    'WKS24': sch.WKS24(init, nt_i, dt_i, np.full((nt_i,nx), u), np.full(nx,dx))})

    # Common clock in integer ticks: a tick (1/800) is the longest time that divides all dt, so time step n of panel i is at tick n*nticks[i]
    dt_exact = [Fraction(dt_i).limit_denominator() for dt_i in dt]
    tick = Fraction(1, lcm(*[dt_exact_i.denominator for dt_exact_i in dt_exact]))
    nticks = [int(dt_exact_i/tick) for dt_exact_i in dt_exact]
    tick_end = nticks[0]*nt[0] # one revolution, the same for all panels
    ticks = sorted(set(n*nticks_i for nticks_i, nt_i in zip(nticks, nt) for n in range(nt_i+1))) # times at which any panel reaches a new time step

    # Wall-clock start time of each frame in centiseconds (GIF frame durations are multiples of 10 ms), merging frames shorter than 20 ms into the next frame
    start = [round(T/tick_end*duration*100) for T in ticks]
    keep = [k for k in range(len(ticks)) if k == len(ticks)-1 or start[k+1] - start[k] >= 2]
    ticks, start = [ticks[k] for k in keep], [start[k] for k in keep]
    durations = [10*(start_next - start_k) for start_k, start_next in zip(start[:-1], start[1:])] + [int(1000*pause)]

    # Plot as in the paper, with the analytic solution in place of the initial condition and the y-range of all time steps
    fig, ax = plt.subplots(2, 2, figsize=(20,12))
    ymin = min(psi_i[name].min() for psi_i in psi for name in styles)
    ymax = max(psi_i[name].max() for psi_i in psi for name in styles)
    lines, titles = [], []
    for i, a in enumerate(ax.flat):
        lines.append({'Analytic': a.plot(xc, init, color='gray', linestyle='--', label='Analytic')[0]})
        for name, style in styles.items():
            lines[i][name] = a.plot(xc, psi[i][name][0], label=name, **style)[0]
        titles.append(a.set_title('', size=20))
        a.text(0.04, 0.9, string.ascii_lowercase[i], transform=a.transAxes, size=20, weight='bold')
        a.tick_params(labelsize=20)
        a.set_xlim(0.,1.)
        a.set_ylim(ymin - 0.05*(ymax - ymin), ymax + 0.05*(ymax - ymin)) # default matplotlib margins
        a.set_xlabel('x', size=20)
        a.set_ylabel('$\\Psi$', size=20)
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc='lower center', fontsize=20)
    clock = fig.suptitle('', size=24)

    # Show each panel at its latest time step at tick T, with all times to the number of decimals that shows every tick exactly
    decimals = next(d for d in range(10) if (tick*10**d).denominator == 1)
    def update(T):
        for i in range(len(dt)):
            n = T//nticks[i]
            t = float(n*dt_exact[i])
            lines[i]['Analytic'].set_ydata(an.combi(xc, xmax, u=u, t=t))
            for name in styles:
                lines[i][name].set_ydata(psi[i][name][n])
            titles[i].set_text(f'$C={C[i]:.2f}$, $\\theta={theta[i]:.2f}$, $n_t={n}$, $t={t:.{decimals}f}$')
        clock.set_text(f'$t={float(T*tick):.{decimals}f}$')

    update(0)
    fig.tight_layout(rect=(0., 0.06, 1., 0.95))

    # Render the frames and save them as a GIF with their durations
    images = []
    for T in ticks:
        update(T)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        images.append(Image.open(buf).convert('RGB'))
    images[0].save('animations/uniform_advection.gif', save_all=True, append_images=images[1:], duration=durations, loop=0)
    plt.close(fig)


def main():

    if not os.path.exists('animations'):
        os.makedirs('animations')

    print('Producing animations...')

    ######## ANIMATION: Uniform advection (paper figure) ########
    anim_uniform_advection()

    print('...done')


if __name__ == "__main__":
    main()
