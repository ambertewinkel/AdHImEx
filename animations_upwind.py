"""This file produces animations of the uniform advection test case (cosine bell and top hat, as in fig_uniform_advection in main.py) for the explicit, implicit and AdImEx upwind schemes."""


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import schemes as sch
import analytic as an


def implicitness_explicit(C):
    """This function returns theta=0 (fully explicit, forward Euler) for any Courant number C."""
    return np.zeros_like(C)


def implicitness_implicit(C):
    """This function returns theta=1 (fully implicit, backward Euler) for any Courant number C."""
    return np.ones_like(C)


def implicitness_AdImEx_upwind(C):
    """This function returns the implicitness theta = max(0, 1 - 1/C) of the first-order bounded AdImEx upwind scheme (Weller et al., 2023),
    i.e. the minimum theta for which the explicit part is bounded ((1 - theta)*C <= 1)."""
    return 1. - 1./np.maximum(C, 1.)


def upwind(init, nt, dt, uf, dxc, implicitness_function):
    """Implements the first-order upwind scheme with theta-weighted forward and backward Euler time stepping (theta=0 explicit, theta=1 implicit),
    using the upwind flux and matrix from schemes.py. Returns the field at all time steps, including the initial condition: field[nt+1, nx].
    Assumes periodic boundaries, uniform grid and nonnegative velocity at faces.
    init : 1D array of floats, initial condition at t=0
    nt   : int, number of time steps
    dt   : float, time step length
    uf   : 2D array of floats, velocity field at cell faces for all time
    dxc  : 1D array of floats, cell widths
    implicitness_function: function, theta as a function of the Courant number
    """
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()

    # Time-stepping loop
    for it in range(nt):
        thetaf = implicitness_function(uf[it]*dt/dxc) # [i] at i-1/2
        flxEx = (1. - thetaf)*uf[it]*sch.space_upwind(field[it]) # [i] at i-1/2
        rhs = field[it] - dt*sch.ddx(flxEx, np.roll(flxEx,-1), dxc) # [i] at i
        M = sch.matrix_upwind(nx, dt, dxc, thetaf*uf[it], 1.) # [i] at i
        field[it+1] = np.linalg.solve(M, rhs) # [i] at i

    return field


def anim_uniform_advection_upwind(duration=12., pause=1.):
    """This function saves an animation of one revolution of uniform advection for each upwind scheme and Courant number.
    Each frame is one time step and lasts duration/nt seconds, so all animations have the same wall-clock time
    (10x fewer time steps give 10x fewer frames that each last 10x longer).
    duration : float, wall-clock time (s) of one revolution
    pause    : float, wall-clock time (s) that the final time step is held for
    """

    # Setup
    nx, xmax = 40, 1.
    dx = xmax/nx
    u = 1.
    C = [0.16, 1.6] #[0.25, 2.5]
    dt = [C_i*dx/u for C_i in C]
    nt = [int(round(xmax/(u*dt_i))) for dt_i in dt] # one revolution
    xf = np.linspace(0., xmax, nx, endpoint=False)
    xc = xf + 0.5*dx
    xfine = np.linspace(0., xmax, 1001) # to plot the analytic solution
    init = an.combi(xc, xmax, u=0., t=0.)
    schemes = {'Explicit upwind': implicitness_explicit,
               'Implicit upwind': implicitness_implicit,
               'AdImEx upwind': implicitness_AdImEx_upwind}

    # Run schemes and animate
    for name, implicitness_function in schemes.items():
        for C_i, dt_i, nt_i in zip(C, dt, nt):
            theta_i = float(implicitness_function(C_i))
            psi = upwind(init, nt_i, dt_i, np.full((nt_i,nx), u), np.full(nx,dx), implicitness_function)

            fig, ax = plt.subplots(figsize=(6.5,4))
            line_analytic, = ax.plot(xfine, an.combi(xfine, xmax, u=u, t=0.), color='gray', linestyle='--', label='Analytic')
            line_psi, = ax.plot(xc, psi[0], color='firebrick', linestyle='-', label=name, marker='x')
            title = ax.set_title('', size=15)
            ax.tick_params(labelsize=13)
            ax.set_xlim(0., xmax)
            ax.set_ylim(-0.25, 1.25)
            ax.set_xlabel('x', size=15)
            ax.set_ylabel('$\\Psi$', size=15)
            #fig.legend(loc='lower center', ncol=2, fontsize=13, frameon=False)

            # Update the fields and the time in the title to time step it
            def update(it):
                line_analytic.set_ydata(an.combi(xfine, xmax, u=u, t=it*dt_i))
                line_psi.set_ydata(psi[it])
                title.set_text(f'{name}, $C={C_i:.2f}$, $\\theta={theta_i:.2f}$, $t={it*dt_i:.3f}$')
                return line_analytic, line_psi, title

            update(0)
            fig.tight_layout(rect=(0., 0.07, 1., 1.))

            # One frame per time step, then hold the final time step for pause seconds
            fps = nt_i/duration
            frames = list(range(nt_i)) + [nt_i]*max(1, int(round(pause*fps)))
            anim = FuncAnimation(fig, update, frames=frames)
            anim.save(f'animations/uniform_advection_{name.lower().replace(" ", "_")}_C{C_i:g}.gif', writer=PillowWriter(fps=fps), dpi=160)
            plt.close(fig)


def main():

    if not os.path.exists('animations'):
        os.makedirs('animations')

    print('Producing animations...')

    ######## ANIMATION: Uniform advection with upwind schemes ########
    anim_uniform_advection_upwind()

    print('...done')


if __name__ == "__main__":
    main()
