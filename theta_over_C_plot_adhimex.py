"""This file produces the figure of the AdHImEx implicitness theta over the Courant number C (formatted as fig_amplification_factor in main.py, without the amplification factor)."""


import os
import numpy as np
import matplotlib.pyplot as plt
import schemes as sch


def fig_theta_over_C():

    # Calculate implicitness as defined in the AdHImEx paper: theta = 1 - 1/(1 + 0.7*max(0, C - 1.4))
    C = np.logspace(-1, 2, 401, endpoint=True)
    theta_AdHImEx = sch.implicitness(C)
    print(theta_AdHImEx)

    # Produce and save figure
    fig = plt.figure(figsize=(3.5,3))
    plt.plot(C, theta_AdHImEx, 'r', linewidth=1.5, label=f'AdHImEx $\\theta$')
    plt.legend()
    plt.xscale('log')
    plt.xlim(C[0], C[-1])
    plt.ylim(0., 1.)
    plt.xlabel('C')
    plt.ylabel(f'$\\theta$')
    plt.tight_layout()
    fig.savefig('figures/theta_over_C_adhimex.pdf', dpi=300)
    plt.close(fig)


def main():

    if not os.path.exists('figures'):
        os.makedirs('figures')

    print('Producing figure...')

    ######## FIGURE: Implicitness over C ########
    fig_theta_over_C()

    print('...done')


if __name__ == "__main__":
    main()
