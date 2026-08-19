"""This file includes functions to produce the figures and data for the AdHImEx paper."""


import os
import numpy as np
import amplification_factor as af
import matplotlib.pyplot as plt
import schemes as sch
import analytic as an
import matplotlib as mpl
import string
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def fig_amplification_factor():

    # Calculate amplification factor and implicitness
    C = np.logspace(-1, 2, 401, endpoint=True)
    theta = np.linspace(0., 1., 401, endpoint=True)
    A = af.calculate_amplification_AdHImEx(C, theta, kdx=np.linspace(0., 2*np.pi, 401, endpoint=True))
    theta_AdHImEx = sch.implicitness(C)

    # Design colorbar
    bounds = np.array([0.,0.5, 1.00000001])
    bounds = np.append(bounds, np.logspace(0,5,21)[1:])
    cmap = mpl.colormaps['viridis'].resampled(len(bounds))
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0], cmaplist[1] = (.8, .8, .8, 1.0), (.8, .8, .8, 1.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Produce and save figure
    fig = plt.figure(figsize=(4,3))
    C2D, theta2D = np.meshgrid(C, theta)
    plt.contourf(C2D, theta2D, A, bounds, cmap=cmap, norm=norm, extend='max')
    plt.plot(C, theta_AdHImEx, 'r--', linewidth=0.7, label=f'AdHImEx $\\theta$')
    cbar = plt.colorbar(label='$\\max$$_{k \\Delta x}$$||A||$', boundaries = bounds, extend='max')
    cbar.set_ticks([0, 1, 1e1, 1e2, 1e3, 1e4, 1e5])
    cbar.set_ticklabels(['0', '1', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$'])
    plt.legend()
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel(f'$\\theta$')
    plt.tight_layout()
    fig.savefig('figures/amplification_factor.pdf', dpi=300)
    plt.close(fig)

 
def fig_uniform_advection():

    # Setup
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

    # Run schemes and plot
    fig, ax = plt.subplots(2, 2, figsize=(20,12))
    axrow = 0
    axcol = 0
    for C_i, dt_i, nt_i, theta_i in zip(C, dt, nt, theta):
        psi_AdHImEx = sch.AdHImEx(init, nt_i, dt_i, np.full((nt_i,nx), u), np.full(nx,dx))
        psi_AdHImEx_FCT1 = sch.AdHImEx(init, nt_i, dt_i, np.full((nt_i,nx), u), np.full(nx,dx), FCT=True, nondivergent=True)
        psi_AdHImEx_FCTNN = sch.AdHImEx(init, nt_i, dt_i, np.full((nt_i,nx), u), np.full(nx,dx), FCT=True, ymin=0., ymax=10000., nondivergent=True)
        psi_WKS24 = sch.WKS24(init, nt_i, dt_i, np.full((nt_i,nx), u), np.full(nx,dx))
        ax[axrow, axcol].plot(xc, init, color='gray', linestyle='--', label='Initial')
        ax[axrow, axcol].plot(xc, psi_AdHImEx[-1], color='black', linestyle='-', label='AdHImEx', marker='x')
        ax[axrow, axcol].plot(xc, psi_AdHImEx_FCT1[-1], color='blue', linestyle='-', label='AdHImEx AdImEx FCT')
        ax[axrow, axcol].plot(xc, psi_AdHImEx_FCTNN[-1], color='cyan', linestyle='-', label='AdHImEx NN FCT', marker='x')
        ax[axrow, axcol].plot(xc, psi_WKS24[-1], color='magenta', linestyle='-', label='WKS24')
        ax[axrow, axcol].set_title(f'$C={C_i:.2f}$, $\\theta={theta_i:.2f}$', size=20)
        ax[axrow, axcol].text(0.04, 0.9, string.ascii_lowercase[C.index(C_i)], transform=ax[axrow, axcol].transAxes, size=20, weight='bold')
        ax[axrow, axcol].tick_params(labelsize=20)
        ax[axrow, axcol].set_xlim(0.,1.)
        ax[axrow, axcol].set_xlabel('x', size=20)    
        ax[axrow, axcol].set_ylabel('$\\Psi$', size=20)
        if axcol == 0 and axrow == 0:
            handles, labels = ax[axrow, axcol].get_legend_handles_labels()
        axcol += 1
        if axcol == 2:
            axcol = 0
            axrow += 1
    
    # Further plot details
    fig.legend(handles, labels, ncol=5, bbox_to_anchor=(0.5, -0.05), loc='lower center', fontsize=20)
    plt.tight_layout()
    fig.savefig('figures/uniform_advection.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_order_of_accuracy():

    # Setup
    xmax = 1.0
    Cfac = [1.e2, 20., 8., 4., 1.]
    nx = np.array([80, 40, 20], dtype=int)
    dx = xmax/nx
    dt = np.array([0.125, 0.25, 0.5], dtype=float)
    nt = np.array([4, 2, 1])
    u = 1.
    C = [dt[0]*u/dx[0]/Cfac_i for Cfac_i in Cfac]
    theta = [sch.implicitness(C_i) for C_i in C]

    # Run schemes and plot
    fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True)
    colors = ['black', '#810f7c','#8856a7','#8c96c6','#b3cde3']
    for C_i, Cfac_i, theta_i, colors_i in zip(C, Cfac, theta, colors):
        l2_WKS24, l2_AdHImEx = [], []
        for dt_i, dx_i, nt_i, nx_i in zip(dt, dx, nt, nx):
            nt_temp = int(nt_i*Cfac_i)
            dt_temp = dt_i/Cfac_i
            xc = np.linspace(0., xmax, nx_i, endpoint=False)
            init = an.sine(xc, xmax, u=0., t=0.)
            analytic = an.sine(xc, xmax, u=u, t=dt_i*nt_i)
            psi_WKS24 = sch.WKS24(init, nt_temp, dt_temp, np.full((nt_temp,nx_i), u), np.full(nx_i, dx_i))
            psi_AdHImEx = sch.AdHImEx(init, nt_temp, dt_temp, np.full((nt_temp,nx_i), u), np.full(nx_i, dx_i))
            l2_WKS24.append(an.l2norm(psi_WKS24[-1], analytic, dx_i))
            l2_AdHImEx.append(an.l2norm(psi_AdHImEx[-1], analytic, dx_i))
        ax[0].plot(dx, l2_WKS24, marker='x', label=f'$C={C_i:.2f}$', color=colors_i)
        ax[1].plot(dx, l2_AdHImEx, marker='x', label=f'$C={C_i:.2f}$', color=colors_i)

    # Plotting details
    gridscale = np.logspace(0, np.log10(4), num=3)
    firstorder = 2e-2*gridscale
    secondorder = 1e-3*gridscale**2
    thirdorder = 3e-5*gridscale**3 #3e-6*gridscale**3
    fifthorder = 7e-8*gridscale**5
    for a in ax:
        a.plot(dx, firstorder, color='grey', linestyle=':')
        a.plot(dx, secondorder, color='grey', linestyle=':')
        a.plot(dx, thirdorder, color='grey', linestyle=':')
        a.plot(dx, fifthorder, color='grey', linestyle=':')
        a.set_yscale('log')
        a.set_xscale('log')
        a.set_xlabel('$\\Delta x$')
    ax[0].legend(loc='lower right')
    ax[0].set_ylabel(r'$\ell_2$ norm')
    ax[0].set_title('WKS24')
    ax[1].set_title('AdHImEx')
    fig.savefig('figures/order_of_accuracy.pdf', dpi=300)
    plt.close(fig)

def fig_substages():

    # Setup
    xmax = 1.0
    nx = 250
    dt_AdImEx = 0.004
    nt_AdImEx = 1
    dtfactor_ExAdImEx = 20
    dt_Ex = dt_AdImEx/dtfactor_ExAdImEx
    nt_Ex = nt_AdImEx*dtfactor_ExAdImEx
    xf = np.linspace(0., xmax, nx, endpoint=False)
    dx = np.full(nx, xf[1] - xf[0])
    xc = xf + 0.5*dx
    psi_in = an.sine(xc, xmax, u=0., t=0., shifty=50., ampl=50., shiftx=0.3)
    uf = np.zeros((nt_Ex, nx))
    for it in range(nt_Ex):
        uf[it] = an.velocity_varying_space(xf)

    # Run schemes
    nstages = 7    
    substages_spurdiv, substages = np.zeros((nstages,nx)), np.zeros((nstages,nx))
    psi_AdHImEx_spurdiv = sch.AdHImEx(psi_in, nt_AdImEx, dt_AdImEx, uf, dx, unity=False, output_substages=True, substages=substages_spurdiv)
    psi_AdHImEx = sch.AdHImEx(psi_in, nt_AdImEx, dt_AdImEx, uf, dx,output_substages=True, substages=substages)
    psi_Ex = sch.AdHImEx(psi_in, nt_Ex, dt_Ex, uf, dx)

    # Calculate l2 norms of final time steps
    l2_AdHImEx_spurdiv = an.l2norm(psi_AdHImEx_spurdiv[-1], psi_Ex[-1], dx[0])
    l2_AdHImEx_unity = an.l2norm(psi_AdHImEx[-1], psi_Ex[-1], dx[0])
    with open('substage_l2norms.out', 'w') as f:
        f.write(f'l2 norm AdHImEx not unity-preserving: {l2_AdHImEx_spurdiv:.6e}\n')
        f.write(f'l2 norm AdHImEx unity-preserving: {l2_AdHImEx_unity:.6e}\n')

    # Plot substages fields
    fig, ax = plt.subplots(1, 2, figsize=(20,6.5), sharey=True)
    colors = ['blue','olive','mediumaquamarine']
    for a in ax:
        a.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
        a.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
        a.plot(xc, psi_in, linestyle='--', color='gray', label='Initial')
        a.plot(xc, psi_Ex[-1], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    for isub in range(2,nstages-2):
        ax[0].plot(xc, substages_spurdiv[isub], marker='x', linestyle='-', color=colors[isub-2], label=f'$s={isub}$')
        ax[1].plot(xc, substages[isub], marker='x', linestyle='-', color=colors[isub-2], label=f'$s={isub}$')
    ax[0].plot(xc, substages_spurdiv[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    ax[1].plot(xc, substages[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    for a in ax:
        a.tick_params(labelsize=15)
        a.set_xlim(0.,1.)
        a.set_xlabel('x', size=15)
    ax[0].set_ylabel('$\\Psi$', size=15)
    ax[0].legend(fontsize=15)
    ax[0].set_title('Without constancy preservation', size=15)
    ax[1].set_title('With constancy preservation', size=15)

    # First inset in both plots
    x1_range = (0.28, 0.33)
    inset1_ax0 = ax[0].inset_axes([0.05, 0.55, 0.2, 0.4])
    inset1_ax0.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
    inset1_ax0.plot(xc, psi_in, linestyle='--', color='gray')
    inset1_ax0.plot(xc, psi_Ex[-1], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    for isub in range(2,nstages-2):
        inset1_ax0.plot(xc, substages_spurdiv[isub], marker='x', linestyle='-', color=colors[isub-2], label=f'$s={isub}$')
    inset1_ax0.plot(xc, substages_spurdiv[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    inset1_ax0.tick_params(labelsize=12)
    inset1_ax0.set_xlim(*x1_range)
    inset1_ax0.set_ylim(48, 60)
    mark_inset(ax[0], inset1_ax0, loc1=1, loc2=3, fc="none", ec="0.5")    

    inset1_ax1 = ax[1].inset_axes([0.05, 0.55, 0.2, 0.4])
    inset1_ax1.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
    inset1_ax1.plot(xc, psi_in, linestyle='--', color='gray')
    inset1_ax1.plot(xc, psi_Ex[-1], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    for isub in range(2,nstages-2):
        inset1_ax1.plot(xc, substages[isub], marker='x', linestyle='-', color=colors[isub-2], label=f'$s={isub}$')
    inset1_ax1.plot(xc, substages[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    inset1_ax1.tick_params(labelsize=12)
    inset1_ax1.set_xlim(*x1_range)
    inset1_ax1.set_ylim(48, 60)
    mark_inset(ax[1], inset1_ax1, loc1=1, loc2=3, fc="none", ec="0.5")    

    # Second inset in both plots
    x2_range = (0.68, 0.73)
    inset2_ax0 = ax[0].inset_axes([0.45, 0.2, 0.2, 0.4])
    inset2_ax0.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
    inset2_ax0.plot(xc, psi_in, linestyle='--', color='gray')
    inset2_ax0.plot(xc, psi_Ex[-1], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    for isub in range(2,nstages-2):
        inset2_ax0.plot(xc, substages_spurdiv[isub], marker='x', linestyle='-', color=colors[isub-2], label=f'$s={isub}$')
    inset2_ax0.plot(xc, substages_spurdiv[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    inset2_ax0.tick_params(labelsize=12)
    inset2_ax0.set_xlim(*x2_range)
    inset2_ax0.set_ylim(70, 82)
    mark_inset(ax[0], inset2_ax0, loc1=2, loc2=4, fc="none", ec="0.5")

    inset2_ax1 = ax[1].inset_axes([0.45, 0.2, 0.2, 0.4])
    inset2_ax1.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
    inset2_ax1.plot(xc, psi_in, linestyle='--', color='gray')
    inset2_ax1.plot(xc, psi_Ex[-1], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    for isub in range(2,nstages-2):
        inset2_ax1.plot(xc, substages[isub], marker='x', linestyle='-', color=colors[isub-2], label=f'$s={isub}$')
    inset2_ax1.plot(xc, substages[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    inset2_ax1.tick_params(labelsize=12)
    inset2_ax1.set_xlim(*x2_range)
    inset2_ax1.set_ylim(70, 82)
    mark_inset(ax[1], inset2_ax1, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()
    plt.savefig('figures/substages.pdf', dpi=300)
    plt.close()


def main():
    
    if not os.path.exists('figures'):
        os.makedirs('figures')

    ######## FIGURE: Amplification factor ########
    fig_amplification_factor()
    
    ######## FIGURE: Uniform advection ########
    fig_uniform_advection()

    ######## FIGURE: Order of accuracy ########
    fig_order_of_accuracy()

    ######## FIGURE: Substage fields ########
    fig_substages()


if __name__ == "__main__":
    main()