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
    dt, nx, xmax = 0.01, 40, 1.
    dx = xmax/nx
    u = [1., 3.125, 6.25, 10.]
    C = [dt*u_i/dx for u_i in u]
    nt = [int(100/u_i) for u_i in u]
    xf = np.linspace(0., xmax, nx, endpoint=False)
    xc = xf + 0.5*dx
    init = an.combi(xc, xmax, u=0., t=0.)
    theta = [sch.implicitness(C_i) for C_i in C]

    # Run schemes and plot
    fig, ax = plt.subplots(2, 2, figsize=(20,12))
    axrow = 0
    axcol = 0
    for C_i, u_i, nt_i, theta_i in zip(C, u, nt, theta):
        psi_AdHImEx = sch.AdHImEx(init, nt_i, dt, np.full((nt_i,nx), u_i), np.full(nx,dx))
        psi_AdHImEx_FCT1 = sch.AdHImEx(init, nt_i, dt, np.full((nt_i,nx), u_i), np.full(nx,dx), FCT=True, nondivergent=True)
        psi_AdHImEx_FCTNN = sch.AdHImEx(init, nt_i, dt, np.full((nt_i,nx), u_i), np.full(nx,dx), FCT=True, ymin=0., ymax=10000., nondivergent=True)
        psi_WKS24 = sch.WKS24(init, nt_i, dt, np.full((nt_i,nx), u_i), np.full(nx,dx))
        ax[axrow, axcol].plot(xc, init, color='gray', linestyle='--', label='Initial')
        ax[axrow, axcol].plot(xc, psi_AdHImEx[-1], color='black', linestyle='-', label='AdHImEx', marker='x')
        ax[axrow, axcol].plot(xc, psi_AdHImEx_FCT1[-1], color='blue', linestyle='-', label='AdHImEx FCT')
        ax[axrow, axcol].plot(xc, psi_AdHImEx_FCTNN[-1], color='cyan', linestyle='-', label='AdHImEx FCT NN', marker='x')
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
    nx = np.array([80, 40, 20], dtype=int)
    dx = xmax/nx
    dt = np.array([0.005, 0.01, 0.02], dtype=float)
    nt = np.array([4, 2, 1])
    u = [0.1, 1., 3.125, 6.25, 10.]
    C = [dt[0]*u_i/dx[0] for u_i in u]
    theta = [sch.implicitness(C_i) for C_i in C]

    # Run schemes and plot
    fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True)
    colors = ['black', '#810f7c','#8856a7','#8c96c6','#b3cde3']
    for C_i, theta_i, colors_i, u_i in zip(C, theta, colors, u):
        l2_WKS24, l2_AdHImEx = [], []
        for dt_i, dx_i, nt_i, nx_i in zip(dt, dx, nt, nx):
            xc = np.linspace(0., xmax, nx_i, endpoint=False)
            init = an.sine(xc, xmax, u=0., t=0.)
            analytic = an.sine(xc, xmax, u=u_i, t=dt_i*nt_i)
            psi_WKS24 = sch.WKS24(init, nt_i, dt_i, np.full((nt_i,nx_i), u_i), np.full(nx_i, dx_i))
            psi_AdHImEx = sch.AdHImEx(init, nt_i, dt_i, np.full((nt_i,nx_i), u_i), np.full(nx_i, dx_i))
            l2_WKS24.append(an.l2norm(psi_WKS24[-1], analytic, dx_i))
            l2_AdHImEx.append(an.l2norm(psi_AdHImEx[-1], analytic, dx_i))
        ax[0].plot(dx, l2_WKS24, marker='x', label=f'$C={C_i:.2f}$', color=colors_i)
        ax[1].plot(dx, l2_AdHImEx, marker='x', label=f'$C={C_i:.2f}$', color=colors_i)

    # Plotting details
    gridscale = np.logspace(0, np.log10(4), num=3)
    firstorder = 2.5e-2*gridscale
    secondorder = 7e-5*gridscale**2
    thirdorder = 5e-6*gridscale**3
    fifthorder = 1.9e-10*gridscale**5
    for a in ax:
        a.plot(dx, firstorder, color='grey', linestyle=':')
        a.plot(dx, secondorder, color='grey', linestyle=':')
        a.plot(dx, thirdorder, color='grey', linestyle=':')
        a.plot(dx, fifthorder, color='grey', linestyle=':')
        a.set_yscale('log')
        a.set_xscale('log')
        a.set_xlabel('$\\Delta x$')
    ax[0].legend(loc='lower right')
    ax[0].set_ylabel('$l_2$ norm')
    ax[0].set_title('WKS24')
    ax[1].set_title('AdHImEx')
    fig.savefig('figures/order_of_accuracy.pdf', dpi=300)
    plt.close(fig)


def fig_l2_norm_over_C():

    # Setup 
    xmax = 1.
    nx = np.logspace(1, 4, 30, endpoint=True, dtype=int)
    dx = xmax/nx
    dt, nt, u = 0.01, 1, 1.
    C = dt*u/dx
    theta = sch.implicitness(C)

    # Run schemes
    l2_AdHImEx, l2_WKS24 = [], []
    for dx_i, nx_i, C_i, theta_i in zip(dx, nx, C, theta):
        xf = np.linspace(0., xmax, nx_i, endpoint=False)
        xc = xf + 0.5*dx_i
        init = an.sine(xc, xmax, u=0., t=0.)
        analytic = an.sine(xc, xmax, u, t=dt*nt)
        psi_AdHImEx = sch.AdHImEx(init, nt, dt, np.full((nt,nx_i), u), np.full(nx_i, dx_i))
        psi_WKS24 = sch.WKS24(init, nt, dt, np.full((nt,nx_i), u), np.full(nx_i, dx_i))
        l2_AdHImEx.append(an.l2norm(psi_AdHImEx[-1], analytic, dx_i))
        l2_WKS24.append(an.l2norm(psi_WKS24[-1], analytic, dx_i))

    # Plotting details
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(C, l2_AdHImEx, marker='x', label='AdHImEx', color='k')
    ax.plot(C, l2_WKS24, marker='x', label='WKS24', color='magenta')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$C$')
    ax.set_ylabel('$l_2$ norm')
    ax.legend()
    plt.tight_layout()
    fig.savefig('figures/l2_norm_over_C.pdf', dpi=300)
    plt.close(fig)


def fig_nonuniform_advection():

    # Setup
    xmax = 1.0
    dt_AdImEx = 0.004
    nt_AdImEx = 20
    nx = 250
    dtfactor_ExAdImEx = 20
    dt_Ex = dt_AdImEx/dtfactor_ExAdImEx
    nt_Ex = nt_AdImEx*dtfactor_ExAdImEx
    xf = np.linspace(0., xmax, nx, endpoint=False)
    dx = np.full(nx,xf[1] - xf[0])
    xc = xf + 0.5*dx
    psi_in = an.sine(xc, xmax, u=0., t=0., shifty=50., ampl=50., shiftx=0.3)
    uf = np.zeros((nt_Ex, nx))
    for it in range(nt_Ex):
        uf[it] = an.velocity_varying_space(xf)
    C_Ex, C_AdImEx = dt_Ex*uf[0]/dx, dt_AdImEx*uf[0]/dx
    theta_Ex = sch.implicitness(C_Ex)
    theta_AdImEx = sch.implicitness(C_AdImEx)

    # Plot courant and implicitness
    fig, ax1 = plt.subplots(figsize=(4,3))
    ax2 = ax1.twinx()
    ax1.axhline(1, color='k', linestyle=':', linewidth=0.7)
    ax1.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
    ax1.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
    line_CEx = ax1.plot(xf, C_Ex, label='Ex $C$', color='gray', linestyle='-')
    line_thetaEx = ax2.plot(xf, theta_Ex, label='Ex $\\theta$', color='gray', linestyle='--')
    line_CAdImEx = ax1.plot(xf, C_AdImEx, label='AdImEx $C$', color='k', linestyle='-')            
    line_thetaAdImEx = ax2.plot(xf, theta_AdImEx, label='AdImEx $\\theta$', color='k', linestyle='--')   
    ax1.set_xlim(0.,1.)
    ax1.set_xlabel('x')
    ax1.set_ylabel('$C$')
    ax2.set_ylabel('$\\theta$')
    # Create a single legend for both axes
    lns = line_CEx + line_CAdImEx + line_thetaEx + line_thetaAdImEx
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    fig.tight_layout()
    plt.savefig('figures/nonuniform_courant_implicitness.pdf')
    plt.close()

    # Run schemes
    psi_Ex = sch.AdHImEx(psi_in, nt_Ex, dt_Ex, uf, dx)
    psi_AdImEx = sch.AdHImEx(psi_in, nt_AdImEx, dt_AdImEx, uf, dx)

    # Plot time steps
    plt.figure(figsize=(10,5))
    plt.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
    plt.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
    plt.plot(xc, psi_in, linestyle='--', color='grey', label='Initial')
    AdImExcolors = ['#543005', '#bf812d', '#dfc27d',  '#80cdc1', '#01665e']
    for it in [4,8,12,16,20]:
        if it == 4:
            plt.plot(xc, psi_Ex[it*dtfactor_ExAdImEx], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
        else:
            plt.plot(xc, psi_Ex[it*dtfactor_ExAdImEx], marker='+', linestyle='-', linewidth=0.5, color='silver')
    for it in [4,8,12,16,20]:
        plt.plot(xc, psi_AdImEx[it], marker='', linestyle='-', color=AdImExcolors[(it//4-1)%len(AdImExcolors)], label=f'$n_t = {it}$')
    plt.tick_params(labelsize=15)
    plt.xlim(0.,1.)
    plt.xlabel('x', size=15)
    plt.ylabel('$\\Psi$', size=15)
    plt.legend(fontsize=15)
    plt.savefig('figures/nonuniform_advection.pdf', dpi=300)
    plt.close()


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
    C_Ex = dt_Ex*uf[0]/dx
    C_AdImEx = dt_AdImEx*uf[0]/dx 
    theta_Ex = sch.implicitness(C_Ex)
    theta_AdImEx = sch.implicitness(C_AdImEx)

    # Plot courant and implicitness
    fig, ax1 = plt.subplots(figsize=(4,3))
    ax2 = ax1.twinx()
    ax1.axhline(1, color='k', linestyle=':', linewidth=0.7)
    ax1.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
    ax1.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
    line_CEx = ax1.plot(xf, C_Ex, label='Ex $C_f$', color='gray', linestyle='-')
    line_thetaEx = ax2.plot(xf, theta_Ex, label='Ex $\\theta_f$', color='gray', linestyle='--')
    line_CAdImEx = ax1.plot(xf, C_AdImEx, label='AdImEx $C_f$', color='k', linestyle='-')
    line_thetaAdImEx = ax2.plot(xf, theta_AdImEx, label='AdImEx $\\theta_f$', color='k', linestyle='--')
    ax1.set_xlim(0.,1.)
    ax1.set_xlabel('x')
    ax1.set_ylabel('$C_f$')
    ax2.set_ylabel('$\\theta_f$')
    # Create a single legend for both axes
    lns = line_CEx + line_CAdImEx + line_thetaEx + line_thetaAdImEx
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    fig.tight_layout()
    plt.savefig('figures/substages_courant_implicitness.pdf')
    plt.close()

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
    fig, ax = plt.subplots(1, 2, figsize=(20,6), sharey=True)
    colors = ['blue','olive','mediumaquamarine']
    for a in ax:
        a.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
        a.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
        a.plot(xc, psi_in, linestyle='--', color='gray', label='Initial')
        a.plot(xc, psi_Ex[-1], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    for isub in range(2,nstages-2):
        ax[0].plot(xc, substages_spurdiv[isub], marker='x', linestyle='-', color=colors[isub-2], label=f'$k={isub}$')
        ax[1].plot(xc, substages[isub], marker='x', linestyle='-', color=colors[isub-2], label=f'$k={isub}$')
    ax[0].plot(xc, substages_spurdiv[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    ax[1].plot(xc, substages[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    for a in ax:
        a.tick_params(labelsize=15)
        a.set_xlim(0.,1.)
        a.set_xlabel('x', size=15)
    ax[0].set_ylabel('$\\Psi$', size=15)
    ax[0].legend(fontsize=15)

    # First inset in both plots
    x1_range = (0.28, 0.33)
    inset1_ax0 = ax[0].inset_axes([0.05, 0.55, 0.2, 0.4])
    inset1_ax0.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
    inset1_ax0.plot(xc, psi_in, linestyle='--', color='gray')
    inset1_ax0.plot(xc, psi_Ex[-1], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    for isub in range(2,nstages-2):
        inset1_ax0.plot(xc, substages_spurdiv[isub], marker='x', linestyle='-', color=colors[isub-2], label=f'$k={isub}$')
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
        inset1_ax1.plot(xc, substages[isub], marker='x', linestyle='-', color=colors[isub-2], label=f'$k={isub}$')
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
        inset2_ax0.plot(xc, substages_spurdiv[isub], marker='x', linestyle='-', color=colors[isub-2], label=f'$k={isub}$')
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
        inset2_ax1.plot(xc, substages[isub], marker='x', linestyle='-', color=colors[isub-2], label=f'$k={isub}$')
    inset2_ax1.plot(xc, substages[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    inset2_ax1.tick_params(labelsize=12)
    inset2_ax1.set_xlim(*x2_range)
    inset2_ax1.set_ylim(70, 82)
    mark_inset(ax[1], inset2_ax1, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()
    plt.savefig('figures/substages.pdf', dpi=300)
    plt.close()



def fig_nonuniform_advection_swift():

    # Setup
    dt, nx, xmax = 2., 64, 1000.
    dx = xmax/nx
    nt = 50
    xf = np.linspace(-0.5*xmax, 0.5*xmax, nx, endpoint=False)
    xc = xf + 0.5*dx
    uf = an.velocity_varying_time_space_swift(nt, dt, xf)
    ufEx = an.velocity_varying_time_space_swift(nt*5, dt*0.2, xf)
    init = an.sine_swift(xc, xmax)
    
    # Plot the final C and theta fields
    C = dt*uf[-1]/dx
    theta = sch.implicitness(C)
    fig, ax1 = plt.subplots(figsize=(4,3))
    ax2 = ax1.twinx()
    ax1.axhline(1, color='k', linestyle=':', linewidth=0.7)
    line_C = ax1.plot(xf, C, label='$C$', color='k', linestyle='-')
    line_theta = ax2.plot(xf, theta, label='$\\theta$', color='k', linestyle='--')
    ax1.set_xlabel('x')
    ax1.set_ylabel('$C$')
    ax2.set_ylabel('$\\theta$')

    # Create a single legend for both axes
    lns = line_C + line_theta
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    fig.tight_layout()
    plt.savefig('figures/nonuniform_swift_courant_implicitness.pdf')
    plt.close()

    # Run schemes
    psi_Ex = sch.AdHImEx(init, nt*5, dt*0.2, ufEx, np.full(nx,dx))
    psi_AdHImEx = sch.AdHImEx(init, nt, dt, uf, np.full(nx,dx))
    psi_AdHImEx_FCT1 = sch.AdHImEx(init, nt, dt, np.full((nt,nx), uf), np.full(nx,dx), FCT=True)
    psi_AdHImEx_FCTNN = sch.AdHImEx(init, nt, dt, np.full((nt,nx), uf), np.full(nx,dx), FCT=True, ymin=0., ymax=10000.)
    psi_WKS24 = sch.WKS24(init, nt, dt, np.full((nt,nx), uf), np.full(nx,dx))

    # Calculate l2 norms
    l2_Ex = an.l2norm(psi_Ex[-1], init, dx)
    l2_AdHImEx = an.l2norm(psi_AdHImEx[-1], init, dx)
    l2_AdHImEx_FCT1 = an.l2norm(psi_AdHImEx_FCT1[-1], init, dx)
    l2_AdHImEx_FCTNN = an.l2norm(psi_AdHImEx_FCTNN[-1], init, dx)
    l2_WKS24 = an.l2norm(psi_WKS24[-1], init, dx)

    # Output l2 norms to file
    with open('swift_l2norms.out', 'w') as f:
        f.write('l2_Ex, l2_AdHImEx, l2_AdHImEx_FCT1, l2_AdHImEx_FCTNN, l2_WKS24\n')
        f.write(f'{l2_Ex:.6e} {l2_AdHImEx:.6e} {l2_AdHImEx_FCT1:.6e} {l2_AdHImEx_FCTNN:.6e} {l2_WKS24:.6e}\n')

    # Plot results
    fig, ax = plt.subplots(1,2, figsize=(13,5), width_ratios=[4, 1])
    AdImExcolors = ['#543005', '#bf812d', 'orange', 'orangered']
    ax[0].plot(xc, init, color='gray', linestyle='--', marker='x', label='Initial')
    ax[0].plot(xc, psi_Ex[nt*5], color='green', linestyle='-', label=f'Ex $n_t = {nt*5}$', marker='+')
    for it in [10, 20, 30, 40, 50]:
        if it == 50:
            ax[0].plot(xc, psi_AdHImEx[it], color='k', linestyle='-', label=f'AdHImEx $n_t = {it}$', marker='+')
            ax[0].plot(xc, psi_AdHImEx_FCT1[it], color='blue', linestyle='-', label=f'AdHImEx FCT $n_t = {it}$')
            ax[0].plot(xc, psi_AdHImEx_FCTNN[it], color='cyan', linestyle='-', label=f'AdHImEx FCT NN $n_t = {it}$')
            ax[0].plot(xc, psi_WKS24[it], color='magenta', linestyle='-', label=f'WKS24 $n_t = {it}$')
        else: 
            ax[0].plot(xc, psi_AdHImEx[it], color=AdImExcolors[(it-1)//10], linestyle=':', label=f'AdHImEx $n_t = {it}$', linewidth=0.9)
    ax[0].tick_params(labelsize=15)
    ax[0].set_xlabel('x', size=15)
    ax[0].set_ylabel('$\\Psi$', size=15)
    ax[0].legend(fontsize=15)

    # Plot l2 norms
    l2norms = [l2_Ex, l2_AdHImEx, l2_AdHImEx_FCT1, l2_AdHImEx_FCTNN, l2_WKS24]
    ax[1].bar(np.arange(len(l2norms)), l2norms, color=['green', 'k', 'blue', 'cyan', 'magenta'])
    ax[1].tick_params(size=15)
    ax[1].set_yscale('log')
    ax[1].set_ylabel('$l_2$ norm', size=15)
    ax[1].yaxis.tick_right()               
    ax[1].yaxis.set_label_position("right")
    ax[1].set_xticks([])
    plt.tight_layout()
    fig.savefig('figures/nonuniform_advection_swift.pdf', dpi=300)
    plt.close(fig)


def main():
    
    if not os.path.exists('figures'):
        os.makedirs('figures')

    print('Producing figures...')

    ######## FIGURE: Amplification factor ########
    fig_amplification_factor()
    
    ######## FIGURE: Uniform advection ########
    fig_uniform_advection()

    ######## FIGURE: Order of accuracy ########
    fig_order_of_accuracy()

    ######## FIGURE: l2 norm over C ########
    fig_l2_norm_over_C()

    ######## FIGURE: Nonuniform advection ########
    fig_nonuniform_advection()

    ######## FIGURE: Substage fields ########
    fig_substages()

    ######## FIGURE: Nonuniform advection SWIFT testcase ########
    fig_nonuniform_advection_swift()

    print('...done')


if __name__ == "__main__":
    main()