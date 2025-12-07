



import os
import numpy as np
import amplification_factor as af
import matplotlib.pyplot as plt
import schemes as sch
import analytic as an
import matplotlib as mpl
import string
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def l2norm(field, analytic, dxc):
    """This calculates the l2 norm from an output field compared to the analytic solution."""
    numerator = np.sum(dxc*(field - analytic)*(field - analytic))
    denominator = np.sum(dxc*analytic*analytic)
    return np.sqrt(numerator/(denominator + 1.e-16))


def fig_amplification_factor():
    # Calculate amplification factor and implicitness
    C = np.logspace(-1, 2, 401, endpoint=True)
    theta = np.linspace(0., 1., 401, endpoint=True)
    A = af.calculate_amplification_AdHImEx(C, theta, kdx=np.linspace(0., 2*np.pi, 401, endpoint=True))
    theta_AdHImEx = sch.implicitness(C)

    # Design colorbar
    bounds = np.array([0.,1.000001,5.,10.,50.,100.,500.,1000.])
    cmap = mpl.colormaps['viridis'].resampled(len(bounds))
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (.8, .8, .8, 1.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Produce and save figure
    fig = plt.figure(figsize=(4,3))
    C2D, theta2D = np.meshgrid(C, theta)
    plt.contourf(C2D, theta2D, A, bounds, cmap=cmap, norm=norm)
    plt.plot(C, theta_AdHImEx, 'r--', linewidth=0.7, label=f'AdHImEx $\\theta$')
    plt.colorbar(label='|A|', boundaries = bounds)
    plt.legend()
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel(f'$\\theta$')
    plt.tight_layout()
    fig.savefig('figures/amplification_factor.pdf', dpi=300)
    plt.close(fig)


def fig_uniform_advection():
    dt, nx, xmax = 0.01, 40, 1.
    dx = xmax/nx
    u = [1., 3.125, 6.25, 10.]
    C = [dt*u_i/dx for u_i in u]
    nt = [int(100/u_i) for u_i in u]
    xf = np.linspace(0., xmax, nx, endpoint=False)
    xc = xf + 0.5*dx
    init = an.combi(xc, xmax, u=0., t=0.)
    theta = [sch.implicitness(C_i) for C_i in C]

    fig, ax = plt.subplots(2, 2, figsize=(20,12))
    axrow = 0
    axcol = 0
    for C_i, u_i, nt_i, theta_i in zip(C, u, nt, theta):
        psi_AdHImEx = sch.AdHImEx(init, nt_i, dt, np.full((nt_i,nx), u_i), np.full(nx,dx))
        psi_AdHImEx_FCT1 = sch.AdHImEx(init, nt_i, dt, np.full((nt_i,nx), u_i), np.full(nx,dx), FCTiter=1, FCT_use_previous=C_i<=1.)
        psi_AdHImEx_FCTPD = sch.AdHImEx(init, nt_i, dt, np.full((nt_i,nx), u_i), np.full(nx,dx), FCTiter=1, FCT_min=0., FCT_max=10000.)
        psi_WKS24 = sch.WKS24(init, nt_i, dt, np.full((nt_i,nx), u_i), np.full(nx,dx))
        ax[axrow, axcol].plot(xc, init, color='gray', linestyle='--', label='Initial')
        ax[axrow, axcol].plot(xc, psi_AdHImEx[-1], color='black', linestyle='-', label='AdHImEx', marker='x')
        ax[axrow, axcol].plot(xc, psi_AdHImEx_FCT1[-1], color='blue', linestyle='-', label='AdHImEx FCT')
        ax[axrow, axcol].plot(xc, psi_AdHImEx_FCTPD[-1], color='cyan', linestyle='-', label='AdHImEx FCT PD', marker='x')
        ax[axrow, axcol].plot(xc, psi_WKS24[-1], color='magenta', linestyle='-', label='WKS24')
        ax[axrow, axcol].set_title(f'$C={C_i:.2f}$, $\\theta={theta_i:.2f}$', size=20)
        #ax[axrow, axcol].text(-0.1, 1.1, string.ascii_lowercase[C.index(C_i)], transform=ax[axrow, axcol].transAxes, size=20, weight='bold')        
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
    fig.legend(handles, labels, ncol=5, bbox_to_anchor=(0.5, -0.05), loc='lower center', fontsize=20)
    plt.tight_layout()
    fig.savefig('figures/uniform_advection.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_order_of_accuracy():
    # I need to run the AdHImEx scheme for different C's for three dx/dt combinations, while keeping C constant for each combination. Then also add the order of accuracy lines for 1st, 2nd, and 3rd order.

    xmax = 1.0
    nx = np.array([80, 40, 20], dtype=int)
    dx = xmax/nx
    dt = np.array([0.005, 0.01, 0.02], dtype=float)
    nt = np.array([4, 2, 1])
    u = [0.1, 1., 3.125, 6.25, 10.]
    C = [dt[0]*u_i/dx[0] for u_i in u]
    theta = [sch.implicitness(C_i) for C_i in C]

    fig, ax = plt.subplots(figsize=(4,4))
    colors = ['black', '#810f7c','#8856a7','#8c96c6','#b3cde3']
    for C_i, theta_i, colors_i, u_i in zip(C, theta, colors, u):
        l2_AdHImEx = []
        for dt_i, dx_i, nt_i, nx_i in zip(dt, dx, nt, nx):
            xc = np.linspace(0., xmax, nx_i, endpoint=False)
            init = an.sine(xc, xmax, u=0., t=0.)
            analytic = an.sine(xc, xmax, u=u_i, t=dt_i*nt_i)
            psi_AdHImEx = sch.AdHImEx(init, nt_i, dt_i, np.full((nt_i,nx_i), u_i), np.full(nx_i, dx_i))
            l2_AdHImEx.append(l2norm(psi_AdHImEx[-1], analytic, dx_i))
        ax.plot(dx, l2_AdHImEx, marker='x', label=f'$C={C_i:.2f}$', color=colors_i)
    gridscale = np.logspace(0, np.log10(4), num=3)
    secondorder = 7e-5*gridscale**2#2.*1e-4*0.8*gridscale**2
    thirdorder = 5e-6*gridscale**3#7.*1e-8*0.8*gridscale**3
    fifthorder = 1.9e-10*gridscale**5
    ax.plot(dx, secondorder, color='grey', linestyle=':')
    ax.plot(dx, thirdorder, color='grey', linestyle=':')
    ax.plot(dx, fifthorder, color='grey', linestyle=':')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('$\\Delta x$')
    ax.set_ylabel('$l_2$ norm')
    #plt.tight_layout()
    fig.savefig('figures/order_of_accuracy.pdf', dpi=300)
    plt.close(fig)


def fig_l2_norm_over_C():
    xmax = 1.
    nx = np.logspace(1, 4, 30, endpoint=True, dtype=int)
    dx = xmax/nx
    dt, nt, u = 0.01, 1, 1.
    C = dt*u/dx
    theta = sch.implicitness(C)

    l2_AdHImEx, l2_WKS24 = [], []
    for dx_i, nx_i, C_i, theta_i in zip(dx, nx, C, theta):
        xf = np.linspace(0., xmax, nx_i, endpoint=False)
        xc = xf + 0.5*dx_i
        init = an.sine(xc, xmax, u=0., t=0.)
        analytic = an.sine(xc, xmax, u, t=dt*nt)
        psi_AdHImEx = sch.AdHImEx(init, nt, dt, np.full((nt,nx_i), u), np.full(nx_i, dx_i))
        psi_WKS24 = sch.WKS24(init, nt, dt, np.full((nt,nx_i), u), np.full(nx_i, dx_i))
        l2_AdHImEx.append(l2norm(psi_AdHImEx[-1], analytic, dx_i))
        l2_WKS24.append(l2norm(psi_WKS24[-1], analytic, dx_i))

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


def fig_AdImEx_FCT():
    dt, nx, xmax = 0.01, 40, 1.
    dx = xmax/nx
    u = [3.125, 6.25]#[1., 3.125, 6.25, 10.]
    C = [dt*u_i/dx for u_i in u]
    nt = (100/np.array(u)).astype(int)
    theta = [sch.implicitness(C_i) for C_i in C]
    xf = np.linspace(0., xmax, nx, endpoint=False)
    xc = xf + 0.5*dx
    
    fig, ax = plt.subplots(1, 2, figsize=(20,6.5))#12))
    axcol = 0
    #axcol = 0
    for C_i, theta_i, u_i, nt_i in zip(C, theta, u, nt):
        init = an.combi(xc, xmax, u=0., t=0.)
        psi_AdHImEx = sch.AdHImEx(init, nt_i, dt, np.full((nt_i, nx), u_i), np.full(nx, dx))
        psi_AdHImEx_FCT1 = sch.AdHImEx(init, nt_i, dt, np.full((nt_i, nx), u_i), np.full(nx, dx), FCTiter=1)
        psi_AdHImEx_FCT2 = sch.AdHImEx(init, nt_i, dt, np.full((nt_i, nx), u_i), np.full(nx, dx), FCTiter=2)
        psi_AdHImEx_FCT3 = sch.AdHImEx(init, nt_i, dt, np.full((nt_i, nx), u_i), np.full(nx, dx), FCTiter=3)
        ax[axcol].plot(xc, init, color='gray', linestyle='--', label='Initial')
        ax[axcol].plot(xc, psi_AdHImEx[-1], color='black', linestyle='-', label='AdHImEx')
        ax[axcol].plot(xc, psi_AdHImEx_FCT1[-1], color='blue', linestyle='-', label='AdHImEx FCT')
        ax[axcol].plot(xc, psi_AdHImEx_FCT2[-1], color='cornflowerblue', linestyle='-', label='AdHImEx 2FCT')
        ax[axcol].plot(xc, psi_AdHImEx_FCT3[-1], color='skyblue', linestyle='-', label='AdHImEx 3FCT')
        ax[axcol].text(0.04, 0.9, string.ascii_lowercase[C.index(C_i)], transform=ax[axcol].transAxes, size=20, weight='bold')
        ax[axcol].set_title(f'$C={C_i:.2f}$, $\\theta={theta_i:.2f}$', size=20)
        ax[axcol].tick_params(labelsize=20)
        ax[axcol].set_xlim(0.,1.)
        ax[axcol].set_xlabel('x', size=20)
        ax[axcol].set_ylabel('$\\Psi$', size=20)
        #ax[axcol].legend() # !!! figure out how to make one legend for all subplots
        if axcol == 0:
            handles, labels = ax[axcol].get_legend_handles_labels()
        axcol += 1
        #if axcol == 2:
        #    axcol = 0
        #    axrow += 1
#
    #dt, nx, xmax = 0.01, 40, 1.
    #dx = xmax/nx
    #u = [1., 3.125, 6.25, 10.]
    #C = [dt*u_i/dx for u_i in u]
    #nt = (100/np.array(u)).astype(int)
    #theta = [sch.implicitness(C_i) for C_i in C]
    #xf = np.linspace(0., xmax, nx, endpoint=False)
    #xc = xf + 0.5*dx
#
    #fig, ax = plt.subplots(2, 2, figsize=(20,12))
    #axrow = 0
    #axcol = 0
    #for C_i, theta_i, u_i, nt_i in zip(C, theta, u, nt):
    #    init = an.combi(xc, xmax, u=0., t=0.)
    #    psi_AdHImEx = sch.AdHImEx(init, nt_i, dt, np.full((nt_i, nx), u_i), np.full(nx, dx))
    #    psi_AdHImEx_FCT1 = sch.AdHImEx(init, nt_i, dt, np.full((nt_i, nx), u_i), np.full(nx, dx), FCTiter=1)
    #    psi_AdHImEx_FCT2 = sch.AdHImEx(init, nt_i, dt, np.full((nt_i, nx), u_i), np.full(nx, dx), FCTiter=2)
    #    psi_AdHImEx_FCT3 = sch.AdHImEx(init, nt_i, dt, np.full((nt_i, nx), u_i), np.full(nx, dx), FCTiter=3)
    #    ax[axrow, axcol].plot(xc, init, color='gray', linestyle='--', label='Initial')
    #    ax[axrow, axcol].plot(xc, psi_AdHImEx[-1], color='black', linestyle='-', label='AdHImEx')
    #    ax[axrow, axcol].plot(xc, psi_AdHImEx_FCT1[-1], color='blue', linestyle='-', label='AdHImEx FCT')
    #    ax[axrow, axcol].plot(xc, psi_AdHImEx_FCT2[-1], color='cornflowerblue', linestyle='-', label='AdHImEx 2FCT')
    #    ax[axrow, axcol].plot(xc, psi_AdHImEx_FCT3[-1], color='skyblue', linestyle='-', label='AdHImEx 3FCT')
    #    ax[axrow, axcol].text(0.04, 0.9, string.ascii_lowercase[C.index(C_i)], transform=ax[axrow, axcol].transAxes, size=20, weight='bold')
    #    ax[axrow, axcol].set_title(f'$C={C_i:.2f}$, $\\theta={theta_i:.2f}$')
    #    ax[axrow, axcol].set_xlim(0.,1.)
    #    ax[axrow, axcol].set_xlabel('x')
    #    ax[axrow, axcol].set_ylabel('$\\Psi$')
    #    if axcol == 0 and axrow == 0:
    #        handles, labels = ax[axrow, axcol].get_legend_handles_labels()
    #    axcol += 1
    #    if axcol == 2:
    #        axcol = 0
    #        axrow += 1
    fig.legend(handles, labels, ncol=5, bbox_to_anchor=(0.5, -0.1), loc='lower center', fontsize=20)
    plt.tight_layout()
    fig.savefig('figures/AdImEx_FCT.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_nonuniform_advection():
    # We need to set: dt, nx, nt, u_setting, analytic
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

    psi_in = an.sine_xyshiftampl3(xc, xmax, u=0., t=0.)
    uf = np.zeros((nt_Ex, nx)) # !!! check what happens with AdImEx time steps
    for it in range(nt_Ex):
        uf[it] = an.velocity_varying_space701(xf)
    C_Ex, C_AdImEx = dt_Ex*uf[0]/dx, dt_AdImEx*uf[0]/dx # !!! is this at faces? + decide if uf[0] fine
    theta_Ex = sch.implicitness(C_Ex)
    theta_AdImEx = sch.implicitness(C_AdImEx)

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
    plt.savefig('figures/nonuniform_courant_implicitness.pdf')
    plt.savefig('figures/nonuniform_courant_implicitness.svg')
    plt.close()

    # Run schemes
    psi_Ex = sch.AdHImEx(psi_in, nt_Ex, dt_Ex, uf, dx)
    psi_AdImEx = sch.AdHImEx(psi_in, nt_AdImEx, dt_AdImEx, uf, dx)

    # Plot time steps
    plt.figure(figsize=(10,5))
    plt.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
    plt.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
    plt.plot(xc, psi_in, linestyle='--', color='grey', label='Initial') # plot initial condition
    AdImExcolors = ['#543005', '#bf812d', '#dfc27d',  '#80cdc1', '#01665e']
    for it in [4,8,12,16,20]: # Ex time steps
        if it == 4:
            plt.plot(xc, psi_Ex[it*dtfactor_ExAdImEx], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
        else:
            plt.plot(xc, psi_Ex[it*dtfactor_ExAdImEx], marker='+', linestyle='-', linewidth=0.5, color='silver')
    for it in [4,8,12,16,20]: # AdImEx time steps
        plt.plot(xc, psi_AdImEx[it], marker='', linestyle='-', color=AdImExcolors[(it//4-1)%len(AdImExcolors)], label=f'$n_t = {it}$')
    plt.tick_params(labelsize=15)
    plt.xlim(0.,1.)
    plt.xlabel('x', size=15)
    plt.ylabel('$\\Psi$', size=15)
    plt.legend(fontsize=15)
    plt.savefig('figures/nonuniform_advection.pdf', dpi=300)
    plt.savefig('figures/nonuniform_advection.svg', dpi=300)
    plt.close()


def fig_substages():


    #psi_in = an.sine_xyshiftampl3(xc, xmax, u=0., t=0.)
    #uf = np.zeros((nt_Ex, nx)) # !!! check what happens with AdImEx time steps
    #for it in range(nt_Ex):
    #    uf[it] = an.velocity_varying_space701(xf)
    #C_Ex, C_AdImEx = dt_Ex*uf[0]/dx, dt_AdImEx*uf[0]/dx # !!! is this at faces? + decide if uf[0] fine
    #theta_Ex = sch.implicitness(C_Ex)
    #theta_AdImEx = sch.implicitness(C_AdImEx)
#
    #fig, ax1 = plt.subplots(figsize=(4,3))
    #ax2 = ax1.twinx()
    #ax1.axhline(1, color='k', linestyle=':', linewidth=0.5)
    #line_CEx = ax1.plot(xf, C_Ex, label='Ex $C_f$', color='gray', linestyle='-')
    #line_thetaEx = ax2.plot(xf, theta_Ex, label='Ex $\\theta_f$', color='gray', linestyle='--')
    #line_CAdImEx = ax1.plot(xf, C_AdImEx, label='AdImEx $C_f$', color='k', linestyle='-')            
    #line_thetaAdImEx = ax2.plot(xf, theta_AdImEx, label='AdImEx $\\theta_f$', color='k', linestyle='--')   
    #ax1.set_xlabel('x')
    #ax1.set_ylabel('$C_f$')
    #ax2.set_ylabel('$\\theta_f$')
    ## Create a single legend for both axes
    #lns = line_CEx + line_CAdImEx + line_thetaEx + line_thetaAdImEx
    #labs = [l.get_label() for l in lns]
    #ax1.legend(lns, labs, loc='best')
    #fig.tight_layout()
    #plt.savefig('figures/nonuniform_courant_implicitness.pdf')
    #plt.close()
#
    ## Run schemes
    #psi_Ex = sch.AdHImEx(psi_in, nt_Ex, dt_Ex, uf, dx)
    #psi_AdImEx = sch.AdHImEx(psi_in, nt_AdImEx, dt_AdImEx, uf, dx)
#
    ## Plot time steps
    #plt.figure(figsize=(10,5))
    #plt.plot(xc, psi_in, linestyle='--', color='grey', label='Initial') # plot initial condition
    #AdImExcolors = ['#543005', '#bf812d', '#dfc27d',  '#80cdc1', '#01665e']
    #for it in [4,8,12,16,20]: # Ex time steps
    #    if it == 4:
    #        plt.plot(xc, psi_Ex[it*dtfactor_ExAdImEx], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    #    else:
    #        plt.plot(xc, psi_Ex[it*dtfactor_ExAdImEx], marker='+', linestyle='-', linewidth=0.5, color='silver')
    #for it in [4,8,12,16,20]: # AdImEx time steps
    #    plt.plot(xc, psi_AdImEx[it], marker='', linestyle='-', color=AdImExcolors[(it//4-1)%len(AdImExcolors)], label=f'$n_t = {it}$')
    #plt.xlabel('x')
    #plt.ylabel('$\\Psi$')
    #plt.legend()
    #plt.savefig('figures/nonuniform_advection.pdf', dpi=300)
    #plt.close()




    ############## !!! 29-09-2025: I think the difference in Ex and AdImEx for nx=50 and dx=0.02 is because of the temporal order of convergence of the implicit scheme?? For same Courant number but higher resolution (nx=250 and dx=0.004) the difference is much smaller. So this is not a problem with the AdHImEx scheme, it is just that you want to use a certain resolution to get good results.  # maybe use nx=100 and dx=0.01 as middle ground? but slightly clogged but hey...
    xmax = 1.0
    nx = 250#100#50#100#250#50#250#50
    dt_AdImEx = 0.004#0.01#0.02#0.01#0.004#0.02#0.01#0.02#0.004#0.02
    nt_AdImEx = 1#5#1#2#1#2#1#3#5#1
    dtfactor_ExAdImEx = 20
    dt_Ex = dt_AdImEx/dtfactor_ExAdImEx
    nt_Ex = nt_AdImEx*dtfactor_ExAdImEx
    xf = np.linspace(0., xmax, nx, endpoint=False)
    dx = np.full(nx, xf[1] - xf[0])
    xc = xf + 0.5*dx

    #print(nt_AdImEx, dt_AdImEx, nt_Ex, dt_Ex)
    #print(nt_AdImEx*dt_AdImEx, nt_Ex*dt_Ex)
    

    psi_in = an.sine_xyshiftampl3(xc, xmax, u=0., t=0.)
    uf = np.zeros((nt_Ex, nx))
    for it in range(nt_Ex):
        uf[it] = an.velocity_varying_space701(xf)
    C_Ex = dt_Ex*uf[0]/dx # !!! is this at faces?
    C_AdImEx = dt_AdImEx*uf[0]/dx # !!! is this at faces?
    theta_Ex = sch.implicitness(C_Ex)
    theta_AdImEx = sch.implicitness(C_AdImEx)
    #plt.plot(theta_AdImEx)
    #plt.show()
    
    # !!! remove in final version
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

    ## Run schemes
    #nstages = 7    
    #substages = np.zeros((nstages,nx))
    #psi_AdHImEx = sch.AdHImEx(psi_in, nt_AdImEx, dt_AdImEx, uf, dx, output_substages=True, substages=substages)
    #psi_Ex = sch.AdHImEx(psi_in, nt_Ex, dt_Ex, uf, dx)
    ## Plot substages fields
    #plt.figure(figsize=(10,5))
    #plt.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
    #plt.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
    #plt.plot(xc, psi_in, linestyle='--', color='gray', label='Initial')
    #plt.plot(xc, psi_Ex[-1], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    #for isub in range(2,nstages-2):
    #    plt.plot(xc, substages[isub], marker='x', linestyle='-', color=plt.cm.viridis(isub*1.5/nstages), label=f'$k={isub}$')
    #plt.plot(xc, substages[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    #plt.xlim(0.,1.)
    #plt.xlabel('x')
    #plt.ylabel('$\\Psi$')
    #plt.legend()
    #plt.savefig('figures/substages.pdf', dpi=300)
    #plt.close()
#
    ## Plot zoomed in substages
    #plt.figure(figsize=(5,5))
    #plt.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
    ##plt.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
    #plt.plot(xc[65:85], psi_in[65:85], linestyle='--', color='gray', label='Initial')
    #plt.plot(xc[65:85], psi_Ex[-1][65:85], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    #for isub in range(2,nstages-2):
    #    plt.plot(xc[65:85], substages[isub][65:85], marker='x', linestyle='-', color=plt.cm.viridis(isub*1.5/nstages), label=f'$k={isub}$')
    #plt.plot(xc[65:85], substages[nstages-1][65:85], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    ##plt.xlim(0.,1.)
    #plt.xlabel('x')
    #plt.ylabel('$\\Psi$')
    #plt.legend()
    #plt.savefig('figures/substages_leftborder.pdf', dpi=300)
    #plt.close()
#
    ## Plot zoomed in substages
    #plt.figure(figsize=(5,5))
    ##plt.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
    #plt.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
    #plt.plot(xc[165:185], psi_in[165:185], linestyle='--', color='gray', label='Initial')
    #plt.plot(xc[165:185], psi_Ex[-1][165:185], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    #for isub in range(2,nstages-2):
    #    plt.plot(xc[165:185], substages[isub][165:185], marker='x', linestyle='-', color=plt.cm.viridis(isub*1.5/nstages), label=f'$k={isub}$')
    #plt.plot(xc[165:185], substages[nstages-1][165:185], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    ##plt.xlim(0.,1.)
    #plt.xlabel('x')
    #plt.ylabel('$\\Psi$')
    #plt.legend()
    #plt.savefig('figures/substages_rightborder.pdf', dpi=300)
    #plt.close()


    # Run schemes and plot all in one plot
    nstages = 7    
    substages = np.zeros((nstages,nx))
    psi_AdHImEx = sch.AdHImEx(psi_in, nt_AdImEx, dt_AdImEx, uf, dx, output_substages=True, substages=substages)
    psi_Ex = sch.AdHImEx(psi_in, nt_Ex, dt_Ex, uf, dx)
    # Plot substages fields
    fig, ax = plt.subplots(figsize=(10,5))
    ax.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
    ax.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
    ax.plot(xc, psi_in, linestyle='--', color='gray', label='Initial')
    ax.plot(xc, psi_Ex[-1], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    for isub in range(2,nstages-2):
        ax.plot(xc, substages[isub], marker='x', linestyle='-', color=plt.cm.viridis(isub*1.5/nstages), label=f'$k={isub}$')
    ax.plot(xc, substages[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    ax.tick_params(labelsize=15)
    ax.set_xlim(0.,1.)
    ax.set_xlabel('x', size=15)
    ax.set_ylabel('$\\Psi$', size=15)
    ax.legend(fontsize=15)

    x1_range = (0.28, 0.33)
    inset1 = ax.inset_axes([0.05, 0.55, 0.2, 0.4])
    inset1.axvline(0.3, color='k', linestyle=':', linewidth=0.7)
    inset1.plot(xc, psi_in, linestyle='--', color='gray')
    inset1.plot(xc, psi_Ex[-1], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    for isub in range(2,nstages-2):
        inset1.plot(xc, substages[isub], marker='x', linestyle='-', color=plt.cm.viridis(isub*1.5/nstages), label=f'$k={isub}$')
    inset1.plot(xc, substages[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    inset1.tick_params(labelsize=12)
    inset1.set_xlim(*x1_range)
    inset1.set_ylim(48, 60)
    #inset1.tick_params(labelsize=8)
    mark_inset(ax, inset1, loc1=1, loc2=3, fc="none", ec="0.5")    
    
    x2_range = (0.68, 0.73)
    inset2 = ax.inset_axes([0.45, 0.2, 0.2, 0.4])
    inset2.axvline(0.7, color='k', linestyle=':', linewidth=0.7)
    inset2.plot(xc, psi_in, linestyle='--', color='gray')
    inset2.plot(xc, psi_Ex[-1], marker='+', linestyle='-', linewidth=0.5, color='silver', label='Ex')
    for isub in range(2,nstages-2):
        inset2.plot(xc, substages[isub], marker='x', linestyle='-', color=plt.cm.viridis(isub*1.5/nstages), label=f'$k={isub}$')
    inset2.plot(xc, substages[nstages-1], color='k', marker='', linestyle='-', label=f'$n_t=1$')
    inset2.tick_params(labelsize=12)
    inset2.set_xlim(*x2_range)
    inset2.set_ylim(70, 82)
    #inset2.tick_params(labelsize=8)
    mark_inset(ax, inset2, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.savefig('figures/substages.pdf', dpi=300)
    plt.close()



def fig_nonuniform_advection_swift():

    # Set up file to store l2 norms
    
    dt, nx, xmax = 2., 64, 1000. #128, 1000.
    dx = xmax/nx
    nt = 50
    xf = np.linspace(-0.5*xmax, 0.5*xmax, nx, endpoint=False)
    xc = xf + 0.5*dx
    uf = an.velocity_varying_time_space_swift_2Dnondiv(nt, dt, xf)
    init = an.sine_swift(xc, xmax)
    
    # Plot the final C and theta fields
    C = dt*uf[-1]/dx # !!! is this at faces?
    theta = sch.implicitness(C)
    fig, ax1 = plt.subplots(figsize=(4,3))
    ax2 = ax1.twinx()
    ax1.axhline(1, color='k', linestyle=':', linewidth=0.7)
    line_C = ax1.plot(xf, C, label='$C_f$', color='k', linestyle='-')
    line_theta = ax2.plot(xf, theta, label='$\\theta_f$', color='k', linestyle='--')
    ax1.set_xlabel('x')
    ax1.set_ylabel('$C_f$')
    ax2.set_ylabel('$\\theta_f$')
    # Create a single legend for both axes
    lns = line_C + line_theta
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    fig.tight_layout()
    plt.savefig('figures/nonuniform_swift_courant_implicitness.pdf')
    plt.close()

    # Run schemes
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    psi_AdHImEx = sch.AdHImEx(init, nt, dt, uf, np.full(nx,dx))
    psi_AdHImEx_FCT1 = sch.AdHImEx(init, nt, dt, np.full((nt,nx), uf), np.full(nx,dx), FCTiter=1)
    psi_AdHImEx_FCTPD = sch.AdHImEx(init, nt, dt, np.full((nt,nx), uf), np.full(nx,dx), FCTiter=1, FCT_min=0., FCT_max=10000.)
    #psi_AdHImEx_FCT2 = sch.AdHImEx(init, nt, dt, np.full((nt,nx), uf), np.full(nx,dx), FCTiter=2)
    #psi_AdHImEx_FCT3 = sch.AdHImEx(init, nt, dt, np.full((nt,nx), uf), np.full(nx,dx), FCTiter=3)
    psi_WKS24 = sch.WKS24(init, nt, dt, np.full((nt,nx), uf), np.full(nx,dx))

    # Calculate l2 norms
    with open('swift_l2norms.out', 'w') as f:
        f.write('l2_AdHImEx, l2_AdHImEx_FCT1, l2_AdHImEx_FCTPD, l2_WKS24\n')
        #for it in range(nt):
            #analytic = an.sine_swift(xc - np.sum(uf[:it+1], axis=0)*dt, xmax)
        l2_AdHImEx = l2norm(psi_AdHImEx[-1], init, dx)
        l2_AdHImEx_FCT1 = l2norm(psi_AdHImEx_FCT1[-1], init, dx)
        l2_AdHImEx_FCTPD = l2norm(psi_AdHImEx_FCTPD[-1], init, dx)
        #l2_AdHImEx_FCT2 = l2norm(psi_AdHImEx_FCT2[-1], analytic, dx)
        #l2_AdHImEx_FCT3 = l2norm(psi_AdHImEx_FCT3[-1], analytic, dx)
        l2_WKS24 = l2norm(psi_WKS24[-1], init, dx)
        f.write(f'{l2_AdHImEx:.6e} {l2_AdHImEx_FCT1:.6e} {l2_AdHImEx_FCTPD:.6e} {l2_WKS24:.6e}\n')
    
    # Plot results
    AdImExcolors = ['#543005', '#bf812d', 'orange', 'orangered']
    ax.plot(xc, init, color='gray', linestyle='--', marker='x', label='Initial')
    for it in [10, 20, 30, 40, 50]:
        if it == 50:
            ax.plot(xc, psi_AdHImEx[it], color='k', linestyle='-', label=f'AdHImEx $n_t = {it}$', marker='+')
            ax.plot(xc, psi_AdHImEx_FCT1[it], color='blue', linestyle='-', label=f'AdHImEx FCT $n_t = {it}$')
            ax.plot(xc, psi_AdHImEx_FCTPD[it], color='cyan', linestyle='-', label=f'AdHImEx FCT PD $n_t = {it}$')
            #ax.plot(xc, psi_AdHImEx_FCT2[it], color='cornflowerblue', linestyle='-', label=f'AdHImEx 2FCT $n_t = {it}$')
            #ax.plot(xc, psi_AdHImEx_FCT3[it], color='skyblue', linestyle='-', label=f'AdHImEx 3FCT $n_t = {it}$')
            ax.plot(xc, psi_WKS24[it], color='magenta', linestyle='-', label=f'WKS24 $n_t = {it}$')
        else: 
            ax.plot(xc, psi_AdHImEx[it], color=AdImExcolors[(it-1)//10], linestyle=':', label=f'AdHImEx $n_t = {it}$', linewidth=0.9)
    ax.tick_params(labelsize=15)
    ax.set_xlabel('x', size=15)
    ax.set_ylabel('$\\Psi$', size=15)
    ax.legend(fontsize=15)
    fig.savefig('figures/nonuniform_advection_swift.pdf', dpi=300)
    plt.close(fig)


def main():

    # !!! change fontsize everywhere
    #     
    if not os.path.exists('figures'):
        os.makedirs('figures')

    print('Producing figures...')

    ######## FIGURE: Amplification factor ########
    #fig_amplification_factor()
    
    ######## FIGURE: Uniform advection ########
    #fig_uniform_advection()

    ######## FIGURE: Order of accuracy ########
    #fig_order_of_accuracy()

    ######## FIGURE: l2 norm over C ########
    #fig_l2_norm_over_C()

    ######## FIGURE: AdImEx FCT ########
    fig_AdImEx_FCT()

    ######## FIGURE: Nonuniform advection ########
    #fig_nonuniform_advection()

    ######## FIGURE: Substage fields ########
    #fig_substages()

    ######## FIGURE: Nonuniform advection SWIFT testcase ########
    #fig_nonuniform_advection_swift()


    print('...done')


if __name__ == "__main__":
    main()