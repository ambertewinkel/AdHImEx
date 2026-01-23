import numpy as np
import matplotlib.pyplot as plt

def set_extrema(nx, uf, field_bounded, field_previous, use_previous=False, only_global=False, ymin=None, ymax=None):
    """This function returns the min and max values allowed for each grid cell, defined at center. Used in iterFCT."""

    fieldmin, fieldmax = np.zeros(nx), np.zeros(nx)
    if not only_global:
        if use_previous: # 01-07-2025: previous[i] is at i-1/2 so for cell i we check whether at least one of previous[i] or previous[i+1] is True (i.e. whether C is smaller than 1 on both sides) but what about u velocity? 
            for i in range(nx): #!!!
                #if uf[i] > 0.: # upwind value can be taken for max # !!! discuss 2 or 3 values with HW
                fieldmax[i] = max([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx], field_previous[i-1], field_previous[i], field_previous[(i+1)%nx]])
                fieldmin[i] = min([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx], field_previous[i-1], field_previous[i], field_previous[(i+1)%nx]])
                #elif uf[i] < 0.:
                #    fieldmax[i] = max([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx], field_previous[i], field_previous[(i+1)%nx]])
                #    fieldmin[i] = min([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx],  field_previous[i], field_previous[(i+1)%nx]])
                #else: 
                #    fieldmax[i] = max([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx], field_previous[i]])
                #    fieldmin[i] = min([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx],  field_previous[i]])
        else:
            for i in range(nx):
                fieldmax[i] = max([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx]]) # To avoid monotonic artefacts I would need to remove i+1 for max and min to avoid artefacts in the solution (came up with the SWIFT testcase). However, decided not to do so for the paper in the end.
                fieldmin[i] = min([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx]]) 
    # For global allowable min/max
    if ymin is not None:
        fieldmin = np.full(nx, ymin)#np.where(fieldmin < ymin, ymin, fieldmin)
    if ymax is not None:
        fieldmax = np.full(nx, ymax)#np.where(fieldmax > ymax, ymax, fieldmax)

    return fieldmin, fieldmax


def iterFCT(flx_HO, dxc, dt, uf, C, theta_remove, field_previous, use_previous=False, niter=1, ymin=None, ymax=None):
    """This function implements iterative FCT, as described in MULES_HW.pdf (/strang_carryover_1d paper as MULES_HW.pdf has some notational problems)
    ymax and ymin are overall global min/max values if set.
    previous: previous time step field - True: element uses field_previous for extrema, not if False. Defined at i-1/2
    02-07-2025: dxc assumed constant. uf and C assumed potentially nonconstant and negative
    theta off-centering for AdImEx Upwind is assumed to be max(0,1-1/C at neighboring cells) to ensure monotonicity.
    flx_HO needs to be u*field at i-1/2
    """
    nx = len(flx_HO)
    flx_bounded, field_bounded = np.zeros(nx), np.zeros(nx)

    ufp = 0.5*(uf + abs(uf)) # [i] at i-1/2
    ufm = 0.5*(uf - abs(uf)) # [i] at i-1/2

    C_out = (-ufm + np.roll(ufp,-1))*dt/dxc # [i] at i
    C_in = (ufp - np.roll(ufm,-1))*dt/dxc # [i] at i
    theta = np.maximum(0., 1. - 1./(C_in + C_out)) # [i] at i # paper version needs this in a separate function
    thetaf = np.maximum(np.roll(theta,1), theta) # [i] at i-1/2
    #thetaf = np.ones(nx) # [i] at i-1/2

    # !!! include the previous field into extrema when C<=1 as well?

    # Calculate bounded field and bounded flux - 1 time step of AdImEx Upwind low-order bounded (.. with 1-1/c_total for theta) solution (this will subsequently be updated in FCT iteration loop)
    ufp_thetaf, ufm_thetaf = thetaf*ufp, thetaf*ufm # [i] at i-1/2
    M = np.zeros((nx, nx))
    for i in range(nx): 
        M[i,i] = 1. + dt*(ufp_thetaf[(i+1)%nx] - ufm_thetaf[i])/dxc[i]
        M[i,(i-1)%nx] = -dt*ufp_thetaf[i]/dxc[i]
        M[i,(i+1)%nx] = dt*ufm_thetaf[(i+1)%nx]/dxc[i]

    ufp_1mthetaf, ufm_1mthetaf = (1.-thetaf)*ufp, (1.-thetaf)*ufm # [i] at i-1/2
    rhs = field_previous - (np.roll(ufp_1mthetaf,-1)*field_previous - ufp_1mthetaf*np.roll(field_previous,1) + np.roll(ufm_1mthetaf*field_previous,-1) - ufm_1mthetaf*field_previous)*dt/dxc # [i] at i
    field_bounded = np.linalg.solve(M, rhs) # [i] at i
    flx_bounded = ufp_1mthetaf*np.roll(field_previous,1) + ufp_thetaf*np.roll(field_bounded,1) + ufm_1mthetaf*field_previous + ufm_thetaf*field_bounded # [i] at i-1/2

    # Set allowable min and max values (not iterated over!)
    fieldmin, fieldmax = set_extrema(nx, uf, field_bounded, field_previous, use_previous, ymin=ymin, ymax=ymax) 

    #plt.plot(ufp_thetaf)
    #plt.plot(ufm_thetaf)
    #plt.show()

    #plt.plot(field_bounded, label='bounded')
    #plt.plot(fieldmin, label='min')
    #plt.plot(fieldmax, label='max')
    #plt.legend()
    #plt.show()


    ###!#### # Making it implicit upwind:
###!#### 
    ###!#### # Calculate bounded field and bounded flux - 1 time step of AdImEx Upwind low-order bounded (.. with 1-1/c_total for theta) solution (this will subsequently be updated in FCT iteration loop)
    ###!#### #ufp_thetaf, ufm_thetaf = thetaf*ufp, thetaf*ufm # [i] at i-1/2
    ###!#### M = np.zeros((nx, nx))
    ###!#### for i in range(nx): 
    ###!####     M[i,i] = 1. + dt*(ufp[(i+1)%nx] - ufm[i])/dxc[i]
    ###!####     M[i,(i-1)%nx] = -dt*ufp[i]/dxc[i]
    ###!####     M[i,(i+1)%nx] = dt*ufm[(i+1)%nx]/dxc[i]
###!#### 
    ###!#### #ufp_1mthetaf, ufm_1mthetaf = (1.-thetaf)*ufp, (1.-thetaf)*ufm # [i] at i-1/2
    ###!#### #rhs = field_previous # [i] at i
    ###!#### field_bounded = np.linalg.solve(M, field_previous) # [i] at i
    ###!#### flx_bounded = ufp*np.roll(field_bounded,1) + ufm*field_bounded # [i] at i-1/2
###!#### 
    ###!#### #print(use_previous)
    ###!#### # Set allowable min and max values (not iterated over!)
    ###!#### fieldmin, fieldmax = set_extrema(nx, uf, field_bounded, field_previous, use_previous, ymin=ymin, ymax=ymax) 
###!#### 
    ###plt.plot(field_previous, label='previous')
    ###plt.plot(field_bounded, label='bounded')
    ###plt.plot(ufp/10.)
    ###plt.plot(ufm)
    ###plt.legend()
    ###plt.show()
    #plt.show()


    # FCT iteration loop
    for iiter in range(niter):
        # Calculate high-order correction
        corr = flx_HO - flx_bounded # [i] at i-1/2 # uf*field

        if ymin is None and ymax is None:
            # Checking for rare cases where we need to set corr to zero - necessary for monotonicity (bug 30-06-2025)
            for i in range(nx):
                if corr[i]*(field_bounded[i] - field_bounded[i-1]) <= 0. and (corr[i]*(field_bounded[(i+1)%nx] - field_bounded[i]) <= 0. or corr[i]*(field_bounded[i-1] - field_bounded[i-2]) <= 0.):
                    corr[i] = 0. 

        # Calculate allowable mass I/O for max rise and fall
        Qp = dxc*(fieldmax - field_bounded) # [i] at i
        Qm = dxc*(field_bounded - fieldmin) # [i] at i

        # Calculate I/O fluxes at cell centers
        Pp = dt*(np.maximum(0, corr) - np.minimum(0, np.roll(corr,-1)))
        Pm = dt*(np.maximum(0, np.roll(corr,-1)) - np.minimum(0, corr))

        # Calculate ratios of allowable (Q) to existing high-order (P) fluxes
        Rp = np.where(Pp > 1e-12, np.minimum(1., Qp/np.maximum(Pp,1e-12)), 0.)
        Rm = np.where(Pm > 1e-12, np.minimum(1., Qm/np.maximum(Pm,1e-12)), 0.)

        # Calculate the limiter for each face
        face_limiter = np.where(corr >= 0., np.minimum(Rp, np.roll(Rm,1)), np.minimum(np.roll(Rp,1), Rm)) # [i] at i-1/2

        # Update the bounded flux and field
        flx_bounded += face_limiter*corr
        field_bounded = field_previous - dt/dxc*(np.roll(flx_bounded,-1) - flx_bounded)

        #plt.plot(flx_HO, label='flx_HO')
        #plt.plot(flx_bounded, label='flx_bounded')
        #plt.plot(field_bounded*8, label='field_bounded')
        #plt.legend()
        #plt.show()

    # Output limited field[it+1] = field_bounded after niter iterations
    return field_bounded
