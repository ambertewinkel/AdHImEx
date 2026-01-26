"""This file includes functions for the FCT limiter."""


import numpy as np


def set_extrema(nx, field_bounded, field_previous, use_previous=False, ymin=None, ymax=None):
    """This function returns the min and max values allowed for each grid cell, defined at center. Used in FCT."""

    fieldmin, fieldmax = np.zeros(nx), np.zeros(nx)

    # Determine allowable minima
    if ymin is not None: # global allowable min/max (nonnegative FCT)
        fieldmin = np.full(nx, ymin)
    else: # AdImEx FCT
        if use_previous:
            for i in range(nx):
                fieldmin[i] = min([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx], field_previous[i-1], field_previous[i], field_previous[(i+1)%nx]])
        else:
            for i in range(nx):
                fieldmin[i] = min([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx]]) 

    # Determine allowable maxima
    if ymax is not None: # global allowable min/max (nonnegative FCT)
        fieldmax = np.full(nx, ymax)
    else: # AdImEx FCT
        if use_previous:
            for i in range(nx):
                fieldmax[i] = max([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx], field_previous[i-1], field_previous[i], field_previous[(i+1)%nx]])
        else:
            for i in range(nx):
                fieldmax[i] = max([field_bounded[i-1], field_bounded[i], field_bounded[(i+1)%nx]])

    return fieldmin, fieldmax


def FCT(flx_HO, dxc, dt, uf, field_previous, use_previous=False, ymin=None, ymax=None, nondivergent=False):
    """This function implements flux-corrected transport (FCT). Either using global bounds or local bounds based on low-order bounded solution and optionally previous time step. Returns the limited field at the new time step. Assumes constant dxc and uf>0.
    flx_HO: high-order flux at i-1/2
    dxc : cell-centred width at i
    dt : time step
    uf : velocity at i-1/2
    field_previous : field at previous time step n, at i
    use_previous : whether to use previous time step field to set extrema (True if all C in space <= 1 else False)
    ymin, ymax : allowable min and max values for the field. If None, local extrema are used.
    nondivergent : whether the winds are nondivergent (True/False)
    """
    nx = len(flx_HO)
    ufp = 0.5*(uf + abs(uf)) # [i] at i-1/2
    ufm = 0.5*(uf - abs(uf)) # [i] at i-1/2
    
    # Calculate low-order bounded flux and field
    flx_bounded, field_bounded = np.zeros(nx), np.zeros(nx)
    if nondivergent:
        C_out = (-ufm + np.roll(ufp,-1))*dt/dxc # [i] at i
        C_in = (ufp - np.roll(ufm,-1))*dt/dxc # [i] at i
        theta = np.maximum(0., 1. - 1./(C_in + C_out)) # [i] at i
        thetaf = np.maximum(np.roll(theta,1), theta) # [i] at i-1/2
    else:
        if use_previous:
            thetaf = np.zeros(nx) # [i] at i-1/2
        else:
            thetaf = np.ones(nx) # [i] at i-1/2
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

    # Set allowable min and max values
    fieldmin, fieldmax = set_extrema(nx, field_bounded, field_previous, use_previous, ymin=ymin, ymax=ymax) 

    ######### FCT ALGORITHM #########

    # Calculate high-order correction
    corr = flx_HO - flx_bounded # [i] at i-1/2

    if ymin is None and ymax is None:
        # Checking for rare cases where we need to set the correction to zero
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

    return field_bounded
