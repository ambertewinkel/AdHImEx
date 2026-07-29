"""This file includes functions for the AdHImEx and WKS24 schemes."""


import numpy as np
import limiter as lim


################################
######## AdHImEx scheme ########
################################


def space_AdHImEx(field): #temporary third
    """Returns the flux for the fifth-order spatial discretisation at [i] using [i+2], [i+1], [i], [i-1], [i-2], [i-3] for field. Output defined at i-1/2."""
    #return -1./20*np.roll(field,-1) + 9./20.*field + 47./60.*np.roll(field,1) - 13./60.*np.roll(field,2) + 1./30.*np.roll(field,3)
    return 1./3.*field + 5./6.*np.roll(field,1) - 1./6.*np.roll(field,2)


def matrix_AdHImEx(nx, dt, dxc, uf, alpha):
    """This function returns the matrix M for the fifth-order AdHImEx spatial discretisation. Assumes uf>0.
    nx : number of grid cells
    dt : time step
    dxc : cell-centred width at i
    uf : velocity at faces ([i] at i-1/2)
    alpha : Runge-Kutta coefficient
    """
    M = np.zeros((nx, nx)) # temporary third
    #for i in range(nx):
    #    M[i,(i-3)] = -1./30.*dt*alpha*uf[i]/dxc[i]
    #    M[i,(i-2)] = dt*alpha*(1./30.*np.roll(uf,-1)[i] + 13./60.*uf[i])/dxc[i]
    #    M[i,i-1] = dt*alpha*(-13./60.*np.roll(uf,-1)[i] - 47./60.*uf[i])/dxc[i]
    #    M[i,i] = 1. + dt*alpha*(47./60.*np.roll(uf,-1)[i] - 9./20.*uf[i])/dxc[i]
    #    M[i,(i+1)%nx] = dt*alpha*(9./20.*np.roll(uf,-1)[i] + 1./20.*uf[i])/dxc[i]
    #    M[i,(i+2)%nx] = -1./20.*dt*alpha*np.roll(uf,-1)[i]/dxc[i]
    #return M 
    for i in range(nx):
        M[i,(i-2)] = dt*alpha*1./6.*uf[i]/dxc[i]
        M[i,i-1] = dt*alpha*(-1./6.*np.roll(uf,-1)[i] - 5./6.*uf[i])/dxc[i]
        M[i,i] = 1. + dt*alpha*(5./6.*np.roll(uf,-1)[i] - 1./3.*uf[i])/dxc[i]
        M[i,(i+1)%nx] = dt*alpha*1./3.*np.roll(uf,-1)[i]/dxc[i]
    return M


def butcherEx_AdHImEx():
    """Left (explicit) butcher tableau from Ullrich and Jablonowski 2012. See the Weller Lock and Wood (2013) UJ3(1+e,3,2) scheme."""   
    A = np.array([[0., 0., 0., 0., 0.],[0., 0., 0., 0., 0.],[0., 1., 0., 0., 0.],[0., 0.25, 0.25, 0., 0.],[0., 1/6, 1/6, 2/3, 0.]])
    b = np.array([[0., 1/6, 1/6, 2/3, 0.]])
    return A, b


def butcherIm_AdHImEx():
    """Right (implicit) butcher tableau from Ullrich and Jablonowski 2012. See the Weller Lock and Wood (2013) UJ3(1+e,3,2) scheme."""   
    A = np.array([[0., 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.5]])
    b = np.array([[0.5, 0., 0., 0., 0.5]])
    return A, b


def ddx(fmh, fph, dxc):
    """This function computes the first derivative of a field f with respect to x with a finite-volume method. 
    dxc : cell-centred width at i
    fmh : field at i-1/2
    fph : field at i+1/2
    """
    return (fph - fmh)/dxc


def AdHImEx(init, nt, dt, uf, dxc, unity=True, FCT=False, output_substages=False, substages=np.array([]), ymin=None, ymax=None, nondivergent=False):
    """Implements the AdHImEx scheme. Returns the field at all time steps, including the initial condition: field[nt+1, nx]. Assumes periodic boundaries, uniform grid and nonnegative velocity at faces.
    init : 1D array of floats, initial condition at t=0
    nt   : int, number of time steps
    dt   : float, time step length
    uf   : 2D array of floats, velocity field at cell faces for all time
    dxc  : 1D array of floats, cell widths
    unity: bool, whether to use the unity preservation modification (default True)
    FCT  : bool, whether to use FCT (default False)
    output_substages: bool, whether to output substages (default False)
    substages: 2D array of floats, to store the substages (default empty array)
    FCT_min: float, minimum value for FCT limiter (None for no limit)
    FCT_max: float, maximum value for FCT limiter (None for no limit)
    nondivergent: bool, whether to use the nondivergent FCT algorithm (default False)
    """
    # General setup
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    xf = np.zeros(nx) 
    for i in range(nx-1):
        xf[i+1] = xf[i] + dxc[i]

    # Set up Butcher tableau
    AIm, bIm = butcherIm_AdHImEx()
    AEx, bEx = butcherEx_AdHImEx()
    nstages = np.shape(bIm)[1]
    AEx = np.concatenate((AEx,bEx), axis=0) # Resetting A to include b
    AIm = np.concatenate((AIm,bIm), axis=0)
    AEx = np.concatenate((AEx, np.zeros((nstages+1,1))), axis=1)
    AIm = np.concatenate((AIm, np.zeros((nstages+1,1))), axis=1)
    flx_k, fEx_f, fIm_f, fEx_c, fIm_c, flx_contribution_from_stage_k = np.zeros((nstages+1, nx)), np.zeros((nstages+1, nx)), np.zeros((nstages+1, nx)), np.zeros((nstages+1, nx)), np.zeros((nstages+1, nx)), np.zeros((nstages+1, nx))

    # Calculating the implicitness at cell centres and faces for all time steps
    thetaf, cf, thetac, cc = np.zeros((nt,nx)), np.zeros((nt,nx)), np.zeros((nt,nx)), np.zeros((nt,nx))
    for it in range(nt):
        cf[it] = uf[it]*dt/dxc # [i] at i-1/2
        thetaf[it] = implicitness(cf[it]) # [i] at i-1/2
        cc[it] = 0.5*(np.abs(uf[it]) - uf[it] + np.abs(np.roll(uf[it],-1)) + np.roll(uf[it],-1))*dt/dxc # [i] at i, Courant defined at cell centers based on the *outward* pointing velocities
        thetac[it] = implicitness(cc[it])

    # Time-stepping loop
    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)

        # Loop over the Runge-Kutta stages
        for ik in range(nstages+1):
            # Calculate the field at stage k (field_k)
            M = matrix_AdHImEx(nx, dt, dxc, thetaf[it]*uf[it], AIm[ik,ik]) # [i] at i
            if unity and (ik == 1 or ik == 2):
                rhs_k = field[it] + dt*np.dot(AEx[ik,:ik], fEx_c[:ik,:]) + dt*np.dot(AIm[ik,:ik], fIm_c[:ik,:]) # [i] at i
            else:
                rhs_k = field[it] + dt*np.dot(AEx[ik,:ik], fEx_f[:ik,:]) + dt*np.dot(AIm[ik,:ik], fIm_f[:ik,:]) # [i] at i
            field_k = np.linalg.solve(M, rhs_k) # [i] at i

            # Store the substage field if required
            if output_substages: 
                substages[ik+1] = field_k

            # Calculate the flux based on the field at stage k
            flx_k[ik,:] = uf[it]*space_AdHImEx(field_k) # [i] at i-1/2
            fEx_f[ik,:] = -ddx((1 - thetaf[it])*flx_k[ik,:], np.roll((1 - thetaf[it])*flx_k[ik,:],-1), dxc)
            fIm_f[ik,:] = -ddx(thetaf[it]*flx_k[ik,:], np.roll(thetaf[it]*flx_k[ik,:],-1), dxc)   
            fEx_c[ik,:] = -(1 - thetac[it])*ddx(flx_k[ik,:], np.roll(flx_k[ik,:],-1), dxc)
            fIm_c[ik,:] = -thetac[it]*ddx(flx_k[ik,:], np.roll(flx_k[ik,:],-1), dxc)  

            # Accumulate the flux contributions from the stages (needed for FCT)
            flx_contribution_from_stage_k[ik,:] = AEx[-1,ik]*(1 - thetaf[it])*flx_k[ik,:] + AIm[-1,ik]*thetaf[it]*flx_k[ik,:]
            flx_HO += flx_contribution_from_stage_k[ik,:] 

        # Implement FCT if required
        if FCT:
            use_previous = np.all(cc[it] <= 1.) 
            field[it+1] = lim.FCT(flx_HO, dxc, dt, uf[it], field[it], use_previous=use_previous, ymin=ymin, ymax=ymax, nondivergent=nondivergent)         
        else:     
            field[it+1] = field_k.copy()

    return field


def implicitness(C):
    """This function returns the AdHImEx implicitness parameter theta for a given Courant number C."""
    return 1. - 1./(1. + 0.7*np.maximum(0., C - 1.4))


##############################
######## WKS24 scheme ########
##############################


def matrix_WKS24(nx, dt, dxc, ufp, ufm, alpha, beta):
    """This function returns the matrix M for the third-order WKS24 spatial discretisation.
    nx : number of grid cells
    dt : time step
    dxc : cell-centred width at i
    ufp : velocity at faces, pointing in positive x-direction ([i] at i-1/2)
    ufm : velocity at faces, pointing in negative x-direction ([i] at i-1/2)
    alpha, beta : parameters"""
    M = np.zeros((nx, nx))
    for i in range(nx):
        M[i,i] = 1. + dt*(np.roll(alpha*beta*ufp,-1)[i] - alpha[i]*beta[i]*ufm[i])/dxc[i]
        M[i,i-1] = -dt*alpha[i]*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%nx] = dt*np.roll(alpha*beta*ufm,-1)[i]/dxc[i]
    return M


def space_WKS24(psi):
    """Returns the flux for the third-order WKS24 spatial discretisation at [i] for field. Output defined at i+1/2."""
    return (2.*np.roll(psi, -1) + 5.*psi - np.roll(psi, 1))/6.


def WKS24(init, nt, dt, uf, dxc, kmax=2, set_alpha='max'):
    """This scheme solves the WKS24 scheme. Assumes uniform grid and uf>0
    init : 1D array of floats, initial condition at t=0
    nt   : int, number of time steps
    dt   : float, time step length
    uf   : 2D array of floats, velocity field at cell faces for all time
    dxc  : 1D array of floats, cell-centred widths
    kmax : int, number of corrector iterations
    set_alpha : string, either 'max' or 'half' to set the alpha parameter
    """
    # Setup 
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    fieldh_HO_n, flx_HO_n, fieldh_1st_km1, flx_1st_km1, fieldh_HO_km1, fieldh_HOC_km1, flx_HOC_km1, rhs, beta = np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx)

    # Time-stepping loop
    for it in range(nt):
        # Set beta
        ufp = 0.5*(uf[it] + abs(uf[it])) # [i] at i-1/2
        ufm = 0.5*(uf[it] - abs(uf[it])) # [i] at i-1/2        
        C = dt*uf[it]/dxc
        for i in range(nx):
            beta[i] = 0. if C[i] < 0.8 else 1 # [i] at i-1/2

        # Set alpha parameter
        if set_alpha == 'max':# [i] at i-1/2
            alpha = np.maximum(0.5, 1. - 1./C)
        elif set_alpha == 'half':
            alpha = np.full(len(init), 0.5)
        else:
            print('Error: set_alpha must be either "half" or "max"')
            return
        
        # Predictor-corrector iterations
        field[it+1] = field[it].copy() # this is to make the code more concise
        M = matrix_WKS24(nx, dt, dxc, ufp, ufm, alpha, beta) # [i] at i
        for k in range(kmax):
            for i in range(nx): 
                fieldh_HO_n[i] = space_WKS24(np.roll(field[it],1))[i] # [i] at i-1/2
                flx_HO_n[i] = (1. - alpha[i])*uf[it, i]*fieldh_HO_n[i] # [i] at i-1/2
                fieldh_1st_km1[i] = field[it+1,i-1] # [i] at i-1/2 # not actually field[it+1], this is to make the code more concise
                flx_1st_km1[i] = alpha[i]*(1. - beta[i])*uf[it, i]*fieldh_1st_km1[i] # [i] at i-1/2
                fieldh_HO_km1[i] = space_WKS24(np.roll(field[it+1],1))[i] # [i] at i-1/2
                fieldh_HOC_km1[i] = fieldh_HO_km1[i] - fieldh_1st_km1[i] # [i] at i-1/2
                flx_HOC_km1[i] = alpha[i]*uf[it, i]*fieldh_HOC_km1[i] # [i] at i-1/2
            for i in range(nx):
                rhs[i] = field[it,i] - dt*(ddx(flx_HO_n[i], flx_HO_n[(i+1)%nx], dxc[i]) + \
                        ddx(flx_1st_km1[i], flx_1st_km1[(i+1)%nx], dxc[i]) + \
                        ddx(flx_HOC_km1[i], flx_HOC_km1[(i+1)%nx], dxc[i])) # [i] at i
            field[it+1] = np.linalg.solve(M, rhs)    

    return field