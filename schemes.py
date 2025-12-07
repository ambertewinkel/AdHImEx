import numpy as np
import limiter as lim

def space_AdHImEx(field):
    """Returns the flux for a spatial discretisation at [i] using [i+2], [i+1], [i], [i-1], [i-2], [i-3]. Output defined at i-1/2."""
    return -1./20*np.roll(field,-1) + 9./20.*field + 47./60.*np.roll(field,1) - 13./60.*np.roll(field,2) + 1./30.*np.roll(field,3)


def matrix_AdHImEx(nx, dt, dx, u, alpha):
    """This function returns the matrix M for the AdHImEx spatial discretisation. That is, the discretisation at time level n+1, which is then on the LHS combined with the field[n+1,i] term."""
    M = np.zeros((nx, nx))
    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!)
        M[i,(i-3)] = -1./30.*dt*alpha*u[i]/dx[i]
        M[i,(i-2)] = dt*alpha*(1./30.*np.roll(u,-1)[i] + 13./60.*u[i])/dx[i]
        M[i,i-1] = dt*alpha*(-13./60.*np.roll(u,-1)[i] - 47./60.*u[i])/dx[i]
        M[i,i] = 1. + dt*alpha*(47./60.*np.roll(u,-1)[i] - 9./20.*u[i])/dx[i]
        M[i,(i+1)%nx] = dt*alpha*(9./20.*np.roll(u,-1)[i] + 1./20.*u[i])/dx[i]
        M[i,(i+2)%nx] = -1./20.*dt*alpha*np.roll(u,-1)[i]/dx[i]
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
    dfdx[i] = (f_{i+1/2} - f_{i-1/2})/dx
    fmh : f_{i-1/2}
    fph : f_{i+1/2}
    """
    return (fph - fmh)/dxc


def AdHImEx(init, nt, dt, uf, dxc, FCTiter=0, output_substages=False, substages=np.array([]), FCT_min=None, FCT_max=None, FCT_use_previous=False):
    """Implements the AdHImEx scheme for a given initial field, number of time steps, time step length, velocity field at cell faces, cell widths, number of FCT iterations (0 for no FCT), whether to output substages (True/False) and number of FCT iterations."""
    nx = len(init)

    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    xf = np.zeros(nx) 
    for i in range(nx-1): # assumes uniform grid
        xf[i+1] = xf[i] + dxc[i]

    # Set up Butcher tableau
    AIm, bIm = butcherIm_AdHImEx()
    AEx, bEx = butcherEx_AdHImEx()
    nstages = np.shape(bIm)[1]
    # Resetting A to include b
    AEx = np.concatenate((AEx,bEx), axis=0)
    AIm = np.concatenate((AIm,bIm), axis=0)
    AEx = np.concatenate((AEx, np.zeros((nstages+1,1))), axis=1)
    AIm = np.concatenate((AIm, np.zeros((nstages+1,1))), axis=1)
    flx_k, fEx, fIm, flx_contribution_from_stage_k = np.zeros((nstages+1, nx)), np.zeros((nstages+1, nx)), np.zeros((nstages+1, nx)), np.zeros((nstages+1, nx))

    thetaf, cf = np.zeros((nt,nx)), np.zeros((nt,nx))
    for it in range(nt): # recalculate each unique theta for each time step (inefficient coding for uf=constant in space or time)
        cf[it] = uf[it]*dt/dxc # [i] at i-1/2
        #c_temp = np.maximum(cf[it], np.roll(cf[it],-1)) # [i] at i
        #cf_temp = np.maximum(c_temp, np.roll(c_temp,1)) # [i] at i-1/2
        thetaf[it] = implicitness(cf[it]) # [i] at i-1/2 # !!! is doing the max etc now correct?
        
        #cc_out = 0.5*(np.abs(uf[it]) - uf[it] + np.abs(np.roll(uf[it],-1)) + np.roll(uf[it],-1))*dt/dxc # [i] at i, Courant defined at cell centers based on the *outward* pointing velocities
        #cc_in = 0.5*(np.abs(uf[it]) + uf[it] + np.abs(np.roll(uf[it],-1)) - np.roll(uf[it],-1))*dt/dxc # [i] at i, Courant defined at cell centers based on the *inward* pointing velocities
        #thetac_out = np.maximum(0., 1.-1./cc_out)
        #thetac_in = np.maximum(0., 1.-1./cc_in)
        #theta[it] = np.maximum(np.maximum(thetac_out, np.roll(thetac_out,1)), np.maximum(thetac_in, np.roll(thetac_in, 1))) # [i] at i-1/2

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(nstages+1):
            # Calculate the field at stage k
            M = matrix_AdHImEx(nx, dt, dxc, thetaf[it]*uf[it], AIm[ik,ik]) # [i] at i
            rhs_k = field[it] + dt*np.dot(AEx[ik,:ik], fEx[:ik,:]) + dt*np.dot(AIm[ik,:ik], fIm[:ik,:]) # [i] at i
            field_k = np.linalg.solve(M, rhs_k) # [i] at i
            if output_substages: 
                substages[ik+1] = field_k
            # Calculate the flux based on the field at stage k
            flx_k[ik,:] = uf[it]*space_AdHImEx(field_k) # [i] at i-1/2
            fEx[ik,:] = -ddx((1 - thetaf[it])*flx_k[ik,:], np.roll((1 - thetaf[it])*flx_k[ik,:],-1), dxc)
            fIm[ik,:] = -ddx(thetaf[it]*flx_k[ik,:], np.roll(thetaf[it]*flx_k[ik,:],-1), dxc)   
            flx_contribution_from_stage_k[ik,:] = AEx[-1,ik]*(1 - thetaf[it])*flx_k[ik,:] + AIm[-1,ik]*thetaf[it]*flx_k[ik,:]
            flx_HO += flx_contribution_from_stage_k[ik,:]  
        if FCTiter > 0:
            #previous = np.full(nx, False) #[True if cf[i] <= 1. else False for i in range(nx)] # [i] at i-1/2 # determines whether FCT also uses field[it] for bounds (True/False) # could use further consideration
            field[it+1] = lim.iterFCT(flx_HO, dxc, dt, uf[it], cf[it], thetaf[it], field[it], use_previous=FCT_use_previous, niter=FCTiter, ymin=FCT_min, ymax=FCT_max) # also option for ymin and ymax         
        else:     
            field[it+1] = field_k.copy()

    return field


def implicitness(C):
    """This function returns the AdHImEx implicitness parameter theta for a given Courant number C."""
    #return np.maximum(0., 1.-1./C) # !!! needs changing, but first checking with this definition
    return 1. - 1./(1. + 0.7*np.maximum(0., C - 1.4))


def matrix_WKS24(nx, dt, dxc, ufp, ufm, alpha, theta):
    """This function returns the matrix M for the WKS24 spatial discretisation. That is, the spatial discretisation at time level n+1, which is then on the LHS combined with the field[n+1,i] term."""
    M = np.zeros((nx, nx))
    for i in range(nx):	# includes flx_1st_k # based on implicit upwind matrices above.
        M[i,i] = 1. + dt*(np.roll(alpha*theta*ufp,-1)[i] - alpha[i]*theta[i]*ufm[i])/dxc[i]
        M[i,i-1] = -dt*alpha[i]*theta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%nx] = dt*np.roll(alpha*theta*ufm,-1)[i]/dxc[i]
    return M


def space_WKS24(psi): # quadh
    """-- quasicubic (third-order in space for uniform grid)-- This quadratic interpolation for f[i+1/2] leads to a cubic approximation when put in the FV ddx scheme. The quadratic interpolation matches the integral of the polynomial to the integral of the field over the three cells. See notes sent by James Kent on 28-11-2024.
    --- in ---
    psi : field at [i]
    --- out --- 
    psi at [i+1/2] 
    """
    return (2.*np.roll(psi, -1) + 5.*psi - np.roll(psi, 1))/6.


def WKS24(init, nt, dt, uf, dxc, kmax=2, set_alpha='max'):
    """This scheme solves second-order Runge-Kutta quasi-cubic scheme. See HW notes sent on 27-11-2024."""
    # assumes u>0

    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    fieldh_HO_n, flx_HO_n, fieldh_1st_km1, flx_1st_km1, fieldh_HO_km1, fieldh_HOC_km1, flx_HOC_km1, rhs, theta = np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx)

    for it in range(nt):
        ufp = 0.5*(uf[it] + abs(uf[it])) # uf[i] is at i-1/2
        ufm = 0.5*(uf[it] - abs(uf[it]))
        C = dt*uf[it]/dxc # assumes uniform grid
        #alpha = np.maximum(0.5, 1. - 1./c) # assumes uniform grid # alpha[i] is at i-1/2
        for i in range(nx):
            theta[i] = 0. if C[i] < 0.8 else 1 # theta[i] is at i-1/2 # check definition with paper

        if set_alpha == 'max': # assumes uniform grid # alpha[i] is at i-1/2
            alpha = np.maximum(0.5, 1. - 1./C)
        elif set_alpha == 'half':
            alpha = np.full(len(init), 0.5)
        else:
            print('Error: set_alpha must be either "half" or "max"') # !!! figure out this error handling properly. + perhaps can just set alpha as one option... check scheme definition in paper
            return
        
        field[it+1] = field[it].copy() # not actually in the equations, this is to make the computer code more concise
        M = matrix_WKS24(nx, dt, dxc, ufp, ufm, alpha, theta) # [i] at i
        for k in range(kmax):
            for i in range(nx): # !!! make this upwind? # also make it look nicer with np.roll
                fieldh_HO_n[i] = space_WKS24(np.roll(field[it],1))[i] # [i] defined at i-1/2
                flx_HO_n[i] = (1. - alpha[i])*uf[it, i]*fieldh_HO_n[i] # [i] defined at i-1/2
                fieldh_1st_km1[i] = field[it+1,i-1] # upwind # [i] defined at i-1/2 # not actually field[it+1], this is to make the computer code more concise
                flx_1st_km1[i] = alpha[i]*(1. - theta[i])*uf[it, i]*fieldh_1st_km1[i] # [i] defined at i-1/2
                fieldh_HO_km1[i] = space_WKS24(np.roll(field[it+1],1))[i] # [i] defined at i-1/2
                fieldh_HOC_km1[i] = fieldh_HO_km1[i] - fieldh_1st_km1[i] #fieldh_HO_n[i] - fieldh_1st_km1[i] # [i] defined at i-1/2
                flx_HOC_km1[i] = alpha[i]*uf[it, i]*fieldh_HOC_km1[i] # [i] defined at i-1/2
            for i in range(nx):
                rhs[i] = field[it,i] - dt*(ddx(flx_HO_n[i], flx_HO_n[(i+1)%nx], dxc[i]) + \
                        ddx(flx_1st_km1[i], flx_1st_km1[(i+1)%nx], dxc[i]) + \
                        ddx(flx_HOC_km1[i], flx_HOC_km1[(i+1)%nx], dxc[i])) # [i] defined at i
            field[it+1] = np.linalg.solve(M, rhs)    

    return field

# !!! make sure upwind code is actually upwind