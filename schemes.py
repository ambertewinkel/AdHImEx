"""This file includes functions for the AdHImEx and WKS24 schemes."""


import numpy as np
import limiter as lim
from functools import partial


################################
######## AdHImEx scheme ########
################################


def space_AdHImEx(field):
    """Returns the flux for the fifth-order spatial discretisation at [i] using [i+2], [i+1], [i], [i-1], [i-2], [i-3] for field. Output defined at i-1/2."""
    return -1./20*np.roll(field,-1) + 9./20.*field + 47./60.*np.roll(field,1) - 13./60.*np.roll(field,2) + 1./30.*np.roll(field,3)


def matrix_AdHImEx(nx, dt, dxc, uf, alpha):
    """This function returns the matrix M for the fifth-order AdHImEx spatial discretisation. Assumes uf>0.
    nx : number of grid cells
    dt : time step
    dxc : cell-centred width at i
    uf : velocity at faces ([i] at i-1/2)
    alpha : Runge-Kutta coefficient
    """
    M = np.zeros((nx, nx))
    for i in range(nx):
        M[i,(i-3)] = -1./30.*dt*alpha*uf[i]/dxc[i]
        M[i,(i-2)] = dt*alpha*(1./30.*np.roll(uf,-1)[i] + 13./60.*uf[i])/dxc[i]
        M[i,i-1] = dt*alpha*(-13./60.*np.roll(uf,-1)[i] - 47./60.*uf[i])/dxc[i]
        M[i,i] = 1. + dt*alpha*(47./60.*np.roll(uf,-1)[i] - 9./20.*uf[i])/dxc[i]
        M[i,(i+1)%nx] = dt*alpha*(9./20.*np.roll(uf,-1)[i] + 1./20.*uf[i])/dxc[i]
        M[i,(i+2)%nx] = -1./20.*dt*alpha*np.roll(uf,-1)[i]/dxc[i]
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


def AdHImEx_gmresm(init, nt, dt, uf, dxc, unity=True, FCT=False, output_substages=False, substages=np.array([]), ymin=None, ymax=None, nondivergent=False):
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

    uses GMRESm to solve the matrix
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
            
            if ik == 4 and np.any(thetac): # 22-12-2025: I think this is necessary for GMRES not breaking down because of existing convergence (when the matrix is full of zeros)
                matrix = partial(matrix_func, M) # at [i,j]
                field_k = gmresm(matrix, rhs_k, field_3, kiter=200, jiter=5, tolerance=1e-6, it=it)
            elif ik == 2:
                field_3 = rhs_k.copy()
                field_k = rhs_k.copy()
            else:
                field_k = rhs_k.copy()

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


def matrix_func(M, x):
    """This function computes the matrix-vector product M x, where M is a matrix and x is a vector."""
    return np.dot(M, x)


def gmresm(A, b, x, kiter=10, jiter=5, tolerance=1e-6, irestarts_convergence=np.zeros(10), j_convergence=np.zeros(10), iterations_convergence=np.zeros(10), it=0):
    """
    Matrixfree solution of linear Ax=b system using GMRES(m) method. (matrixfree through a function that computes Ax with def A(x))).
    Semi-optimised version (i.e., implemented QR factorisation/least squares minimisation in Saad and Schultz 1986 p.860-862, but not the last step part).
    However, GMRES(m) does need a small matrix H (R_k here) to be stored and solved. Apart from that, it currently stores a V matrix, arrays of size (m+1,N) where N is the size of the problem. This could be improved to reduce memory usage (memory usage is already improved with the restarting).
    --- IN --- 
    A: function to implement the A matrix
    b: N vector, rhs of equation
    x: N vector, initial guess for solution
    --- OUT ---
    x : converged (or cut short) solution
    """
    x = x.copy()

    r0 = b - A(x)

    reltol = tolerance * np.linalg.norm(b) # relative tolerance; see GMRES slides https://www.dmsa.unipd.it/~berga/Teaching/Phd/gmres_slides.pdf and Wikipedia https://en.wikipedia.org/wiki/Generalized_minimal_residual_method; I think MATLAB and Python compare the residual to the relative tolerance as well: https://www.mathworks.com/help/matlab/ref/gmres.html and https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html
    norm_oldres = np.linalg.norm(r0)
    if norm_oldres < reltol: # This could also be a problem. Now we are doing more iterations once the tolerance has not been achieved yet - but if it has been achieved, we don't do anything (which is fair) - so now you have a good number of time steps where you would maybe preferably want a better solution as well but you don't get this. Could this explain the difficulty in reducing the error for small max C (just beyond the 1.4 threshold?)
        print(f"Initial guess is already good enough with residual {norm_oldres} (relative tolerance {reltol}).")
        return x
    #reduction_tolerance = 1e-5*norm_oldres
    #print('values', norm_oldres, reltol, reduction_tolerance)

    converged = False
    for irestart in range(kiter):
        h = np.zeros((jiter+1, jiter), dtype=np.float64)
        cols = h.shape[1]
        v = np.zeros((jiter+1, *np.shape(r0)), dtype=np.float64)
        v_hat = np.zeros(np.shape(r0), dtype=np.float64)
        v[0] = r0/norm_oldres

        cos = np.zeros(jiter, dtype=np.float64)
        sin = np.zeros(jiter, dtype=np.float64)
        g = np.zeros(jiter+1, dtype=np.float64)
        g[0] = norm_oldres # residual norm at start of restart loop
        jend = jiter

        for j in range(jiter):            
            Avj = A(v[j])
            for i in range(j+1):
                h[i, j] = np.dot(Avj.ravel(), v[i].ravel())

            v_hat = Avj.copy()
            for i in range(j+1):
                v_hat -= h[i, j] * v[i]

            h[j+1,j] = np.linalg.norm(v_hat)
            if h[j+1, j] < 1e-15:
                # exact Krylov solution found
                break
            
            v[j+1] = v_hat/h[j+1,j]

            # Apply previous rotations
            for i in range(j):
                temp = cos[i]*h[i, j] - sin[i]*h[i+1, j]
                h[i+1, j] = sin[i]*h[i, j] + cos[i]*h[i+1, j]
                h[i, j] = temp

            # Compute new rotation
            denom = np.hypot(h[j+1, j], h[j, j]) # sqrt(h[j+1,j]*h[j+1,j] + h[j,j]*h[j,j])
            cos[j], sin[j] = h[j, j]/denom, - h[j+1, j]/denom

            # Rotating Hbar (builds R in Q Hbar = R factorisation)
            h[j, j], h[j+1, j] = cos[j]*h[j, j] - sin[j]*h[j+1, j], sin[j]*h[j, j] + cos[j]*h[j+1, j]

            # Apply rotation to g (effectively applying Q to g)
            temp = cos[j] * g[j] - sin[j] * g[j+1]
            g[j+1] = sin[j] * g[j] + cos[j] * g[j+1]
            g[j] = temp

            # Residual norm
            #print(g[j+1])
            #print()
            residual = abs(g[j+1])
            if residual < reltol:# and residual < reduction_tolerance: # and (j == jiter-1): # !!! check if I need to use norm(r0) or norm(b) for the relative tolerance... It might just be a choice.
                print(f"Converged after restart {irestart, j} with residual {residual} (relative tolerance {reltol}).")
                converged = True
                jend = j
                irestarts_convergence[it] = irestart
                j_convergence[it] = j
                iterations_convergence[it] = jiter * irestart + j + 1
                break
            else:
                if residual > norm_oldres: # if residual increased after restart, print a warning (this can happen with GMRES(m) if m is too small or the problem is hard)
                    raise ValueError(f"Residual increased after restart {irestart} (residual: {residual}, old: {norm_oldres}). This may indicate a problem with the solver or the choice of parameters.")
            norm_oldres = residual

        #y = np.linalg.lstsq(h[:cols, :cols], g[:cols], rcond=None)[0] # ! replace this (24-03-2026: I think h refers to R_k in the GMRES paper)

        # h (or R_k) is an upper triangular matrix. To find y, we need to minimise ||g-R_k y ||_2. I believe this means solving R_k y = g. --> We need to backsubstitute to find the y values, this should be easy because R_k is upper triangular.

        y = np.zeros(jiter)
        for ji in range(jend - 1, -1, -1):
            y[ji] = (g[ji] - np.dot(h[ji, ji + 1:], y[ji + 1:]))/h[ji, ji]

        for i in range(len(y)):
            x += y[i] * v[i]

        if converged:
            break

        r0 = b - A(x)

    if residual >= reltol: 
        print(f'GMRES(m) tryopt did not converge within the given iterations (ktotal,jtotal={kiter},{jiter}). Final residual: {residual}, relative tolerance: {reltol}')

    return x


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