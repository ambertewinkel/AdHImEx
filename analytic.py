"""This file includes functions for the l2 norm, initial conditions, and analytic solutions used."""


import numpy as np


def l2norm(field, analytic, dxc):
    """This calculates the l2 norm from an output field compared to the analytic solution.
    field : 1D array of floats, output field from the numerical scheme
    analytic : 1D array of floats, analytic solution
    dxc : 1D array of floats, cell-centred widths
    """
    numerator = np.sum(dxc*(field - analytic)*(field - analytic))
    denominator = np.sum(dxc*analytic*analytic)
    return np.sqrt(numerator/(denominator + 1.e-16))


def sine(x, xmax, u=0., t=0., shifty=0.5, ampl=0.5, shiftx=0.):
    """This function returns an array from input array x advected by velocity u for a time t.
    The initial condition has values from the function y = shifty + ampl*sin(2pi(x-shiftx)/xmax).
    --- Input ---
    x   : 1D array of floats, points to calculate the result of the function for
    xmax: float, domain size
    u   : float or 1D array of floats, velocity
    t   : float, total time
    shifty: float, y shift of the sine wave
    ampl : float, amplitude of the sine wave
    shiftx: float, x shift of the sine wave
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """
    psi = np.zeros(len(x))
    x0 = (x - u*t)%xmax
    psi = shifty + ampl*np.sin(2*np.pi*(x0-shiftx)/xmax)
    return psi


def combi(x, xmax, u=0., t=0., shifty=0., ampl=1., a=0., b=0.5, c=0.6, d=0.8):
    """
    This function returns an array from input array x and constants a and b advected 
    by velocity u for a time t. The initial condition has output values 1 in the range
    of the domain enclosed by a and b and outside of this region, 0. This function is 
    a combination of the cosinebell and tophat functions.
    --- Input ---
    x   : 1D array of floats, points to calculate the result of the function for
    xmax: float, domain size
    u   : float or 1D array of floats, velocity
    t   : float, total time
    shifty: float, y shift of the cosine bell and top hat
    ampl : float, amplitude of the cosine bell and top hat
    a   : float, left boundary of cosine bell
    b   : float, right boundary of cosine bell
    c   : float, left boundary of top hat
    d   : float, right boundary of top hat
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """
    psi = np.zeros(len(x))
    x0 = (x - u*t)%xmax
    
    # Define nonzero region of the cosine bell
    if a < b:
        psi = shifty + ampl*np.where((x0 >= a) & (x0 <= b), 0.5*(1 - np.cos(2*np.pi*(x0-a)/(b-a))), 0.)
    else:
        psi = shifty + ampl*np.where((x0 >= a) | (x0 <= b), 0.5*(1 - np.cos(2*np.pi*(x0-a+xmax)/(b-a+xmax))), 0.)

    # Define nonzero region of the top hat
    if c < d:
        psi = shifty + ampl*np.where((x0 >= c) & (x0 <= d), 1., psi)
    else:
        psi = shifty + ampl*np.where((x0 >= c) | (x0 <= d), 1., psi)

    return psi


def velocity_varying_space(x):
    """This function returns the velocity field that is solely varying in space."""
    return 0.2 + 9.8*np.sin(np.pi*(x+0.5))**4


def velocity_varying_time_space_swift(nt, dt, x, L=1000., U=10., T=100.):
    """Returns the velocity varying in space and time, 1D version of the 2D nondivergent winds in the Bendall and Kent (2025) SWIFT paper. 
    nt : number of time steps
    dt : time step size
    x  : points in domain to calculate velocity for
    L  : domain size
    U  : velocity coefficient
    T  : period of oscillation
    """
    u_x = np.zeros((nt, len(x)))
    for it in range(nt):
        t = (it+0.5)*dt # +0.5 for velocity at the half level in time for second-order accuracy
        x_prime = x + 0.5*L - U*t
        y_prime = 0.75*L - U*t
        u_x[it] = U*np.sin(np.pi*x_prime/L)*np.sin(np.pi*x_prime/L)*np.sin(2.*np.pi*y_prime/L)*np.cos(np.pi*t/T) + U
    return u_x


def sine_swift(x, L):
    """Initial condition for the sine wave field (m in paper) in Bendall and Kent (2025) SWIFT paper (see Appendix B, sine wave initial condition).
    x    : points in domain to calculate m for
    L    : domain size
    """
    M = 0.5 # kg/kg
    return M + M*np.sin(2.*np.pi*x/L)