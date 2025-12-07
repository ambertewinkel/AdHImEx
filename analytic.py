
import numpy as np
import logging

# !!! needs reducing


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


def sine_yshift(x, xmax, u=0., t=0., shifty=10.0, ampl=0.5, shiftx=0.):
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



def combi(x, xmax, u=0., t=0., shift=0., ampl=1., a=0., b=0.5, c=0.6, d=0.8):
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
    shift: float, shift of the cosine bell and top hat
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
        psi = shift + ampl*np.where((x0 >= a) & (x0 <= b), 0.5*(1 - np.cos(2*np.pi*(x0-a)/(b-a))), 0.)
    else:
        psi = shift + ampl*np.where((x0 >= a) | (x0 <= b), 0.5*(1 - np.cos(2*np.pi*(x0-a+xmax)/(b-a+xmax))), 0.)

    # Define nonzero region of the top hat
    if c < d:
        psi = shift + ampl*np.where((x0 >= c) & (x0 <= d), 1., psi)
    else:
        psi = shift + ampl*np.where((x0 >= c) | (x0 <= d), 1., psi)

    return psi

def velocity_varying_space701(x, l=2.*np.pi):
    """This function returns a velocity field that is varying in space. This gives a Courant number always lower than 1 for dt=0.01 and dx=0.025"""
    #u = 5.5 + 4.5*(np.sin(2*np.pi*(x+0.25)))
    #u = 0.5 + 9.5*np.sin(np.pi*(x+0.5))**8
    #u = 0.5 + 9.5*np.sin(np.pi*(x+0.5))**4
    u = 0.2 + 9.8*np.sin(np.pi*(x+0.5))**4
    return u

def sine_xyshiftampl3(x, xmax, u=0., t=0., shifty=50., ampl=50., shiftx=0.3):
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

def velocity_varying_time_space_swift_2Dnondiv(nt, dt, x, L=1000., U=10., T=100.):
    """Velocity setting used in the Bendall and Kent (2025) SWIFT paper. 
    u0   : set for AdImEx time stepping
    L = 1000. #m
    u0 = 10. #ms-1
    T = 100. #s    
    """
    u_x = np.zeros((nt, len(x)))
    for it in range(nt):
        t = (it+0.5)*dt # +0.5 for velocity at the half level in time for second-order accuracy
        x_prime = x + 0.5*L - U*t
        y_prime = 0.75*L - U*t
        u_x[it] = U*np.sin(np.pi*x_prime/L)*np.sin(np.pi*x_prime/L)*np.sin(2.*np.pi*y_prime/L)*np.cos(np.pi*t/T) + U

    return u_x


def sine_swift(x, L, u_dummy=0., t_dummy=0.):
    """Initial condition for the sine wave field (m in paper) in Bendall and Kent (2025) SWIFT paper (see Appendix B, sine wave initial condition).
    x    : points in domain to calculate m for
    xmax : domain size (L_x in paper)
    """
    M = 0.5 # kg/kg
    return M + M*np.sin(2.*np.pi*x/L)