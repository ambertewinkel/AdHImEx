"""This file includes function to calculate the amplification factor for the AdHImEx scheme."""


import numpy as np
import sympy as sy


def butcher_AdHImEx_symbolic():
    """This function returns the symbolic Butcher tableau for the AdHImEx method."""

    theta, A_Ex, b_Ex, A_Im, b_Im = sy.symbols('theta A_Ex b_Ex A_Im b_Im', real=True)

    A_Ex = sy.Matrix([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, sy.Rational(1,4), sy.Rational(1,4), 0, 0],[0, sy.Rational(1,6), sy.Rational(1,6), sy.Rational(2,3), 0]])
    b_Ex = sy.Matrix([0, sy.Rational(1,6), sy.Rational(1,6), sy.Rational(2,3), 0])

    A_Im = sy.Matrix([[0, 0, 0, 0, 0],[sy.Rational(1,2), 0, 0, 0, 0],[sy.Rational(1,2), 0, 0, 0, 0],[sy.Rational(1,2), 0, 0, 0, 0],[sy.Rational(1,2), 0, 0, 0, sy.Rational(1,2)]])
    b_Im = sy.Matrix([sy.Rational(1,2), 0, 0, 0, sy.Rational(1,2)])

    A = theta*A_Im + (1 - theta)*A_Ex
    b = theta*b_Im + (1 - theta)*b_Ex

    # Combine A and b such that we have one matrix with both of them in
    return A.col_join(b.T).row_join(sy.Matrix(np.zeros(len(b)+1)))


def butcher_AdImExRK4CN_symbolic():
    """This function returns the symbolic Butcher tableau for the AdHImEx method."""

    theta = sy.symbols('theta', real=True)

    # Explicit and implicit tableaux (each already combined with b) from butcherExIm_AdImExRK4CN_symbolic
    butcherEx, butcherIm = butcherExIm_AdImExRK4CN_symbolic()

    return theta*butcherIm + (1 - theta)*butcherEx


def butcherExIm_AdImExRK4CN_symbolic():
    """This function returns the symbolic explicit and implicit Butcher tableaux for the AdImEx RK4CN method separately (i.e. not weighted by theta),
    each combined with b such that we have one matrix with both A and b in. Needed when the explicit and implicit parts use different spatial discretisations."""

    A_Ex = sy.Matrix([[0, 0, 0, 0, 0],[sy.Rational(1,2), 0, 0, 0, 0],[0, sy.Rational(1,2), 0, 0, 0],[0, 0, 1, 0, 0],[sy.Rational(1,6), sy.Rational(1,3), sy.Rational(1,3), sy.Rational(1,6), 0]])
    b_Ex = sy.Matrix([sy.Rational(1,6), sy.Rational(1,3), sy.Rational(1,3), sy.Rational(1,6), 0])

    A_Im = sy.Matrix([[0, 0, 0, 0, 0],[sy.Rational(1,2), 0, 0, 0, 0],[sy.Rational(1,2), 0, 0, 0, 0],[sy.Rational(1,2), 0, 0, sy.Rational(1,2), 0],[sy.Rational(1,2), 0, 0, sy.Rational(1,2), 0]])
    b_Im = sy.Matrix([sy.Rational(1,2), 0, 0, sy.Rational(1,2), 0])

    # Combine A and b such that we have one matrix with both of them in
    butcherEx = A_Ex.col_join(b_Ex.T).row_join(sy.Matrix(np.zeros(len(b_Ex)+1)))
    butcherIm = A_Im.col_join(b_Im.T).row_join(sy.Matrix(np.zeros(len(b_Im)+1)))
    return butcherEx, butcherIm


def space_AdHImEx_symbolic(kdx):
    """This function returns the symbolic amplification for the fifth-order spatial discretisation, without the division by dx (this is included in C outside of this function)."""
    return -1/20*np.exp(2j*kdx) + 0.5*np.exp(1j*kdx) + 1/3 - np.exp(-1j*kdx) + 0.25*np.exp(-2j*kdx) - 1/30*np.exp(-3j*kdx)
    #return 1./3.*np.exp(1j*kdx) + 0.5 - np.exp(-1j*kdx) + 1./6.*np.exp(-2j*kdx) # temporary third order


def space_quintic_symbolic(kdx):
    """This function returns the amplification for the fifth-order spatial discretisation (consistent with schemes.space_quintic), without the division by dx.
    Same as the (commented out) quintic in space_AdHImEx_symbolic, but independent of the temporary switch there."""
    return -1/20*np.exp(2j*kdx) + 0.5*np.exp(1j*kdx) + 1/3 - np.exp(-1j*kdx) + 0.25*np.exp(-2j*kdx) - 1/30*np.exp(-3j*kdx)


def space_cubic_symbolic(kdx):
    """This function returns the amplification for the third-order spatial discretisation (consistent with schemes.space_cubic), without the division by dx.
    Same as the (temporary) cubic in space_AdHImEx_symbolic, but independent of the temporary switch there."""
    return 1./3.*np.exp(1j*kdx) + 0.5 - np.exp(-1j*kdx) + 1./6.*np.exp(-2j*kdx)


def space_upwind_symbolic(kdx):
    """This function returns the amplification for the first-order upwind spatial discretisation (consistent with schemes.space_upwind), without the division by dx."""
    return 1 - np.exp(-1j*kdx)


def amplificationfactor_symbolic(butcher):
    """This function calculates the symbolic amplification factor for a given Butcher tableau function."""

    SD, C = sy.symbols('SD C')
    nstages = len(butcher()[0,:])-1
    A0 = 1
    for ns in range(1,nstages+2):
        stagesum = 0
        for subns in range(1,ns):
            stagesum += butcher()[ns-1,subns-1]*SD*locals()['A' + str(subns)]
        lhs = 1 + C*butcher()[ns-1,ns-1]*SD
        rhs = 1 - C*stagesum
        locals()['psi' + str(ns)] = rhs/lhs
        locals()['A' + str(ns)] = sy.simplify(locals()['psi' + str(ns)])
    A = sy.simplify(locals()['A' + str(nstages+1)])
    return A


def amplificationfactor_ExIm_symbolic(butcherExIm):
    """This function calculates the symbolic amplification factor for a given function returning the separate explicit and implicit Butcher tableaux,
    with a different spatial discretisation SDEx for the explicit part and SDIm for the implicit part.
    Same recursion as amplificationfactor_symbolic, where the theta-weighted tableau entry times SD, i.e. (theta*AIm + (1-theta)*AEx)*SD,
    is replaced by theta*AIm*SDIm + (1-theta)*AEx*SDEx. Reduces to amplificationfactor_symbolic for SDEx = SDIm."""

    SDEx, SDIm, C = sy.symbols('SDEx SDIm C')
    theta = sy.symbols('theta', real=True)
    butcherEx, butcherIm = butcherExIm()
    nstages = len(butcherEx[0,:])-1
    Astage = {} # amplification factor of each stage
    for ns in range(1,nstages+2):
        stagesum = 0
        for subns in range(1,ns):
            stagesum += ((1 - theta)*butcherEx[ns-1,subns-1]*SDEx + theta*butcherIm[ns-1,subns-1]*SDIm)*Astage[subns]
        lhs = 1 + C*((1 - theta)*butcherEx[ns-1,ns-1]*SDEx + theta*butcherIm[ns-1,ns-1]*SDIm) # explicit diagonal is zero
        rhs = 1 - C*stagesum
        Astage[ns] = sy.simplify(rhs/lhs)
    A = sy.simplify(Astage[nstages+1])
    return A


def calculate_amplification_AdHImEx(C, theta, kdx):
    """This function calculates the amplification factor for the AdHImEx scheme for given C, theta and wavenumber k."""

    C_sy, theta_sy = sy.symbols('C theta', real=True)
    SD = sy.symbols('SD')
    # Get the amplification factor equation with SymPy
    A_eqn = sy.lambdify([C_sy, SD, theta_sy], amplificationfactor_symbolic(butcher_AdHImEx_symbolic))

    # Calculate max values of amplification factor for given C, theta (max over kdx)
    maxAabs = np.zeros((len(theta), len(C)), dtype=float)
    for j in range(len(theta)):
        for i in range(len(C)):
            maxAabs[j, i] = np.abs(A_eqn(C[i], space_AdHImEx_symbolic(kdx), theta[j])).max()
    return maxAabs


def calculate_amplification_AdImExRK4CN(C, theta, kdx, space_Ex=space_cubic_symbolic, space_Im=space_cubic_symbolic):
    """This function calculates the amplification factor for the RK4CN AdImEx scheme for given C, theta and wavenumber k (08-10-2026).
    space_Ex: function, spatial discretisation for the explicit part of the Butcher tableau (default space_AdHImEx_symbolic)
    space_Im: function, spatial discretisation for the implicit part of the Butcher tableau (default space_AdHImEx_symbolic)
    By default uses the same cubic spatial discretisation for both the explicit and implicit parts."""

    C_sy, theta_sy = sy.symbols('C theta', real=True)
    SDEx, SDIm = sy.symbols('SDEx SDIm')
    # Get the amplification factor equation with SymPy (separate spatial discretisations for the explicit and implicit parts)
    A_eqn = sy.lambdify([C_sy, SDEx, SDIm, theta_sy], amplificationfactor_ExIm_symbolic(butcherExIm_AdImExRK4CN_symbolic))

    # Calculate max values of amplification factor for given C, theta (max over kdx)
    maxAabs = np.zeros((len(theta), len(C)), dtype=float)
    for j in range(len(theta)):
        for i in range(len(C)):
            maxAabs[j, i] = np.abs(A_eqn(C[i], space_Ex(kdx), space_Im(kdx), theta[j])).max()
    return maxAabs


def calculate_amplification_AdImExRK4CN_quinticEx_cubicIm(C, theta, kdx):
    """This function calculates the amplification factor for the RK4CN AdImEx scheme with a fifth-order (quintic) spatial discretisation for the explicit part
    and a third-order (cubic) spatial discretisation for the implicit part, for given C, theta and wavenumber k. Consistent with schemes.AdImExRK4CN_quinticEx_cubicIm."""
    return calculate_amplification_AdImExRK4CN(C, theta, kdx, space_Ex=space_quintic_symbolic, space_Im=space_cubic_symbolic)


def calculate_amplification_AdImExRK4CN_quinticEx_upwindIm(C, theta, kdx):
    """This function calculates the amplification factor for the RK4CN AdImEx scheme with a fifth-order (quintic) spatial discretisation for the explicit part
    and a first-order upwind spatial discretisation for the implicit part, for given C, theta and wavenumber k. Consistent with schemes.AdImExRK4CN_quinticEx_upwindIm."""
    return calculate_amplification_AdImExRK4CN(C, theta, kdx, space_Ex=space_quintic_symbolic, space_Im=space_upwind_symbolic)
