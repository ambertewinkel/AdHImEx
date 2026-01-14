"""This file includes function to calculate the amplification factor for the AdHImEx scheme."""

import numpy as np
import sympy as sy


def butcher_AdHImEx_symbolic():
    """This function returns the Butcher tableau for the AdHImEx method."""
    # Defining the Butcher tableaus - theta is implicitness. theta = 0 is explicit, theta = 1 is implicit

    theta, A_Ex, b_Ex, A_Im, b_Im = sy.symbols('theta A_Ex b_Ex A_Im b_Im', real=True)

    A_Ex = sy.Matrix([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, sy.Rational(1,4), sy.Rational(1,4), 0, 0],[0, sy.Rational(1,6), sy.Rational(1,6), sy.Rational(2,3), 0]])
    b_Ex = sy.Matrix([0, sy.Rational(1,6), sy.Rational(1,6), sy.Rational(2,3), 0])

    A_Im = sy.Matrix([[0, 0, 0, 0, 0],[sy.Rational(1,2), 0, 0, 0, 0],[sy.Rational(1,2), 0, 0, 0, 0],[sy.Rational(1,2), 0, 0, 0, 0],[sy.Rational(1,2), 0, 0, 0, sy.Rational(1,2)]])
    b_Im = sy.Matrix([sy.Rational(1,2), 0, 0, 0, sy.Rational(1,2)])

    A = theta*A_Im + (1 - theta)*A_Ex
    b = theta*b_Im + (1 - theta)*b_Ex

    # Combine A and b such that we have one matrix with both of them in
    return A.col_join(b.T).row_join(sy.Matrix(np.zeros(len(b)+1)))
    

def space_AdHImEx_symbolic(kdx): # assumes exp() form of solution for VNSA
    """This function returns the amplification factor for the derivative of the fifth302 method without the division by dx (this is included in C outside of this function)."""
    return -1/20*np.exp(2j*kdx) + 0.5*np.exp(1j*kdx) + 1/3 - np.exp(-1j*kdx) + 0.25*np.exp(-2j*kdx) - 1/30*np.exp(-3j*kdx)



def amplificationfactor_symbolic(butcher):
    # assume uniform velocity in space and time and a uniform grid.

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

