import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import numpy as np
from collocation import *


    

def cbc_coll():
    ## Must have degree >= smoothness + 2; eg. piecewise-constant
    ## (deg.1) can be at most -1 smooth (discontinuous)

    MESHSIZE = 4
    DEGREE = 2 ### NOT polynomial order! Degree 3 = 3 coefficients = quadratic, NOT cubic
    SMOOTHNESS = 0  ### -1 = non-continuous; 0 = continuous; 1 = continuous + continuous 1st derivative, etc

    ts = np.linspace(0, 2*np.pi, 100)
    ys = np.sin(ts) #+ np.sin(3*ts)
    signal = np.c_[ts, ys].T
    coeffs_guess = np.random.rand(DEGREE * MESHSIZE)*2 - 1

    mesh = np.linspace(0, 2*np.pi, MESHSIZE+1)
    discretisor = KroghCBCCollocation(mesh, SMOOTHNESS)

    soln = scipy.optimize.root(lambda x: discretisor(x, signal), coeffs_guess)
    print(soln)

    plotts = np.linspace(0,2*np.pi,1000)
    _, ax = plt.subplots()
    ax.plot(ts, ys, label="Signal")
    ax.plot(plotts, discretisor.evaluate_solution(plotts, soln.x), label="Model")
    ax.plot(plotts, discretisor.evaluate_derivatives(plotts, soln.x, 1), label="Gradient")
    ax.scatter(mesh, discretisor.evaluate_solution(mesh, soln.x))
    ax.legend()
    plt.show()

def bvp_coll():
    MESHSIZE = 4
    DEGREE = 4
    START = 0
    END = 1

    coeffs_guess = np.zeros(2* DEGREE * MESHSIZE)
    mesh = np.linspace(START, END, MESHSIZE+1)
    discretisor = KroghBVP(mesh)

    def bvp(x, y):
        return np.vstack((y[1], -(2*np.pi)**2*y[0]))

    def boundary_conds(start, end):
        return np.array([start[0], end[1]-1])

    zero_problem = discretisor.get_default_zero_problem(bvp, boundary_conds, 2)
    soln = scipy.optimize.root(zero_problem, coeffs_guess)
    print(soln)
    soln_coeffs = soln.x.reshape((2, -1))

    plotts = np.linspace(START, END, 1000)
    _, ax = plt.subplots()
    ax.plot(plotts, discretisor.evaluate_solution(plotts, soln_coeffs)[0], label="Model x")
    ax.plot(plotts, discretisor.evaluate_solution(plotts, soln_coeffs)[1], label="Model y")
    ax.scatter(mesh, discretisor.evaluate_solution(mesh, soln_coeffs)[0])
    ax.legend()
    plt.show()

def bvp_coll_1d():
    MESHSIZE = 4
    ORDER = 2
    mesh = np.linspace(0, 1, MESHSIZE+1)
    discretisor = KroghBVP(mesh)
    coeffs_guess = np.zeros(MESHSIZE*ORDER + 1)

    def bvp(t, x, par):
        return par*t

    def zero_problem(coeffs):
        par, soln = coeffs[0], coeffs[1:]
        start, end = discretisor.evaluate_solution([0, 1], soln)
        boundary_conds = np.array([1-start, 2-end])
        return np.r_[discretisor(soln, lambda t, x: bvp(t, x, par)), boundary_conds]

    soln = scipy.optimize.root(zero_problem, coeffs_guess)
    print(soln)

    plotts = np.linspace(0,1)
    _,ax = plt.subplots()
    ax.plot(plotts,discretisor.evaluate_solution(plotts, soln.x[1:]), label="Model")
    ax.scatter(mesh, discretisor.evaluate_solution(mesh, soln.x[1:]), label="Mesh")
    ax.legend()
    plt.show()
        

def main():
    bvp_coll()
   # bvp_coll_1d()
   # cbc_coll()


if __name__ == "__main__":
    main()
