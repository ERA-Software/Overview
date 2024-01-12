import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from FORM_Sobol_indices import FORM_Sobol_indices

"""
---------------------------------------------------------------------------
Optimization using the fmincon function (scipy.optimize.minimize() in Python)
---------------------------------------------------------------------------
Created by:
Matthias Willer
Felipe Uribe
Luca Sardi
Daniel Koutas
Max Ehre
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
First version: 2018-05
---------------------------------------------------------------------------
Changelog 
Changelog 
2022-04: * included first order and total-effect Sobol' indices computation
2023-05: * added optional inputs to control design point search parameters
         * added option to pass analytical LSF gradient
---------------------------------------------------------------------------
Comment:
* The FORM method uses a first order approximation of the LSF and is
  therefore not accurate for non-linear LSF's
---------------------------------------------------------------------------
Input:
* g                     : limit state function
* dg                    : anonynmous function containing the analytical 
                          gradient of the limit state function (optional)
* distr                 : ERANataf-Object containing the distribution
* sensitivity_analysis  : implementation of sensitivity analysis: 
                          1 - perform, 
                          0 - not perform

Optional (optimization parameters for design point search):

* u0                   : initial guess 
* tol                  : tolerance 
* maxit                : maximum number of iterations
---------------------------------------------------------------------------
Output:
* u_star : design point in the standard space
* x_star : design point in the original space
* beta   : reliability index
* Pf     : probability of failure
* S_F1   : vector of first-order indices
* S_F1_T : vector of total-effect indices
---------------------------------------------------------------------------
"""

def FORM_fmincon(g, dg, distr,sensitivity_analysis, u0 = 1, tol = 1e-6, maxit = 500):

    #  initial check if there exists a Nataf object
    if not(isinstance(distr, ERANataf)):
        raise RuntimeError('Incorrect distribution. Please create an ERANataf object!')

    d = len(distr.Marginals)

    # objective function
    dist_fun = lambda u: np.linalg.norm(u)

    # arameters of the minimize function
    u0 = u0*np.ones((1,d))  # initial search point

    # nonlinear constraint: H(u) <= 0
    H    = lambda u: g(distr.U2X(u))

    # method for minimization
    alg = 'SLSQP'

    #  minimize with constraints without analytical gradients
    if dg == []:
        cons = ({'type': 'ineq', 'fun': lambda u: -H(u)})
        res = sp.optimize.minimize(dist_fun, u0, constraints=cons, method=alg, options = {'maxiter' : maxit, 'ftol' : tol})

    #  minimize with constraints using analytical gradients dg
    else:
        
        def dgu(u):
            [x, J] = distr.U2X(u, Jacobian = True)
            grad   = J @ dg(x)
            return grad.ravel()
        
        cons = ({'type': 'ineq', 'fun': lambda u: -H(u), 'jac': lambda u: -dgu(u)})
        res = sp.optimize.minimize(dist_fun, u0, constraints=cons, method=alg, options={'maxiter' : maxit, 'ftol' : tol})

    # unpack results
    u_star = res.x
    beta   = res.fun
    it     = res.nit

    # compute design point in orignal space and failure probability
    x_star = distr.U2X(u_star)
    Pf     = sp.stats.norm.cdf(-beta)

    #  sensitivity analysis
    if sensitivity_analysis == 1:
        [S_F1, S_F1_T, exitflag, errormsg] = FORM_Sobol_indices(u_star/beta, beta, Pf)
    else:
        S_F1 = []
        S_F1_T = []


    # print results
    print('\n*scipy.optimize.minimize() with ', alg, ' Method\n')
    print(' ', it, ' iterations... Reliability index = ', beta, ' --- Failure probability = ', Pf, '\n\n')

    # print first order and total-effect indices
    if sensitivity_analysis == 1:
        if hasattr(distr, 'Marginals'):
            if not (distr.Rho_X == np.eye(len(distr.Marginals))).all():
                print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Results of sensitivity analysis do not apply for dependent inputs.")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
        if exitflag == 1:
            print(" First order indices:")
            print(' ', S_F1, '\n')
            print(" Total-effect indices:")
            print(' ', S_F1_T, '\n\n')
        else:
            print('Sensitivity analysis could not be performed, because:')
            print(errormsg)


    return u_star, x_star, beta, Pf, S_F1, S_F1_T
