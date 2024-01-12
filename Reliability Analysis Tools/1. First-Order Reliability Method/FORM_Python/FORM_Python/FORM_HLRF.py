import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from FORM_Sobol_indices import FORM_Sobol_indices

try:
    import autograd.numpy as np
    from autograd import grad
except:
    import numpy as np

"""
---------------------------------------------------------------------------
HLRF function
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
2022-04: * included first order and total-effect Sobol' indices computation
2023-05: * added optional inputs to control design point search parameters
         * include autograd and finite difference methods for LSF gradient
---------------------------------------------------------------------------
Comment:
* The FORM method uses a first order approximation of the LSF and is
  therefore not accurate for non-linear LSF's
---------------------------------------------------------------------------
Input:
* g                     : limit state function
* dg                    : anonynmous function containing the analytical 
                          gradient of the limit state function (optional)
                          - if not provided, finite differences are used
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
References:
1. "Structural reliability under combined random load sequences."
   Rackwitz, R., and B. Fiessler (1979).
   Computers and Structures, 9.5, pp 489-494
---------------------------------------------------------------------------
"""

def FORM_HLRF(g,dg,distr,sensitivity_analysis, u0 = 1, tol = 1e-6, maxit = 500):

    #  initial check if there exists a Nataf object
    if not(isinstance(distr, ERANataf)):
        raise RuntimeError('Incorrect distribution. Please create an ERANataf object!')

    d = len(distr.Marginals)    # number of random variables (dimension)

    # check if analytical gradients are passed in dg; if not, use finite differnces

    fd_grad = False

    if dg == []:
        
        try:     
            # check if autograd works       
            dg = grad(g)
            test_dg = dg(distr.U2X(np.random.randn(1,d)))
        except:
            # use finite differences if autograd fails
            fd_grad = True
            eps     = lambda gg: 0.0001 * max(np.abs(gg),0.000001)
            dg      = lambda xg: (g(xg[0] + np.diag(eps(xg[1])*np.ones(d))) - xg[1]) / eps(xg[1])

    #  initialization
    u     = u0*np.ones([maxit+2,d])
    beta  = np.zeros(maxit)

    #  HLRF method
    for k in range(maxit):
        # 0. Get x and Jacobi from u (important for transformation)
        [xk, J] = distr.U2X(u[k,:], Jacobian=True)

        # 1. evaluate LSF at point u_k
        H_uk = g(xk)

        # 2. evaluate LSF gradient at point u_k and direction cosines

        if fd_grad:
            DH_uk      = J @ dg([xk,H_uk])
        else:
            DH_uk      = J @ dg(xk)
    
        norm_DH_uk = np.linalg.norm(DH_uk)
        alpha      = DH_uk/norm_DH_uk
        alpha      = alpha.squeeze()

        # 3. calculate beta
        beta[k] = - u[k,:] @ alpha + H_uk/norm_DH_uk

        # 4. calculate u_{k+1}
        u[k+1,:] = -beta[k]*alpha

        # next iteration
        if (np.linalg.norm(u[k+1,:]-u[k,:]) <= tol):
            break

    # delete unnecessary data
    u = u[:k+2,:]

    # compute design point, reliability index and Pf
    u_star = u[-1,:]
    x_star = distr.U2X(u_star)
    beta   = beta[k]
    Pf     = sp.stats.norm.cdf(-beta)

    #  sensitivity analysis
    if sensitivity_analysis == 1:
        if hasattr(distr, 'Marginals'):
            if not (distr.Rho_X == np.eye(len(distr.Marginals))).all():
                print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Results of sensitivity analysis do not apply for dependent inputs.")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
        [S_F1, S_F1_T, exitflag, errormsg] = FORM_Sobol_indices(alpha, beta, Pf)
    else:
        S_F1 = []
        S_F1_T = []

    # print results
    print('\n*FORM with HLRF algorithm\n')
    print(' ', k+1, ' iterations... Reliability index = ', beta, ' --- Failure probability = ', Pf, '\n')
    # Check if final design point lies on LSF
    if not(abs(g(x_star)) <= 1e-6):
        print('Warning! HLRF may have converged to wrong value! The LSF of the design point is: ', g(x_star))
    print('\n')

    # print first order and total-effect indices
    if sensitivity_analysis == 1:
        if exitflag == 1:
            print(" First order indices:")
            print(' ', S_F1, '\n')
            print(" Total-effect indices:")
            print(' ', S_F1_T, '\n\n')
        else:
            print('Sensitivity analysis could not be performed, because:')
            print(errormsg)

    return u_star, x_star, beta, Pf, S_F1, S_F1_T
