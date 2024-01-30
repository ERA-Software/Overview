import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist

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
Ivan Olarte Rodriguez
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
2023-09: * Modified the inclusion of Sobol Indices by having the FORM alpha
           indices as direct output of the function
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

Optional (optimization parameters for design point search):

* u0                   : initial guess 
* tol                  : tolerance 
* maxit                : maximum number of iterations
---------------------------------------------------------------------------
Output:
* u_star : design point in the standard space
* x_star : design point in the original space
* beta   : reliability index
* alpha  : FORM sensitivity indices
* Pf     : probability of failure


---------------------------------------------------------------------------
References:
1. "Structural reliability under combined random load sequences."
   Rackwitz, R., and B. Fiessler (1979).
   Computers and Structures, 9.5, pp 489-494
---------------------------------------------------------------------------
"""

def FORM_HLRF(g,dg,distr:ERANataf, u0:float = 0.1, 
              tol:float = 1e-6, maxit:int = 500):

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
            del test_dg
        except:
            # use finite differences if autograd fails
            fd_grad = True
            epsil     = lambda gg: 1e-04 * np.maximum(np.abs(gg),1e-06)
            dg      = lambda xg,ggg: (g( np.add( xg, np.diag(epsil(ggg)*np.ones(d)) ) ) - ggg) / epsil(ggg)

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
            DH_uk      = np.dot( J, dg(xk,H_uk) )
        else:
            DH_uk      = np.dot( J , dg(xk) ) 
    
        norm_DH_uk = np.linalg.norm(DH_uk,2)

        if np.isnan(norm_DH_uk):
            raise RuntimeError("The Jacobian is a nan value. Check the inputs!")
        

        alpha      = DH_uk/norm_DH_uk
        alpha      = alpha.ravel()

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

    # print results
    print('\n*FORM with HLRF algorithm\n')
    print(' ', k+1, ' iterations... Reliability index = ', beta, ' --- Failure probability = ', Pf, '\n')
    # Check if final design point lies on LSF
    if not(abs(g(x_star)) <= 1e-6):
        print('Warning! HLRF may have converged to wrong value! The LSF of the design point is: ', g(x_star))
    print('\n')


    return u_star, x_star, beta, alpha, Pf 
