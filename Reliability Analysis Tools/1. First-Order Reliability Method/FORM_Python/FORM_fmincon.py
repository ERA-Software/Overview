import scipy as sp
from ERANataf import ERANataf

try:
    import autograd.numpy as np
    from autograd import grad
except:
    import numpy as np

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
"""

def FORM_fmincon(g, dg, distr:ERANataf, u0:float = 0.1, 
                 tol:float = 1e-6, maxit:int = 500):

    #  initial check if there exists a Nataf object
    if not(isinstance(distr, ERANataf)):
        raise RuntimeError('Incorrect distribution. Please create an ERANataf object!')

    d = len(distr.Marginals)

    # objective function
    dist_fun = lambda u: np.linalg.norm(np.ravel(u),2)

    # parameters of the minimize function
    u0 = np.squeeze(u0*np.ones((1,d)))  # initial search point

    # nonlinear constraint: H(u) <= 0
    H    = lambda u: g(distr.U2X(u))

    # method for minimization
    alg = 'SLSQP'

    #  minimize with constraints without analytical gradients
    if dg == []:
        
        fd_grad = False
        try:     
            # check if autograd works       
            dg = grad(g)
            test_dg = dg(distr.U2X(np.random.randn(1,d)))
            del test_dg

            def dgu(u):
                [x, J] = distr.U2X(u, Jacobian = True)
                grad   = np.dot(J , dg(x))
                return grad.ravel()
        except:
            # use finite differences if autograd fails
            fd_grad = True
            epsil     = lambda gg: 1e-04 * np.maximum(np.abs(gg),1e-06)
            dg      = lambda xg,ggg: (g( np.add( xg, np.diag(epsil(ggg)*np.ones(d)) ) ) - ggg) / epsil(ggg)
        
            def dgu(u):
                [x, J] = distr.U2X(np.ravel(u), Jacobian = True)
                
                val = g(x)
                derv = dg(x,np.ravel(val))
                grad   = np.dot(J , derv )

                return grad.ravel()
        
        # Run the optimization function without the gradient information
        cons = ({'type': 'ineq', 'fun': lambda u: -H(u)})
        res = sp.optimize.minimize(dist_fun, u0, constraints=cons, method=alg, options = {'maxiter' : maxit, 'ftol' : tol})

    #  minimize with constraints using analytical gradients dg
    else:
        
        def dgu(u):
            [x, J] = distr.U2X(u, Jacobian = True)
            grad   = np.dot(J , dg(x))
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

    # Compute the FORM alpha-coefficients
    #-dG_dU(u_star)./(norm(dG_dU(u_star),2));
    alpha = dgu(np.ravel(u_star))/np.linalg.norm(dgu(np.ravel(u_star)),2)

    # print results
    print('\n*scipy.optimize.minimize() with ', alg, ' Method\n')
    print(' ', it, ' iterations... Reliability index = ', beta, ' --- Failure probability = ', Pf, '\n\n')


    return u_star, x_star, beta,alpha, Pf
