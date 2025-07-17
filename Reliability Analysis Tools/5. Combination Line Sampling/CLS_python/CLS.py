"""
---------------------------------------------------------------------------
Created by:
Iason Papaioannou
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2021-06
---------------------------------------------------------------------------
Based on:
1. "Combination line sampling for structural reliability analysis" and
    by Iason Papaioannou and Daniel Straub
    Structural Safety 2021, 88. Jg., S. 102025.
---------------------------------------------------------------------------
Comments:
* Here we focus on the case where the failure domain is concentrated in a
  distinct region of the outcome space. Problems with multiple failure
  modes often require accounting for several important directions.
* The potential of this method is limited to problems with low to moderate
  dimensions as it gets more difficult to achieve substantial improvements
  with an increasing sampling space dimension.
---------------------------------------------------------------------------
Input:
* u_star: a point in U-space that defines the initial direction
* N: the  desired number of samples,
* g_fun:  the limit-state function (LSF)
* distr:  the input distribution
---------------------------------------------------------------------------
Output:
* Pf_LS:  estimate of the failure probability
* cv_LS:  estimate of the coefficient of variation of the probability estimate
* u_new:  final selected point in U-space (estimate for the FORM design point)
* x_new:  same point in X-space
* n_eval: the total number of LSF-evaluations
"""

import numpy as np
from scipy import stats
from find_root import find_root
from find_root_RS import find_root_RS

def CLS(u_star, N, g_fun, distr):

    ## initial check if there exists a Nataf object

    # transform to the standard Gaussian space
    if hasattr(distr, 'Marginals'):             # use Nataf transfrom (dependence)
        n = len(distr.Marginals)              # number of random variables (dimension)
        u2x = lambda u: distr.U2X(u)            # from u to x

    else:       # use distribution information for the transformation (independence)
                # Here we are assuming that all the parameters have the same distribution !!!
                # Adjust accordingly otherwise or use an ERANataf object
        try:
            n = len(distr)                    # number of random variables (dimension)
            u2x = lambda u: distr[0].icdf(stats.norm.cdf(u))     # from u to x
        except TypeError:
            n = 1
            u2x = lambda u: distr.icdf(stats.norm.cdf(u))       # from u to x


    ## Initialization

    # LSF in standard space
    G_LSF = lambda u: g_fun(u2x(u).reshape(1,-1))

    # number of LSF evaluations
    n_eval = 0

    # fixed number of LSF evaluations for the find_root_RS function in the
    # first run and subsequent runs in the for-loop
    n_LSF_1 = 10
    n_LSF_subs = 5

    # select root finding function as either 'find_root' or 'find_root_RS'
    rt_fcn = 'find_root'

    if not ((rt_fcn == 'find_root') or (rt_fcn == 'find_root_RS')):
        raise RuntimeError("Select either 'find_root' or 'find_root_RS' as your root finding function")


    ## CLS

    # Initial direction
    alpha = u_star / np.linalg.norm(u_star)

    # Find root in initial direction
    G_LSF1D = lambda d: G_LSF(d*alpha)
    if rt_fcn == 'find_root':
        d0 = 1
        [c0, n_eval1, exitflag] = find_root(G_LSF1D, d0)
        n_eval = n_eval + n_eval1
    elif rt_fcn == 'find_root_RS':
        d0 = 3
        [c0,exitflag] = find_root_RS(G_LSF1D, d0, n_LSF_1)
        n_eval = n_eval + n_LSF_1

    # If no root was found, penalize with a high value (corresponding to ~0
    # contribution)
    if exitflag == 0:
        c0 = 10

    # Loop variables
    l = 0
    k = 0
    d = np.zeros(N)
    pfi = np.zeros(N)
    pfk = []
    var_LSk = []
    wk = []

    for i in range(N):

        l += 1

        # Generate random sample vector
        u = stats.norm.rvs(size=(1,n))
        # Orthogonalization to search direction
        uper = u - np.dot(u, alpha.T)*alpha

        # Solve root finding problem
        G_LSF1D = lambda d: G_LSF(uper+d*alpha)
        if rt_fcn == 'find_root':
            [d1, n_eval1, exitflag] = find_root(G_LSF1D, c0)
            n_eval = n_eval + n_eval1
        elif rt_fcn == 'find_root_RS':
            [d1, exitflag] = find_root_RS(G_LSF1D, c0, n_LSF_subs)
            n_eval = n_eval + n_LSF_subs

        # Save found distance or penalize with a high value
        if exitflag == 1:
            d[i] = d1
        else:
            d[i] = 10

        pfi[i] = stats.norm.cdf(-abs(d[i]))

        # Calculate the distance from the origin to the intersection of the
        # line parallel to alpha with the failure surface
        c1 = np.linalg.norm(uper+d[i]*alpha)

        # If the distance c1 is shorter than the previously found one, do
        # corresponding updates
        if c1 < c0:
            # Line Sampling (LS) estimator for direction alpha
            pfk.append(np.mean(pfi[i-l+1:i+1]))
            # Variance estimator
            var_LSk.append(1/l*(1/l*np.sum(pfi[i-l+1:i+1]**2)-pfk[k]**2))
            # Heuristic weights
            wk.append(l*stats.norm.cdf(-abs(c0)))

            if (var_LSk[k] == 0) or (l <= 2):
                var_LSk[k]= pfk[k]**2

            # Update importance direction and values
            alpha = 1/c1*(uper+d[i]*alpha)
            c0 = c1

            l = 0
            k += 1

    if l > 0:
        # LS estimator for last direction
        pfk.append(np.mean(pfi[i-l+1:i+1]))
        # Variance estimate
        var_LSk.append(1/l*(1/l*np.sum(pfi[i-l+1:i+1]**2)-pfk[k]**2))
        # Heuristic weights
        wk.append(l*stats.norm.cdf(-abs(c0)))

        if (var_LSk[k] == 0) or (l <= 2):
            var_LSk[k]= pfk[k]**2

    # Normalization
    wk_norm = wk / np.sum(wk)

    # Variance estimate is the weighted sum of the individual variance
    # estimates
    var_LS = np.sum((wk_norm**2)*var_LSk)

    # Failure probability estimate is the weighted sum of the individual
    # failure estimates
    Pf_LS = np.sum(pfk*wk_norm)

    # Coefficient of variation estimate
    cv_LS = np.sqrt(var_LS) / Pf_LS

    # Update estimate for the FORM design point in U-space and X-space
    u_new = c0*alpha
    x_new = u2x(u_new)


    return [Pf_LS, cv_LS, u_new, x_new, n_eval]
