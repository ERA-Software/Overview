## Compute first order Sobol' indices from failure samples
"""
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2022-04
2022-10 (Max Ehre): 
Fixed bad performance for ex1_Luyi_paper.m by extending 
integration domain in line 147 (pm 5 dev std -> pm 15 std dev)
---------------------------------------------------------------------------
Based on:
1."Global reliability sensitivity estimation based on failure samples"
   Luyi Li, Iason Papaioannou & Daniel Straub.
   Structural Safety 81 (2019) 101871.
2."Kernel Estimator and Bandwidth Selection for Density and its
   Derivatives"
   Arsalane Chouaib Guidoum.
   Department of Probabilities and Statistics, University of Science and
   Technology, Houari Boumediene, Algeria (2015)
}
---------------------------------------------------------------------------
Comments:
* The upper bound of fminbnd is set to a multiple of the maximum distance
  between the failure samples, because Inf is not handled well.
* Significantly dominates computation time at higher number of samples
* User can trigger plot of posterior kernel density estimations as well as
  maximum likelihood cross validation dependent on the bandwidth (optimal
  bandwidth marked as star)
---------------------------------------------------------------------------
Input:
* samplesX: failure samples
* Pf      : estimated failure probability
* distr   : ERADist or ERANataf object containing the infos about the random
            variables
---------------------------------------------------------------------------
Output:
* S_F1      : vector of first order sensitivity indices
* exitflag  : flag whether method was successful or not
* errormsg  : error message describing what went wrong
---------------------------------------------------------------------------
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def Sim_Sobol_indices(samplesX, Pf, distr):

    ## Initialization
    # Check for ERANataf or ERADist
    if hasattr(distr, 'Marginals'):
        dist = distr.Marginals
        # check for dependent variables
        if not (distr.Rho_X == np.eye(len(distr.Marginals))).all():
            print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Results of sensitivity analysis should be interpreted with care for dependent inputs.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
    else:
        dist = distr
    # remove multiple and Inf failure samples for optimization
    samplesX = samplesX[~(~np.isfinite(samplesX)).any(axis=1), :]
    _, idcs_x, idcs_u, counts = np.unique(samplesX, axis=0, return_index=True, return_inverse=True, return_counts=True )

    N, d = samplesX.shape    # Number of RVs and samples

    # Solver parameters
    Maxiter = 500       # default of fminbnd is 500
    Tolx    = 1e-4      # default of fminbnd is 1e-4
    lb      = 0         # lower boundary for search interval on w
    # heuristic upper boundary for search interval on w; Inf not well handled
    ub      = [int(round(k)) for k in 10*abs(samplesX.max(axis=0)-samplesX.min(axis=0))]
    #ub = [int(round(k)) for k in ub]

    # Silverman's rule of thumb for starting points for band width estimation
    w0_vec = 1.06*np.std(samplesX, axis=0)*N**(-1/5)
    # heuristic starting points (global minimum is often closer to 0)
    w0_vec = 1/10*abs(samplesX.max(axis=0)-samplesX.min(axis=0))

    # variable to plot posterior pdfs and mlcv dependent on w
    plot_v = False

    # select integration method 'inverse' or 'integral'
    int_method = 'integral'

    if int_method not in ['integral', 'inverse']:
        raise RuntimeError("Select either 'integral' or 'inverse' as your numerical integration method")

    ## Find optimal bandwidth for kernel density estimation (kde)
    print('\n-Calculating optimal bandwidths:\n')

    # Find the optimal bandwidth for each dimension with maximum likelihood
    # cross validation (MLCV)
    w_opt = np.zeros(d)
    for u in range(d):
        w_opt_handle = lambda w: w_opt_finder(w, N, samplesX[:,u], idcs_x, counts)
        [w_opt[u], _, ex_flag, _] = sp.optimize.fminbound(w_opt_handle, lb, ub[u], xtol=Tolx, maxfun=Maxiter, full_output=True)

        # if optimal w not found, try fmincon, success if ex_flag == 0
        if ex_flag != 0:
            print('Fminbnd was not succesful, now trying scipy.optimize.minimize\n')
            res = sp.optimize.minimize(w_opt_handle, w0_vec[u], bounds=[(lb, ub[u])], options={'gtol': Tolx})
            w_opt[u] = res.x
            ex_flag = res.success

            # exitflag handling, success if ex_flag == 1
            if ex_flag:
                exitflag = 1
                errormsg = []
            else:
                exitflag = 0
                S_F1 = []
                errormsg = res.message
                return S_F1, exitflag, errormsg

    print("\nOptimal bandwidths:")
    print(w_opt)
    ## Compute sensitivity indices
    print('\nCalculating sensitivity indices:\n')

    if int_method == 'inverse':
        # numerical integration samples
        N_i = int(5e4)
        rng = np.random.default_rng(12345)
        UX = rng.random((N_i, d))
        B = np.zeros((N_i,d))

        for u in range(d):
            # get samples distributed according to the individual pdfs
            s = dist[u].icdf(UX[:,u])
            # get posterior pdf values approximated by kde at sample points
            kernel = sp.stats.gaussian_kde(samplesX[:,u])
            kernel.set_bandwidth(w_opt[u])
            b = kernel.evaluate(s)
            # calculate updated Bayes' samples with maximum value of 1
            B[:,u] = np.minimum(b * Pf / dist[u].pdf(s), 1)

        S_F1 = np.var(B, axis=0) / (Pf*(1-Pf))

    elif int_method == 'integral':
        B = np.zeros(d)
        for u in range(d):
            pdf_int = lambda x: (np.minimum(kde(samplesX[:,u], x, w_opt[u])* Pf / dist[u].pdf(x), 1) - Pf)**2 * dist[u].pdf(x)
            # integration range from: [mean-5std, mean+5std]
            B[u] = sp.integrate.quad(pdf_int, dist[u].mean()-15*dist[u].std(), dist[u].mean()+15*dist[u].std(), limit=1000)[0]

        # Sensitivity indices computed via equ. 4 with the samples obtained from
        # maximum-likelihood cross-validation method
        S_F1 = B / (Pf*(1-Pf))


    # check if computed indices are valid
    if not np.all(np.isreal(S_F1)):
        errormsg = 'Computation was not successful, at least one index is complex.'
        exitflag = 0
    elif np.any(S_F1 > 1):
        errormsg = 'Computation was not successful, at least one index is greater than 1.'
        exitflag = 0
    elif np.any(S_F1 < 0):
        errormsg = 'Computation was not successful, at least one index is smaller than 0.'
        exitflag = 0
    elif not np.all(np.isfinite(S_F1)):
        errormsg = 'Computation was not successful, at least one index is NaN.'
        exitflag = 0
    else:
        errormsg = []
        exitflag = 1

    if plot_v:
        print("\n***Plotting MLCV(w) and kde(x)\n")

        plt.figure()
        w_x = np.linspace(0.001, 2*max(w_opt), 200)
        y = np.zeros((d, len(w_x)))
        for u in range(d):
            for t in range(len(w_x)):
                y[u,t] = w_opt_finder(w_x[t], N, samplesX[:,u], idcs_x, counts)
            plt.plot(w_x, y[u,:])
            plt.plot(w_opt[u], w_opt_finder(w_opt[u], N, samplesX[:,u], idcs_x, counts), '*r')

        plt.figure()
        for u in range(d):
            kernel = sp.stats.gaussian_kde(samplesX[:,u])
            kernel.set_bandwidth(w_opt[u])
            xi = np.linspace(min(samplesX[:,u]), max(samplesX[:,u]), 1000)
            yi = kernel.evaluate(xi)
            plt.plot(xi,yi)
        plt.show()

    return [S_F1, exitflag, errormsg]
#
# ==========================================================================
# ===========================NESTED FUNCTIONS===============================
# ==========================================================================
#
def w_opt_finder(w, N, x, idc, c):
    """
    -----------------------------------------------------------------------
    function which evaluates the mlcv for given bandwidth
    -----------------------------------------------------------------------
    Input:
    * w     : bandwidth
    * N     : total number of failure samples
    * x     : failure samples (samplesX)
    * idc   : indices to reconstruct x (samplesX) from its unique array
    * c     : vector of the multiplicity of the unique values in x (samplesX)
    -----------------------------------------------------------------------
    Output:
    * mlcw_w: maximum-likelihood cross-validation for specfic w
    -----------------------------------------------------------------------
    """

    # Calculate a width sample via maximum likelihood cross-validation
    mlcv_w = 0

    for k in range(len(c)):
        # get all indices from samples which are different than x[k]
        idx = idc[np.arange(len(c))!=k]

        mlcv_w = mlcv_w + c[k]*(np.log(np.sum(sp.stats.norm.pdf((x[idc[k]]-x[idx])/(w+np.finfo(float).eps))*c[np.arange(len(c))!=k])) - np.log((N-c[k])*w + np.finfo(float).eps))

    # Scale mlcv_w:
    #mlcv_w = mlcv_w/N

    # The maximum mlcv is the minimum of -mlcv (minimizer function)
    mlcv_w = -mlcv_w

    return mlcv_w


# ==========================================================================
def kde(samplesX, x_eval, bw):
    """
    -----------------------------------------------------------------------
    function to return a kde at given evaluation points and given bandwidth
    -----------------------------------------------------------------------
    Input:
    * samplesX: array of failure samplesX
    * x_eval  : array of evaluation points
    * bw      : given bandwidth
    -----------------------------------------------------------------------
    Output:
    * y: kernel density estimation evaluated at x_eval with bw
    -----------------------------------------------------------------------
    """

    kernel = sp.stats.gaussian_kde(samplesX)
    kernel.set_bandwidth(bw)
    y = kernel.evaluate(x_eval)
    #y = reshape(f(:), size(x_eval))
    return y
