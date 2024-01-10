import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from EMGM import EMGM
from Sim_Sobol_indices import Sim_Sobol_indices
np.seterr(all='ignore')

"""
---------------------------------------------------------------------------
Improved cross entropy-based importance sampling with Gaussian Mixtures
---------------------------------------------------------------------------
Created by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Fong-Lin Wu
Matthias Willer
Peter Kaplan
Luca Sardi
Daniel Koutas

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2022-04:
* inlusion of sensitivity analysis
---------------------------------------------------------------------------
Comments:
* Adopt draft scripts from Sebastian and reconstruct the code to comply
  with the style of the published codes
* W is the original importance weight (likelihood ratio)
* W_t is the transitional importance weight of samples
* W_approx is the weight (ratio) between real and approximated indicator functions
---------------------------------------------------------------------------
Input:
* N                     : number of samples per level
* g_fun                 : limit state function
* max_it                : maximum number of iterations
* distr                 : Nataf distribution object or marginal distribution object of the input variables
* CV_target             : taeget correlation of variation of weights
* k_init                : initial number of Gaussians in the mixture model
* sensitivity_analysis  : implementation of sensitivity analysis: 1 - perform, 0 - not perform
* samples_return        : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* lv        : total number of levels
* N_tot     : total number of samples
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
* S_F1      : vector of first order Sobol' indices
---------------------------------------------------------------------------
Based on:
1. Papaioannou, I., Geyer, S., & Straub, D. (2019).
   Improved cross entropy-based importance sampling with a flexible mixture model.
   Reliability Engineering & System Safety, 191, 106564
2. Geyer, S., Papaioannou, I., & Straub, D. (2019).
   Cross entropy-based importance sampling using Gaussian densities revisited.
   Structural Safety, 76, 15–27
---------------------------------------------------------------------------
"""


def iCE_GM(N, p, g_fun, distr, max_it, CV_target, k_init, sensitivity_analysis, samples_return):
    if (N * p != np.fix(N * p)) or (1 / p != np.fix(1 / p)):
        raise RuntimeError(
            "N*p and 1/p must be positive integers. Adjust N and p accordingly"
        )

    # initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):  # use Nataf transform (dependence)
        dim = len(distr.Marginals)  # number of random variables (dimension)
        u2x = lambda u: distr.U2X(u)  # from u to x

    elif isinstance(
        distr[0], ERADist
    ):  # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise
        dim = len(distr)  # number of random variables (dimension)
        u2x = lambda u: distr[0].icdf(sp.stats.norm.cdf(u))  # from u to x
    else:
        raise RuntimeError(
            "Incorrect distribution. Please create an ERADist/Nataf object!"
        )

    # LSF in standard space
    G_LSF = lambda u: g_fun(u2x(u))

    # Initialization of variables and storage
    N_tot = 0  # total number of samples

    # Definition of parameters of the random variables (uncorrelated standard normal)
    mu_init = np.zeros([1, dim])
    si_init = np.eye(dim)
    Pi_init = np.array([1.0])
    sigma_t = np.zeros(max_it)
    samplesU = list()

    # CE Procedure
    # Initializing parameters
    mu_hat = mu_init
    si_hat = si_init
    Pi_hat = Pi_init
    k = k_init

    # Function reference
    normalCDF = sp.stats.norm.cdf
    minimize = sp.optimize.fminbound

    # Iteration
    for j in range(max_it):
        # Generate samples and save them
        X = GM_sample(mu_hat, si_hat, Pi_hat, N).reshape(-1, dim)

        # Count generated samples
        N_tot += N

        # Evaluation of the limit state function
        geval = G_LSF(X)

        # initialize sigma_0
        if j == 0:
            sigma_t[j] = 10 * np.mean(geval)
        else:
            sigma_t[j] = sigma_new

        # Calculating h for the likelihood ratio
        h = h_calc(X, mu_hat, si_hat, Pi_hat)

        # Likelihood ratio
        W = (
            sp.stats.multivariate_normal.pdf(X, mean=np.zeros((dim)), cov=np.eye((dim)))
            / h
        )

        # Indicator function
        I = geval <= 0

        # Samples return - all / all by default
        if samples_return not in [0, 1]:
            samplesU.append(X)

        # check convergence
        # transitional weight W_t=I*W when sigma_t approches 0 (smooth approximation:)
        W_approx = np.divide(
            I, approx_normCDF(-geval / sigma_t[j])
        )  # weight of indicator approximations

        # import timeit
        # timeit.timeit('approx_normCDF(-geval / sigma_t[j])', globals=globals(), number=100)
        # timeit.timeit('normalCDF(-geval / sigma_t[j])', globals=globals(), number=100)

        # Cov_x   = std(np.multiply(I,W)) / mean(np.multiply(I,W))                     # poorer numerical stability
        Cov_x = np.std(W_approx) / np.mean(W_approx)
        if Cov_x <= CV_target:
            # Samples return - last
            if samples_return == 1 or (samples_return == 0 and sensitivity_analysis == 1):
                samplesU.append(X)
            break

        # compute sigma and weights for distribution fitting
        # minimize COV of W_t (W_t=approx_normCDF*W)
        fmin = lambda x: abs(
            np.std(np.multiply(approx_normCDF(-geval / x), W))
            / np.mean(np.multiply(approx_normCDF(-geval / x), W))
            - CV_target
        )
        sigma_new = minimize(fmin, 0, sigma_t[j])

        # update W_t
        W_t = np.multiply(approx_normCDF(-geval / sigma_new), W)[:, None]

        # normalize weights
        W_t = W_t / np.sum(W_t)

        # parameter update with EM algorithm
        [mu, si, pi] = EMGM(X.T, W_t, k)

        # assigning the variables with updated parameters
        mu_hat = mu.T
        si_hat = si
        Pi_hat = pi
        k = len(pi)

    # Samples return - all by default message
    if samples_return not in [0, 1, 2]:
        print("\n-Invalid input for samples return, all samples are returned by default")

    # needed steps
    lv = j

    # Calculation of the Probability of failure
    Pr = 1 / N * sum(W[I])

    # transform the samples to the physical/original space
    samplesX = list()
    if samples_return != 0 or (samples_return == 0 and sensitivity_analysis == 1):
        for i in range(len(samplesU)):
            samplesX.append(u2x(samplesU[i][:, :]))

    # %% sensitivity analysis
    if sensitivity_analysis == 1:
        # resample 10000 failure samples with final weights W
        weight_id = np.random.choice(list(np.nonzero(I))[0],10000,list(W[I]))
        f_s = samplesX[-1][weight_id,:]
        S_F1 = []

        if len(f_s) == 0:
            print("\n-Sensitivity analysis could not be performed, because no failure samples are available")
            S_F1 = []
        else:
            S_F1, exitflag, errormsg = Sim_Sobol_indices(f_s, Pr, distr)
            if exitflag == 1:
                print("\n-First order indices: \n", S_F1)
            else:
                print('\n-Sensitivity analysis could not be performed, because: \n', errormsg)
        if samples_return == 0:
            samplesU = list()  # empty return samples U
            samplesX = list()  # and X
    else:
        S_F1 = []

    # Convergence is not achieved message
    if j == max_it:
        print("\n-Exit with no convergence at max iterations \n")
        
    return Pr, lv, N_tot, samplesU, samplesX, k_init, S_F1


# ===========================================================================
# =============================AUX FUNCTIONS=================================
# ===========================================================================
def GM_sample(mu, si, pi, N):
    # ---------------------------------------------------------------------------
    # Algorithm to draw samples from a Gaussian-Mixture (GM) distribution
    # ---------------------------------------------------------------------------
    # Input:
    # * mu : [npi x d]-array of means of Gaussians in the Mixture
    # * Si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
    # * Pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1)
    # * N  : number of samples to draw from the GM distribution
    # ---------------------------------------------------------------------------
    # Output:
    # * X  : samples from the GM distribution
    # ---------------------------------------------------------------------------
    if np.size(mu, axis=0) == 1:
        mu = mu.squeeze()
        si = si.squeeze()
        X = sp.stats.multivariate_normal.rvs(mean=mu, cov=si, size=N)
    else:
        # Determine number of samples from each distribution
        z = np.round(pi * N)
        if np.sum(z) != N:
            dif = np.sum(z) - N
            ind = np.argmax(z)
            z[ind] = z[ind] - dif

        z = z.astype(int)  # integer conversion

        # Generate samples
        d = np.size(mu, axis=1)
        X = np.zeros([N, d])
        ind = 0
        for p in range(len(pi)):
            X[ind : ind + z[p], :] = sp.stats.multivariate_normal.rvs(
                mean=mu[p, :], cov=si[:, :, p], size=z[p]
            ).reshape(-1, d)
            ind = ind + z[p]

    return X


# ===========================================================================
def h_calc(X, mu, si, Pi):
    # ---------------------------------------------------------------------------
    # Basic algorithm to calculate h for the likelihood ratio
    # ---------------------------------------------------------------------------
    # Input:
    # * X  : input samples
    # * mu : [npi x d]-array of means of Gaussians in the Mixture
    # * Si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
    # * Pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1)
    # ---------------------------------------------------------------------------
    # Output:
    # * h  : parameters h (IS density)
    # ---------------------------------------------------------------------------
    N = len(X)
    k_tmp = len(Pi)
    if k_tmp == 1:
        mu = mu.squeeze()
        si = si.squeeze()
        h = sp.stats.multivariate_normal.pdf(X, mu, si)
    else:
        h_pre = np.zeros((N, k_tmp))
        for q in range(k_tmp):
            h_pre[:, q] = Pi[q] * sp.stats.multivariate_normal.pdf(
                X, mu[q, :], si[:, :, q]
            )

        h = np.sum(h_pre, axis=1)

    return h


def approx_normCDF(x):
    # Returns an approximation for the standard normal CDF based on a polynomial fit of degree 9

    erfun = np.zeros(len(x))

    idpos = x > 0
    idneg = x < 0

    t = (1 + 0.5 * abs(x / np.sqrt(2))) ** -1

    tau = t * np.exp(
        -((x / np.sqrt(2)) ** 2)
        - 1.26551223
        + 1.0000236 * t
        + 0.37409196 * (t ** 2)
        + 0.09678418 * (t ** 3)
        - 0.18628806 * (t ** 4)
        + 0.27886807 * (t ** 5)
        - 1.13520398 * (t ** 6)
        + 1.48851587 * (t ** 7)
        - 0.82215223 * (t ** 8)
        + 0.17087277 * (t ** 9)
    )
    erfun[idpos] = 1 - tau[idpos]
    erfun[idneg] = tau[idneg] - 1

    p = 0.5 * (1 + erfun)

    return p
