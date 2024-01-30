import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from EMvMFNM import EMvMFNM
np.seterr(all='ignore')

"""
---------------------------------------------------------------------------
Improved cross entropy-based importance sampling with vMFN mixture model
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
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2023-04:
* Modification to Sensitivity Analysis Calls
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
* samples_return        : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* lv        : total number of levels
* N_tot     : total number of samples
* samplesU  : list with the samples in the standard normal space
* samplesX  : list with the samples in the original space
* f_s_iid  : Independent Identically Distributed samples generated from last step
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


def iCE_vMFNM(N, p, g_fun, distr, max_it, CV_target, k_init, samples_return):
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
    samplesU = list()  # space for the samples in the standard space

    # %% CE procedure
    # initial nakagami parameters (make it equal to chi distribution)
    omega_init = dim  # spread parameter
    m_init = dim / 2  # shape parameter

    # initial von Mises-Fisher parameters
    kappa_init = 0  # Concentration parameter (zero for uniform distribution)
    mu_init = hs_sample(1, dim, 1)  # Initial mean sampled on unit hypersphere

    Pi_init = np.array([1.0])  # weight of the mixtures
    sigma_t = np.zeros(max_it)  # squared difference between COV and target COV

    # %% Initializing parameters
    mu_hat = mu_init
    kappa_hat = kappa_init
    omega_hat = omega_init
    m_hat = m_init
    Pi_hat = Pi_init
    k = k_init

    # Function reference
    minimize = sp.optimize.fminbound

    # Iteration
    for j in range(max_it):
        # save parameters from previous step
        mu_cur = mu_hat
        kappa_cur = kappa_hat
        omega_cur = omega_hat
        m_cur = m_hat

        # Generate samples and save them
        X = vMFNM_sample(mu_cur, kappa_cur, omega_cur, m_cur, Pi_hat, N).reshape(-1, dim)

        # Count generated samples
        N_tot += N

        # Evaluation of the limit state function
        geval = G_LSF(X)

        # initialize sigma_0
        if j == 0:
            sigma_t[j] = 10 * np.mean(geval)
        else:
            sigma_t[j] = sigma_new

        # calculation of the likelihood ratio
        W_log = likelihood_ratio_log(X, mu_cur, kappa_cur, omega_cur, m_cur, Pi_hat)
        W = np.exp(W_log).flatten()

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
            if samples_return == 1:
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

        # EM algorithm
        [mu, kappa, m, omega, pi] = EMvMFNM(X.T, W_t, k)

        # assigning updated parameters
        mu_hat = mu.T
        kappa_hat = kappa
        m_hat = m
        omega_hat = omega
        Pi_hat = pi
        k = len(pi)

    # Samples return - all by default message
    if samples_return not in [0, 1, 2]:
        print("\n-Invalid input for samples return, all samples are returned by default")

    # Store final weights
    W_final:np.ndarray = np.copy(W)
    # needed steps
    lv = j

    # Calculation of the Probability of failure
    Pr = 1 / N * sum(W[I])

    # transform the samples to the physical/original space
    samplesX = list()
    f_s_iid = list()
    if samples_return != 0:
        samplesX = [u2x(samplesU[i][:, :]) for i in range(len(samplesU))]
    
        # resample 10000 failure samples with final weights W
        weight_id = np.random.choice(list(np.nonzero(I))[0],10000,list(W[I]))
        f_s_iid = samplesX[-1][weight_id,:]

    # Convergence is not achieved message
    if j == max_it:
        print("\n-Exit with no convergence at max iterations \n")
        
    return Pr, lv, N_tot, samplesU, samplesX, k_init, W_final, f_s_iid


# ===========================================================================
# =============================AUX FUNCTIONS=================================
# ===========================================================================
# --------------------------------------------------------------------------
# Returns uniformly distributed samples from the surface of an
# n-dimensional hypersphere
# --------------------------------------------------------------------------
# N: # samples
# n: # dimensions
# R: radius of hypersphere
# --------------------------------------------------------------------------
def hs_sample(N, n, R):

    Y = sp.stats.norm.rvs(size=(n, N))  # randn(n,N)
    Y = Y.T
    norm = np.tile(np.sqrt(np.sum(Y ** 2, axis=1)), [1, n])
    X = Y / norm * R  # X = np.matmul(Y/norm,R)

    return X


# ===========================================================================
# --------------------------------------------------------------------------
# Returns samples from the von Mises-Fisher-Nakagami mixture
# --------------------------------------------------------------------------
def vMFNM_sample(mu, kappa, omega, m, alpha, N):

    [k, dim] = np.shape(mu)
    if k == 1:
        # sampling the radius
        #     pd=makedist('Nakagami','mu',m,'omega',omega)
        #     R=pd.random(N,1)
        R = np.sqrt(sp.stats.gamma.rvs(a=m, scale=omega / m, size=[N, 1]))

        # sampling on unit hypersphere
        X_norm = vsamp(mu.T, kappa, N)

    else:
        # Determine number of samples from each distribution
        z = np.sum(dummyvar(np.random.choice(range(k), N, True, alpha)), axis=0)
        k = len(z)

        # Generation of samples
        R = np.zeros([N, 1])
        R_last = 0
        X_norm = np.zeros([N, dim])
        X_last = 0

        for p in range(k):
            # sampling the radius
            R[R_last : R_last + z[p], :] = np.sqrt(
                sp.stats.gamma.rvs(
                    a=m[:, p], scale=omega[:, p] / m[:, p], size=[z[p], 1]
                )
            )
            R_last = R_last + z[p]

            # sampling on unit hypersphere
            X_norm[X_last : X_last + z[p], :] = vsamp(mu[p, :].T, kappa[p], z[p])
            X_last = X_last + z[p]

            # clear pd

    # Assign sample vector
    X = R * X_norm  # bsxfun(@times,R,X_norm)

    return X


# ===========================================================================
# --------------------------------------------------------------------------
# Returns samples from the von Mises-Fisher distribution
# --------------------------------------------------------------------------
def vsamp(center, kappa, n):

    d = np.size(center, axis=0)  # Dimensionality
    l = kappa  # shorthand
    t1 = np.sqrt(4 * l * l + (d - 1) * (d - 1))
    b = (-2 * l + t1) / (d - 1)
    x0 = (1 - b) / (1 + b)
    X = np.zeros([n, d])
    m = (d - 1) / 2
    c = l * x0 + (d - 1) * np.log(1 - x0 * x0)

    for i in range(n):
        t = -1000
        u = 1
        while t < np.log(u):
            z = sp.stats.beta.rvs(m, m)  # z is a beta rand var
            u = sp.stats.uniform.rvs()  # u is unif rand var
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            t = l * w + (d - 1) * np.log(1 - x0 * w) - c

        v = hs_sample(1, d - 1, 1)
        X[i, : d - 1] = (
            np.sqrt(1 - w * w) * v
        )  # X[i,:d-1] = np.matmul(np.sqrt(1-w*w),v.T)
        X[i, d - 1] = w

    [v, b] = house(center)
    Q = np.eye(d) - b * np.matmul(v, v.T)
    for i in range(n):
        tmpv = np.matmul(Q, X[i, :].T)
        X[i, :] = tmpv.T

    return X


# ===========================================================================
# --------------------------------------------------------------------------
# X,mu,kappa
# Returns the von Mises-Fisher mixture log pdf on the unit hypersphere
# --------------------------------------------------------------------------
def vMF_logpdf(X, mu, kappa):

    d = np.size(X, axis=0)
    n = np.size(X, axis=1)

    if kappa == 0:
        A = np.log(d) + np.log(np.pi ** (d / 2)) - sp.special.gammaln(d / 2 + 1)
        y = -A * np.ones([1, n])
    elif kappa > 0:
        c = (
            (d / 2 - 1) * np.log(kappa)
            - (d / 2) * np.log(2 * np.pi)
            - logbesseli(d / 2 - 1, kappa)
        )
        q = np.matmul((mu * kappa).T, X)  # bsxfun(@times,mu,kappa)'*X
        y = q + c.T  # bsxfun(@plus,q,c')
    else:
        raise ValueError("Kappa must not be negative!")

    return y


# ===========================================================================
# --------------------------------------------------------------------------
# Returns the value of the log-nakagami-pdf
# --------------------------------------------------------------------------
def nakagami_logpdf(X, m, om):

    y = (
        np.log(2)
        + m * (np.log(m) - np.log(om) - X ** 2 / om)
        + np.log(X) * (2 * m - 1)
        - sp.special.gammaln(m)
    )

    return y


# ===========================================================================
# --------------------------------------------------------------------------
# likelihood_ratio_log()
# --------------------------------------------------------------------------
def likelihood_ratio_log(X, mu, kappa, omega, m, alpha):

    k = len(alpha)
    [N, dim] = np.shape(X)
    R = np.sqrt(np.sum(X * X, axis=1)).reshape(-1, 1)
    if k == 1:
        # log pdf of vMF distribution
        logpdf_vMF = vMF_logpdf((X / R).T, mu.T, kappa).T
        # log pdf of Nakagami distribution
        logpdf_N = nakagami_logpdf(R, m, omega)
        # log pdf of weighted combined distribution
        h_log = logpdf_vMF + logpdf_N
    else:
        logpdf_vMF = np.zeros([N, k])
        logpdf_N = np.zeros([N, k])
        h_log = np.zeros([N, k])

        # log pdf of distributions in the mixture
        for p in range(k):
            # log pdf of vMF distribution
            logpdf_vMF[:, p] = vMF_logpdf((X / R).T, mu[p, :].T, kappa[p]).squeeze()
            # log pdf of Nakagami distribution
            logpdf_N[:, p] = nakagami_logpdf(R, m[:, p], omega[:, p]).squeeze()
            # log pdf of weighted combined distribution
            h_log[:, p] = logpdf_vMF[:, p] + logpdf_N[:, p] + np.log(alpha[p])

        # mixture log pdf
        h_log = sp.special.logsumexp(h_log, axis = 1,keepdims = True)

    # unit hypersphere uniform log pdf
    A = np.log(dim) + np.log(np.pi ** (dim / 2)) - sp.special.gammaln(dim / 2 + 1)
    f_u = -A

    # chi log pdf
    f_chi = (
        np.log(2) * (1 - dim / 2)
        + np.log(R) * (dim - 1)
        - 0.5 * R ** 2
        - sp.special.gammaln(dim / 2)
    )

    # logpdf of the standard distribution (uniform combined with chi distribution)
    f_log = f_u + f_chi
    W_log = f_log - h_log

    return W_log


# ===========================================================================
# --------------------------------------------------------------------------
# HOUSE Returns the householder transf to reduce x to b*e_n
#
# [V,B] = HOUSE(X)  Returns vector v and multiplier b so that
# H = eye(n)-b*v*v' is the householder matrix that will transform
# Hx ==> [0 0 0 ... ||x||], where  is a constant.
# --------------------------------------------------------------------------
def house(x):

    x = x.squeeze()
    n = len(x)
    s = np.matmul(x[: n - 1].T, x[: n - 1])
    v = np.concatenate([x[: n - 1], np.array([1.0])]).squeeze()
    if s == 0:
        b = 0
    else:
        m = np.sqrt(x[n - 1] * x[n - 1] + s)

        if x[n - 1] <= 0:
            v[n - 1] = x[n - 1] - m
        else:
            v[n - 1] = -s / (x[n - 1] + m)

        b = 2 * v[n - 1] * v[n - 1] / (s + v[n - 1] * v[n - 1])
        v = v / v[n - 1]

    v = v.reshape(-1, 1)

    return [v, b]


# ===========================================================================
# --------------------------------------------------------------------------
# log of the Bessel function, extended for large nu and x
# approximation from Eqn 9.7.7 of Abramowitz and Stegun
# http://www.math.sfu.ca/~cbm/aands/page_378.htm
# --------------------------------------------------------------------------
def logbesseli(nu, x):

    if nu == 0:  # special case when nu=0
        logb = np.log(sp.special.iv(nu, x))  # besseli
    else:  # normal case
        # n    = np.size(x, axis=0)
        n = 1  # since x is always scalar here
        frac = x / nu
        square = np.ones(n) + frac ** 2
        root = np.sqrt(square)
        eta = root + np.log(frac) - np.log(np.ones(n) + root)
        logb = -np.log(np.sqrt(2 * np.pi * nu)) + nu * eta - 0.25 * np.log(square)

    return logb


# ===========================================================================
# --------------------------------------------------------------------------
# Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
#   By default dim = 0 (columns).
# Written by Michael Chen (sth4nth@gmail.com).
# --------------------------------------------------------------------------
def logsumexp(x, dim=0):

    # subtract the largest in each column
    y = np.max(x, axis=dim).reshape(-1, 1)
    x = x - y
    s = y + np.log(np.sum(np.exp(x), axis=dim)).reshape(-1, 1)
    # ===========================================================================
    i = np.where(np.invert(np.isfinite(y).squeeze()))
    s[i] = y[i]

    return s


# ===========================================================================
# --------------------------------------------------------------------------
# Translation of the Matlab-function "dummyvar()" to Python
# --------------------------------------------------------------------------
def dummyvar(idx):

    n = np.max(idx) + 1
    d = np.zeros([len(idx), n], int)
    for i in range(len(idx)):
        d[i, idx[i]] = 1

    return d


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
