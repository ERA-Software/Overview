import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from EMvMFNM import EMvMFNM
"""
---------------------------------------------------------------------------
Cross entropy-based importance sampling with vMFNM-distribution
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de),
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Matthias Willer
Peter Kaplan
Daniel Koutas
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2023-04:
* Modification to Sensitivity Analysis Calls
---------------------------------------------------------------------------
Comments:
* The LSF must be coded to accept the full set of samples and no one by one
  (see line 124)
---------------------------------------------------------------------------
Input:
* N                     : number of samples per level
* p                     : quantile value to select samples for parameter update
* g_fun                 : limit state function
* distr                 : Nataf distribution object or
                          marginal distribution object of the input variables
* k_init                : initial number of distributions in the mixture model
* samples_return        : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* lv        : total number of levels
* N_tot     : total number of samples
* gamma_hat : intermediate levels
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
* k_fin     : final number of distributions in the mixture
* W_final   : final weights
* f_s_iid   : Independent Identically Distributed samples generated from last step
---------------------------------------------------------------------------
Based on:
1."Cross entropy-based importance sampling using Gaussian densities revisited"
   Geyer et al.
   To appear in Structural Safety
2."A new flexible mixture model for cross entropy based importance sampling".
   Papaioannou et al. (2018)
   In preparation.
---------------------------------------------------------------------------
"""


def CEIS_vMFNM(N, p, g_fun, distr, k_init, samples_return):
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

    if dim < 2:
        raise RuntimeError("Sorry, the vMFN-model can only be applied to d > 1!")

    # LSF in standard space
    G_LSF = lambda u: g_fun(u2x(u))

    # %% Initialization of variables and storage
    j = 0  # initial level
    max_it = 50  # estimated number of iterations
    N_tot = 0  # total number of samples

    # Definition of parameters of the random variables (uncorrelated standard normal)
    gamma_hat = np.zeros([max_it + 1])  # space for gamma
    samplesU = list()

    # %% CE procedure
    # initial nakagami parameters (make it equal to chi distribution)
    omega_init = dim  # spread parameter
    m_init = dim / 2  # shape parameter

    # initial von Mises-Fisher parameters
    kappa_init = 0  # Concentration parameter (zero for uniform distribution)
    mu_init = hs_sample(1, dim, 1)  # Initial mean sampled on unit hypersphere

    # initial disribution weight
    alpha_init = np.array([1.0])

    # %% Initializing parameters
    mu_hat = mu_init
    kappa_hat = kappa_init
    omega_hat = omega_init
    m_hat = m_init
    gamma_hat[j] = 1
    alpha_hat = alpha_init

    # %% Iteration
    for j in range(max_it):
        # save parameters from previous step
        mu_cur = mu_hat
        kappa_cur = kappa_hat
        omega_cur = omega_hat
        m_cur = m_hat
        alpha_cur = alpha_hat

        # Generate samples
        X = vMFNM_sample(mu_cur, kappa_cur, omega_cur, m_cur, alpha_cur, N).reshape(
            -1, dim)

        # Count generated samples
        N_tot = N_tot + N

        # Evaluation of the limit state function
        geval = G_LSF(X)

        # Calculation of the likelihood ratio
        W_log = likelihood_ratio_log(X, mu_cur, kappa_cur, omega_cur, m_cur, alpha_cur)

        # Samples return - all / all by default
        if samples_return not in [0, 1]:
            samplesU.append(X)

        # Check convergence
        if gamma_hat[j] == 0:
            k_fin = len(alpha_cur)
            # Samples return - last
            if samples_return == 1:
                samplesU.append(X)
            break

        # obtaining estimator gamma
        gamma_hat[j + 1] = np.maximum(0, np.nanpercentile(geval, p * 100))
        print("\nIntermediate failure threshold: ", gamma_hat[j + 1])

        # Indicator function
        I = geval <= gamma_hat[j + 1]

        # EM algorithm
        [mu, kappa, m, omega, alpha] = EMvMFNM(X[I, :].T, np.exp(W_log[I, :]), k_init)

        # Assigning updated parameters
        mu_hat = mu.T
        kappa_hat = kappa
        m_hat = m
        omega_hat = omega
        alpha_hat = alpha

    # Samples return - all by default message
    if samples_return not in [0, 1, 2]:
        print("\n-Invalid input for samples return, all samples are returned by default")

    # store the needed steps
    lv = j
    gamma_hat = gamma_hat[: lv + 1]

    # Store the final  weights
    W_final = np.exp(W_log)

    # Calculation of Probability of failure
    I = geval <= gamma_hat[j]
    Pr = 1 / N * np.sum(W_final[I, :])

    # transform the samples to the physical/original space
    samplesX = list()
    f_s_iid = list()
    if samples_return != 0:
        samplesX = [u2x(samplesU[i][:, :]) for i in range(len(samplesU))]

        # resample 10000 failure samples with final weights W
        weight_id = np.random.choice(list(np.nonzero(I))[0],10000,list(W_final[I, :]))
        f_s_iid = samplesX[-1][weight_id,:]

    # Convergence is not achieved message
    if j == max_it:
        print("\n-Exit with no convergence at max iterations \n")
        
    return Pr, lv, N_tot, gamma_hat, samplesU, samplesX, k_fin, W_final, f_s_iid


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
        h_log = sp.special.logsumexp(h_log, axis=1,keepdims = True)

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
# Translation of the Matlab-function "dummyvar()" to Python
# --------------------------------------------------------------------------
def dummyvar(idx):

    n = np.max(idx) + 1
    d = np.zeros([len(idx), n], int)
    for i in range(len(idx)):
        d[i, idx[i]] = 1

    return d
