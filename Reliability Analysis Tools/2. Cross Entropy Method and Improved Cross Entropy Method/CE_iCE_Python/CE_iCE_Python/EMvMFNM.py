import numpy as np
import scipy as sp
np.seterr(all='ignore')

"""
---------------------------------------------------------------------------
Perform soft EM algorithm for fitting the von Mises-Fisher-Nakagami mixture model.
---------------------------------------------------------------------------
Created by:
Sebastian Geyer 
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Matthias Willer
Luca Sardi

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2021-03:
* input dimensions of limit state function changed to rows*columns = 
  samples*dimensions
---------------------------------------------------------------------------
Input:
* X   : data matrix (dimensions x Number of samples)
* W   : vector of likelihood ratios for weighted samples
* nGM : number of vMFN-distributions in the mixture
---------------------------------------------------------------------------
Output:
* mu    : mean directions
* kappa : approximated concentration parameter 
* m     : approximated shape parameter
* omega : spread parameter 
* alpha : distribution weights
---------------------------------------------------------------------------
Based on:
1. "EM Demystified: An Expectation-Maximization Tutorial"
   Yihua Chen and Maya R. Gupta
   University of Washington, Dep. of EE (Feb. 2010)
---------------------------------------------------------------------------
"""


def EMvMFNM(X, W, k):
    # reshaping just to be sure
    W = W.reshape(-1, 1)

    # initialization
    M = initialization(X, k)

    R = np.sqrt(np.sum(X * X, axis=0)).reshape(-1, 1)  # R=sqrt(sum(X.^2))'
    X_norm = X / R.T  # X_norm=(bsxfun(@times,X,1./R'))

    tol = 1e-5
    maxiter = 500
    llh = np.full([2, maxiter], -np.inf)
    converged = False
    t = 0

    # soft EM algorithm
    while (not converged) and (t + 1 < maxiter):
        t = t + 1
        label = np.argmax(M, axis=1)
        u = np.unique(label)  # non-empty components
        if np.size(M, axis=1) != np.size(u, axis=0):
            M = M[:, u]  # remove empty components

        [mu, kappa, m, omega, alpha] = maximization(X_norm, W, R, M)
        [M, llh[:, t]] = expectation(X_norm, W, R, mu, kappa, m, omega, alpha)

        if t > 1:
            con1 = abs(llh[0, t] - llh[0, t - 1]) < tol * abs(llh[0, t])
            con2 = abs(llh[1, t] - llh[1, t - 1]) < tol * 100 * abs(llh[1, t])
            converged = min(con1, con2)

    if converged:
        print("Converged in", t, "steps.")
    else:
        print("Not converged in ", maxiter, " steps.")

    return mu, kappa, m, omega, alpha


# ===========================================================================
# =============================AUX FUNCTIONS=================================
# ===========================================================================
# --------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------
def initialization(X, k):

    # Random initialization
    n = np.size(X, axis=1)
    idx = np.random.choice(range(n), k)
    m = X[:, idx]
    label = np.argmax(
        np.matmul(m.T, X) - np.sum(m * m, axis=0).reshape(-1, 1) / 2, axis=0
    )
    u = np.unique(label)
    while k != len(u):
        idx = np.random.choice(range(n), k)
        m = X[:, idx]
        label = np.argmax(
            np.matmul(m.T, X) - np.sum(m * m, axis=0).reshape(-1, 1) / 2, axis=0
        )
        u = np.unique(label)

    M = np.zeros([n, k], dtype=int)
    for i in range(n):
        M[i, label[i]] = 1

    return M


# ===========================================================================
# --------------------------------------------------------------------------
# Expectation
# --------------------------------------------------------------------------
def expectation(X, W, R, mu, kappa, m, omega, alpha):

    n = np.size(X, axis=1)
    k = np.size(mu, axis=1)

    logvMF = np.zeros([n, k])
    lognakagami = np.zeros([n, k])
    logpdf = np.zeros([n, k])

    # logpdf
    for i in range(k):
        logvMF[:, i] = logvMFpdf(X, mu[:, i], kappa[i]).T
        lognakagami[:, i] = lognakagamipdf(R, m[:, i], omega[:, i])
        logpdf[:, i] = logvMF[:, i] + lognakagami[:, i] + np.log(alpha[i])

    # Matrix of posterior probabilities
    T = logsumexp(logpdf, 1)
    logM = logpdf - T  # logM = bsxfun(@minus,logpdf,T)
    M = np.exp(logM)
    M[M < 1e-3] = 0
    M = M / np.sum(M, axis=1).reshape(-1, 1)  # M=bsxfun(@times,M,1./sum(M,2))

    # loglikelihood as tolerance criterion
    logvMF_weighted = logvMF + np.log(alpha)  # bsxfun(@plus,logvMF,log(alpha))
    lognakagami_weighted = lognakagami + np.log(
        alpha
    )  # bsxfun(@plus,lognakagami,log(alpha))
    T_vMF = logsumexp(logvMF_weighted, 1)
    T_nakagami = logsumexp(lognakagami_weighted, 1)

    llh1 = np.array(
        [
            np.sum(W * T_vMF, axis=0) / np.sum(W, axis=0),
            np.sum(W * T_nakagami, axis=0) / np.sum(W, axis=0),
        ]
    ).squeeze()
    llh = llh1

    return M, llh


# ===========================================================================
# --------------------------------------------------------------------------
# Maximization
# --------------------------------------------------------------------------
def maximization(X, W, R, M):

    M = W * M
    d = np.size(X, axis=0)
    nk = np.sum(M, axis=0)

    # distribution weights
    alpha = nk / sum(W)

    # mean directions
    mu_unnormed = np.matmul(X, M)
    norm_mu = np.sqrt(np.sum(mu_unnormed * mu_unnormed, axis=0))
    mu = mu_unnormed / norm_mu

    # approximated concentration parameter
    xi = np.minimum(norm_mu / nk, 0.95)
    kappa = (xi * d - xi ** 3) / (1 - xi ** 2)

    # spread parameter
    omega = np.matmul(M.T, R * R).T / np.sum(M, axis=0)

    # approximated shape parameter
    mu4 = np.matmul(M.T, R ** 4).T / np.sum(M, axis=0)
    m = omega ** 2 / (mu4 - omega ** 2)
    m[m < 0] = d / 2
    m[m > 20 * d] = d / 2

    return mu, kappa, m, omega, alpha


# ===========================================================================
# --------------------------------------------------------------------------
# Returns the log of the vMF-pdf
# --------------------------------------------------------------------------
def logvMFpdf(X, mu, kappa):

    d = np.size(X, axis=0)
    mu = mu.reshape(-1, 1)
    if kappa == 0:
        # unit hypersphere uniform log pdf
        A = np.log(d) + np.log(np.pi ^ (d / 2)) - sp.special.gammaln(d / 2 + 1)
        y = -A
    elif kappa > 0:
        c = (
            (d / 2 - 1) * np.log(kappa)
            - (d / 2) * np.log(2 * np.pi)
            - logbesseli(d / 2 - 1, kappa)
        )
        q = np.matmul((mu * kappa).T, X)
        y = q + c.T
        y = y.squeeze()
    else:
        raise ValueError("Concentration parameter kappa must not be negative!")

    return y


# ===========================================================================
# --------------------------------------------------------------------------
# Returns the log of the nakagami-pdf
# --------------------------------------------------------------------------
def lognakagamipdf(X, m, om):

    y = (
        np.log(2)
        + m * (np.log(m) - np.log(om) - X * X / om)
        + np.log(X) * (2 * m - 1)
        - sp.special.gammaln(m)
    )

    return y.squeeze()


# ===========================================================================
# --------------------------------------------------------------------------
# log of the Bessel function, extended for large nu and x
# approximation from Eqn 9.7.7 of Abramowitz and Stegun
# http://www.math.sfu.ca/~cbm/aands/page_378.htm
# --------------------------------------------------------------------------
def logbesseli(nu, x):

    if nu == 0:  # special case when nu=0
        logb = np.log(sp.special.iv(nu, x))
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
    # if a bug occurs here, maybe find a better translation from matlab:
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
