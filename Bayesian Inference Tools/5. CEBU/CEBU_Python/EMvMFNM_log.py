import numpy as np
import scipy as sp
np.seterr(all='ignore')

"""
%% Perform soft EM algorithm for fitting the von Mises-Fisher-Nakagami mixture model.
%{
---------------------------------------------------------------------------
Created by:
Michael Engel (m.engel@tum.de)

Based upon the code of:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2020-07-29
---------------------------------------------------------------------------
Input:
* X : data matrix (dimensions x Number of samples)
* logW : vector of log-likelihood-ratios for weighted samples
* k : number of vMFN-distributions in the mixture
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


def EMvMFNM_log(X,logW,k):
    # reshaping just to be sure
    logW = logW.reshape(1, -1)

    # initialization
    logM = initialization(X, k)

    R = np.sqrt(np.sum(X * X, axis=0)).reshape(-1, 1)  # R=sqrt(sum(X.^2))'
    X_norm = np.divide(X, R.T)  # X_norm=(bsxfun(@times,X,1./R'))

    tol = 1e-5
    maxiter = 500
    llh = np.full([2, maxiter], -np.inf)
    converged = False
    t = 0

    # soft EM algorithm
    while (not converged) and (t + 1 < maxiter):
        t = t + 1
        label = np.argmax(logM, axis=1)
        u = np.unique(label)  # non-empty components
        if np.size(logM, axis=1) != np.size(u, axis=0):
            logM = logM[:, u]  # remove empty components

        [mu, kappa, m, omega, alpha] = maximization(X_norm, logW, R, logM)
        [logM, llh[:, t]] = expectation(X_norm, logW,  mu, kappa, m, omega, alpha, R)

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
    n:int = np.shape(X)[1]
    whitened = sp.cluster.vq.whiten(X.T)
    # Use the K-Means algorithm from Scipy
    codebook, distortion = sp.cluster.vq.kmeans(whitened,k)
    label,dist = sp.cluster.vq.vq(whitened,codebook)
    # build M
    logM:np.ndarray = np.ones((n,k))*(-1e200)
    for i in range(n):
        logM[i,label[i]]=0

    return logM


# ===========================================================================
# --------------------------------------------------------------------------
# Expectation
# --------------------------------------------------------------------------
def expectation(X:np.ndarray,logW:np.ndarray,
                mu:np.ndarray,kappa:np.ndarray,
                m:np.ndarray,omega:np.ndarray,
                alpha:np.ndarray,R:np.ndarray):

    n = np.size(X, axis=1)
    k = np.size(mu, axis=1)

    logvMF = np.zeros([n, k])
    lognakagami = np.zeros([n, k])
    logpdf = np.zeros([n, k])

    # logpdf
    for i in range(k):
        logvMF[:, i] = logvMFpdf(X, mu[:, i], kappa.ravel()[i]).T
        lognakagami[:, i] = lognakagamipdf(R, m.ravel()[i], omega.ravel()[i])
        logpdf[:, i] = logvMF[:, i] + lognakagami[:, i] + np.log(alpha.ravel()[i])

    # Matrix of posterior probabilities
    T = logsumexp(logpdf, 1)
    logM = logpdf - T  # logM = bsxfun(@minus,logpdf,T)
    logM = logM - logsumexp(logM ,1)

    # loglikelihood as tolerance criterion
    logvMF_weighted = logvMF + np.log(alpha)  # bsxfun(@plus,logvMF,log(alpha))
    lognakagami_weighted = lognakagami + np.log(
        alpha
    )  # bsxfun(@plus,lognakagami,log(alpha))
    T_vMF = logsumexp(logvMF_weighted, 1)
    T_nakagami = logsumexp(lognakagami_weighted, 1)

    llh1 = np.array(
        [
            np.sum(logW.T + T_vMF, axis=0) - logsumexp(logW,1),
            np.sum(logW.T * T_nakagami, axis=0) - logsumexp(logW,1),
        ]
    ).squeeze()
    llh = llh1

    if np.any(np.isnan(logM)) or np.any(np.isinf(logM)):
        print('\nEMvMFNM_log: issue with M\n')

    return logM, llh


# ===========================================================================
# --------------------------------------------------------------------------
# Maximization
# --------------------------------------------------------------------------
def maximization(X_norm:np.ndarray,logW:np.ndarray,R:np.ndarray,logM:np.ndarray):

    logMW = np.add(logW.T,logM)
    d = np.size(X_norm, axis=0)
    lognk = logsumexp(logMW, dim=0)

    if np.any(np.isinf(lognk)) or np.any(np.isnan(lognk)):
        lognk[np.isinf(lognk)] = -1e50
        lognk[np.isnan(lognk)] = -1e50
    
    
    # distribution weights
    alpha = np.exp(np.subtract(lognk, logsumexp(logW,dim=1)))

    # mean directions
    signX = np.sign(X_norm)
    logX = np.log(np.abs(X_norm))

    mu = np.zeros((d,np.shape(lognk)[1]))
    norm_mu = np.zeros((np.shape(lognk)[1],1))
    lognorm_mu = np.zeros_like(norm_mu)

    for ii in range(len(alpha)+1):
        [mu_unnormed,signmu] = signedlogsumexp(np.add(logX,logMW[:,ii].T),1,signX)
        lognorm_mu[ii] = np.ravel(0.5*logsumexp(2*mu_unnormed,0))
        mu[:,ii] = np.ravel(np.multiply(np.exp(mu_unnormed-lognorm_mu[ii]),signmu))

    # approximated concentration parameter
    xi = np.minimum(np.exp(lognorm_mu.T-lognk), 0.999999999)
    kappa = (xi * d - xi ** 3) / (1 - xi ** 2)

    # spread parameter
    R[R==0] =1e-300; # for numerical stability; should actually be equal to the closest number to zero without cancellation error
    logR = np.log(R)
    logRsquare = 2*logR
    omega = np.exp(logsumexp(logMW+logRsquare,0) - logsumexp(logMW,0))

    # approximated shape parameter
    logRpower = 4*logR
    mu4       = np.exp(logsumexp(logMW+logRpower,0) - logsumexp(logMW,0))
    m         = np.divide(omega**2,(mu4-omega**2))

    m[m<0.5] = d/2
    m[np.isinf(m)] = d/2
    m[np.isnan(m)] = d/2

    return mu, kappa, m, omega, alpha


# ===========================================================================
# --------------------------------------------------------------------------
# Returns the log of the vMF-pdf
# --------------------------------------------------------------------------
def logvMFpdf(X:np.ndarray, mu:np.ndarray, kappa:np.ndarray)->np.ndarray:

    d = np.size(X, axis=0)
    mu = mu.reshape(-1, 1)
    if kappa == 0:
        # unit hypersphere uniform log pdf
        A = np.log(d) + np.log(np.pi ** (d / 2)) - sp.special.gammaln(d / 2 + 1)
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
    y = np.max(x, axis=dim)
    y = np.expand_dims(y,axis=dim)
    x = np.subtract(x , y)
    s = np.add(y ,   np.expand_dims(np.log(np.sum(np.exp(x), axis=dim)) , axis=dim ) )
    # if a bug occurs here, maybe find a better translation from matlab:
    i = np.where(np.invert(np.isfinite(y).squeeze()))

    if np.size(i)!=0:
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


#=========================================================================
# ------------------------------------------------------------------------
# Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
# By default dim = 1 (columns).
# Written by Michael Chen (sth4nth@gmail.com).
#   
# Adapted by Michael Engel such that log(sum(exp(x)*b,dim)) and
# negative results are supported.
# Only recommended for b working as a sign-vector of exp(x).
# ------------------------------------------------------------------------
def signedlogsumexp(x, dim=0, b=[]):
   
    
    if np.size(b)==0:
        b = np.ones((1,np.shape(x)[dim]))

    # subtract the largest in each column (rescaling x to (0,1] 
    # where log offers better precision)
    y:np.ndarray = np.max(x,axis = dim)
    y = np.expand_dims(y,axis=dim)
    x:np.ndarray = np.subtract(x,y) # bsxfun(@minus,x,y)
    term:np.ndarray = np.sum(np.multiply(np.exp(x),b),axis=dim)
    term = np.expand_dims(term,axis=dim)
    signn = np.ones(term.shape)
    signn[term.ravel()<0] = -1

    # return nonfinite value if existing
    s = y + np.log(np.abs(term))
    i = np.where(np.invert(np.isfinite(y).squeeze()))
    
    if np.size(i)!=0:
       s[i] = y[i]

    return s,signn
