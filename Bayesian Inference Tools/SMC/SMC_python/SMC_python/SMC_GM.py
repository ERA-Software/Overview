import numpy as np
import math
import scipy as sp
from scipy import stats, optimize

import matplotlib.pyplot as plt

from Distributions.ERADist import ERADist
from Distributions.ERANataf import ERANataf
from EMGM import EMGM

## Sequential Monte Carlo using adaptive conditional sampling (pCN)
"""
---------------------------------------------------------------------------
Created by:
Iason Papaioannou (iason.papaioannou@tum.de)
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
Contact: Antonis Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Version 2021-04
---------------------------------------------------------------------------
Comments:
* The SMC method in combination with a Gaussian Mixture model can only be
  applied for low-dimensional problems, since its accuracy decreases
  dramatically in high dimensions.---------------------------------------------------------------------------
Input:
* N      : number of samples per level
* p      : N/number of chains per level
* log_likelihood  : log-likelihood function
* distr  : Nataf distribution object or
           marginal distribution object of the input variables
* k_init : initial number of Gaussians in the mixture model
* burn   : burn-in period
* tarCoV : target coefficient of variation of the weights
---------------------------------------------------------------------------
Output:
* logcE     : log-evidence
* q         : tempering parameters
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
* k_fin     : final number of Gaussians in the mixture
---------------------------------------------------------------------------
Based on:
1."A sequential particle filter method for static models"
   Chopin
   Biometrika 89 (3) (2002) 539-551

2."Inference for Levy-driven stochastic volatility models via adaptive sequential Monte Carlo"
   Jasra et al.
   Scand. J. Stat. 38 (1) (2011) 1-22

3."Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
---------------------------------------------------------------------------
"""

def SMC_GM(N, p, log_likelihood, distr, k_init, burn, tarCoV):


    if (N*p != np.floor(N*p)) or (1/p != np.floor(1/p)):
        raise RuntimeError('N*p and 1/p must be positive integers. Adjust N and p accordingly')

    # transform to the standard Gaussian space
    if hasattr(distr, 'Marginals'):             # use Nataf transfrom (dependence)
        dim = len(distr.Marginals)              # number of random variables (dimension)
        u2x = lambda u: distr.U2X(u)            # from u to x

    else:       # use distribution information for the transformation (independence)
                # Here we are assuming that all the parameters have the same distribution !!!
                # Adjust accordingly otherwise or use an ERANataf object
        try:
            dim = len(distr)                    # number of random variables (dimension)
            u2x = lambda u: distr[0].icdf(stats.norm.cdf(u))     # from u to x
        except TypeError:
            dim = 1
            u2x = lambda u: distr.icdf(stats.norm.cdf(u))       # from u to x

    # log likelihood in standard space
    loglike_fun = lambda u: log_likelihood(u2x(u))

    # Initialization of variables and storage
    max_it = 100        # estimated number of iterations
    m      = 0          # counter for number of levels

    # Properties of SMC
    nsamlev  = N                        # number of samples
    nchain   = int(nsamlev*p)           # number Markov chains
    lenchain = int(nsamlev/nchain)      # number of samples per Markov chain
    tarESS   = nsamlev/(1+tarCoV**2)    # target effective sample size

    # initialize samples
    logLk   = np.zeros(nsamlev)         # space for evaluations of log likelihood
    accrate = np.zeros(max_it)          # space for acceptance rate
    q       = np.zeros(max_it)          # space for the tempering parameters
    logSk   = np.ones(max_it)           # space for log-expected weights

    samplesU = []                       # samples in normal space
    samplesX = []                       # samples in original space

    ## SMC GM
    #===== Step 1: Perform the first Monte Carlo simulation
    uk = np.random.normal(size=(nsamlev,dim))   # initial samples

    for k in range(nsamlev):
        logLk[k] = loglike_fun(uk[k,:])         # evaluate likelihood

    # save samples
    samplesU.append(uk.T)

    while (q[m] < 1) and (m < max_it):          # adaptively choose q
        m += 1
        #===== Step 2 and 3: compute tempering parameter and weights

        def fun(dq):
            return  np.exp(2*logsumexp(abs(dq)*logLk)-logsumexp(2*abs(dq)*logLk)) - tarESS      # ESS equation

        sol = optimize.root(fun, x0=1, method='lm')

        # if root does not converge try fsolve
        if sol.success:
            dq = abs(sol.x[0])
        else:
            sol = optimize.fsolve(fun, x0=1, full_output=True)
            if sol[2] != 1:
                raise RuntimeError('root and fsolve do not converge')

        if not math.isnan(dq):
            q[m] = min(1, q[m-1]+dq)
        else:
            q[m] = 1
            print('Variable q was set to {}, since it is not possible to find a suitable value\n'.format(q[m]))

        # log-weights
        logwk = (q[m]-q[m-1])*logLk

        #===== Step 4: compute estimate of log-expected w
        logwsum = logsumexp(logwk)
        logSk[m-1] = logwsum - np.log(nsamlev)
        wnork = np.exp(logwk - logwsum)                 # compute normalized weights

        [mu, si, ww] = EMGM(uk.T, wnork.T, k_init)      # fit Gaussian Mixture

        #===== Step 5: resample
        # seeds for chains
        ind = np.random.choice(nsamlev, nchain, replace=True, p=wnork)
        logLk0 = logLk[ind]
        uk0 = uk[ind,:]

        #===== Step 6: perform independent M-H
        count = 0

        # initialize chain acceptance rate
        alphak = np.zeros(nchain)
        # delete previous samples and initialize arrays (not dynamic like in Matlab)
        logLk = np.full(nsamlev+lenchain+burn, np.inf)
        uk = np.full((nsamlev+lenchain+burn, dim), np.inf)

        for k in range(nchain):
            # set seed for chain
            u0 = uk0[k,:]
            logL0 = logLk0[k]

            for j in range(lenchain+burn):
                count += 1
                if j == burn:
                    count -= burn

                # get candidate sample from the fitted GM distribution
                indw = np.random.choice(len(ww), 1, replace=True, p=ww)
                ucand = np.array(stats.multivariate_normal.rvs(mean=(mu[:, indw].T).squeeze(), cov=si[:, :, indw].squeeze()))
                # Evaluate log-likelihood function
                logLcand = loglike_fun(ucand)

                # compute acceptance probability
                logpdfn = np.zeros(len(ww))
                logpdfd = np.zeros(len(ww))
    
                for ii in range(len(ww)):
                    logpdfn[ii] = np.log(ww[ii])+loggausspdf(u0.T, (mu[:,ii].T).squeeze(), si[:,:,ii].squeeze())
                    logpdfd[ii] = np.log(ww[ii])+loggausspdf(ucand.T, (mu[:,ii].T).squeeze(), si[:,:,ii].squeeze())

                logpdfnsum = logsumexp(logpdfn)
                logpdfdsum = logsumexp(logpdfd)
                logpriorn = -0.5*np.dot(ucand,ucand)
                logpriord = -0.5*np.dot(u0,u0)
                alpha     = min(1, np.exp(q[m]*(logLcand-logL0)+logpriorn-logpriord+logpdfnsum-logpdfdsum))
                alphak[k] = alphak[k] + alpha/(lenchain+burn)

                # check if sample is accepted
                uhelp = stats.uniform.rvs()
                if uhelp <= alpha:
                    uk[count-1,:] = ucand
                    logLk[count-1] = logLcand
                    u0  = ucand
                    logL0  = logLcand
                else:
                    uk[count-1,:] = u0
                    logLk[count-1] = logL0

        uk = uk[0:nsamlev,:]
        logLk = logLk[0:nsamlev]

        # save samples
        samplesU.append(uk.T)

        # compute mean acceptance rate of all chains in level m
        accrate[m-1] = np.mean(alphak)

        print('\t *MH-GM accrate = {:.2f}\n'.format(accrate[m-1]))

    k_fin = len(ww)
    l_tot = m+1
    q = q[0:l_tot]
    logSk = logSk[0:l_tot-1]

    # log-evidence
    logcE = np.sum(logSk)

    # transform the samples to the physical/original space
    for i in range(l_tot):
        samplesX.append(u2x(samplesU[i].T).T)

    return [samplesU, samplesX, q, k_fin, logcE]


#===========================================================================
def logsumexp(x, dim='default'):

    x2 = np.copy(x)     # need second variable, maybe numpy overwrites it (same memory)
    if dim == 'default':
        # Determine which dimension sum will use
        dim = np.where(np.asarray(x.shape) != 1)[0].reshape(-1,1)

        if not dim:
            dim = 0
    dim = int(dim)

    # subtract the largest in each column
    y = np.max(x2, axis=dim)
    x2 -= y
    s = y + np.log(np.sum(np.exp(x2), axis=dim))
    try:
        i = np.where([not math.isfinite(k) for k in y])[0]  # if y is an array
    except TypeError:
        i = np.where([not math.isfinite(y)])[0]             # if y is a float
    if i.size > 0:
        s[i] = y[i]

    return s



#===========================================================================
def loggausspdf(X, mu, Sigma):
    X2 = np.copy(X)
    try:
        d = np.size(X, axis=0)
    except IndexError:  # if X is a float
        d = 0
    X2 = X2.reshape(-1,1)
    X2 = X2 - mu.reshape(-1, 1)
    try:
        L = np.linalg.cholesky(Sigma).conj()    # if Sigma is already 2d-array
    except np.linalg.LinAlgError:               # if Sigma is 0d-array
        L = np.linalg.cholesky(np.expand_dims(np.expand_dims(Sigma, 0), 0)).conj()
    Q = np.linalg.solve(L, X2)
    q = np.sum(np.multiply(Q,Q), axis=0)  # quadratic term (M distance)
    c = d * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L)))  # normalization constant
    y = -(c + q) / 2

    return y
