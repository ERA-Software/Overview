import numpy as np
import math
import scipy as sp
from scipy import stats, optimize

from ERADist import ERADist
from ERANataf import ERANataf
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
* The SMC method in combination with the adaptive conditional M-H (or pCN) sampler
  (aCS) performs well in high-dimensions. For low-dimensional problems, the
  Gaussian mixture proposal should be chosen over aCS.
* The way the initial standard deviation is computed can be changed in line 76.
  By default we use option 'a' (it is equal to one).
  In option 'b', it is computed from the seeds.
---------------------------------------------------------------------------
Input:
* N      : number of samples per level
* p      : N/number of chains per level
* log_likelihood  : log-likelihood function
* distr  : Nataf distribution object or
           marginal distribution object of the input variables
* burn   : burn-in period
* tarCoV : target coefficient of variation of the weights
---------------------------------------------------------------------------
Output:
* logcE       : log-evidence
* q    : tempering parameters
* samplesU : list with the samples in the standard normal space
* samplesX : list with the samples in the original space
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

def SMC_aCS(N, p, log_likelihood, distr, burn, tarCoV):

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
    nsamlev  = N                            # number of samples
    nchain   = int(nsamlev*p)                    # number Markov chains
    lenchain = int(nsamlev/nchain)          # number of samples per Markov chain
    tarESS   = nsamlev/(1+tarCoV**2)        # target effective sample size

    # initialize samples
    logLk   = np.zeros(nsamlev)         # space for evaluations of log likelihood
    accrate = np.zeros(max_it)          # space for acceptance rate
    q       = np.zeros(max_it)          # space for the tempering parameters
    logSk   = np.ones(max_it)           # space for log-expected weights
    opc     = 'a'                       # way to estimate the std for the aCS (see comments)
    samplesU = []                       # samples in normal space
    samplesX = []                       # samples in original space

    # parameters for adaptive MCMC
    adapflag   = 1
    adapchains = int(np.ceil(100*nchain/nsamlev))  # number of chains after which the proposal is adapted
    lambda_     = 0.6

    ## SMC aCS
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

        # if root does not converge, try fsolve
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

        wnork = np.exp(logwk - logwsum)          # compute normalized weights

        #===== Step 5: resample
        # seeds for chains
        ind = np.random.choice(nsamlev,nchain,replace=True, p=wnork)
        logLk0 = logLk[ind]
        uk0 = uk[ind]

        #===== Step 6: perform aCS
        # compute initial standard deviation
        if opc == 'a':          # 1a. sigma = np.ones(n,1)
            sigmaf = 1

        elif opc == 'b':        # 1b. sigma = sigma_hat (sample standard deviations)
         muf    = np.mean(np.multiply(np.tile(wnork, [dim, 1]).T, uk), 0)
         sigmaf = np.zeros((1,dim))
         for k in range(nsamlev):
            sigmaf += wnork[k]*(uk[k,:]-muf)**2

        else:
            raise RuntimeError('Choose a or b')

        # compute parameter rho
        sigmafk = min(lambda_*sigmaf, 1)
        rhok    = np.sqrt(1-sigmafk**2)
        counta  = 0
        count   = 0

        # initialize chain acceptance rate
        alphak = np.zeros(nchain)
        # delete previous samples and initialize arrays (not dynamic like in Matlab)
        logLk = np.full(nsamlev+lenchain+burn, np.inf)
        uk = np.full((nsamlev+lenchain+burn, dim), np.inf)

        for k in range(nchain):
            # set seed for chain
            u0 = uk0[k]
            logL0 = logLk0[k]

            for j in range(lenchain+burn):
                count += 1
                if j == burn:
                    count -= burn

                # get candidate sample from conditional normal distribution
                ucand = stats.norm.rvs(loc=rhok*u0, scale=np.sqrt(1-rhok**2))

                # Evaluate log-likelihood function
                logLcand = loglike_fun(ucand)

                # compute acceptance probability
                alpha     = min(1, np.exp(q[m]*(logLcand-logL0)))
                alphak[k] = alphak[k] + alpha / (lenchain+burn)

                # check if sample is accepted
                uhelp = stats.uniform.rvs()

                if uhelp <= alpha:
                    uk[count-1, :] = ucand
                    logLk[count-1] = logLcand
                    u0  = ucand
                    logL0  = logLcand
                else:
                    uk[count-1, :] = u0
                    logLk[count-1] = logL0

            # adapt the chain correlation
            if adapflag == 1:
                # check whether to adapt now
                if np.mod(k,adapchains) == 0 and k != 0:
                    # mean acceptance rate of last adap_chains
                    alpha_mu = np.mean(alphak[(k-adapchains+1):(k+1)])
                    counta   = counta+1
                    gamma    = np.power(counta, -0.5)
                    lambda_  = np.exp(np.log(lambda_) + gamma*(alpha_mu-0.44))

                    # compute parameter rho
                    sigmafk = min(lambda_*sigmaf, 1)
                    rhok    = np.sqrt(1 - sigmafk**2)

        uk = uk[0:nsamlev, :]
        logLk = logLk[0:nsamlev]

        # save samples
        samplesU.append(uk.T)

        # compute mean acceptance rate of all chains in level m
        accrate[m-1] = np.mean(alphak)

        print('\t*aCS sigma = {} \t *aCS accrate = {}\n'.format(sigmafk, accrate[m-1]))

    l_tot = m+1
    q = q[0:l_tot]
    logSk = logSk[0:l_tot-1]

    # log-evidence
    logcE = np.sum(logSk)

    # transform the samples to the physical/original space
    for i in range(l_tot):
        samplesX.append(u2x(samplesU[i].T).T)

    return [samplesU, samplesX, q, logcE]


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
