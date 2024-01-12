import numpy as np
import scipy as sp
np.seterr(all='ignore')

# Make sure to include aCS.py in your workspace
from aCS_aBUS import aCS_aBUS

# Make sure ERADist, ERANataf classes are in the path
# https://www.bgu.tum.de/era/software/eradist/
from ERADist import ERADist
from ERANataf import ERANataf

"""
---------------------------------------------------------------------------
adaptive BUS with subset simulation
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu
Matthias Willer
Felipe Uribe
Luca Sardi

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Last Version 2021-03
* Adaptation to new ERANataf class
* Use of log-evidence
---------------------------------------------------------------------------
Input:
* N              : number of samples per level
* p0             : conditional probability of each subset
* log_likelihood : log-Likelihood function of the problem at hand
* T_nataf        : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* h        : intermediate levels of the subset simulation method
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* logcE    : model log-evidence
* c        : scaling constant that holds 1/c >= Lmax
* sigma    : last spread of the proposal
---------------------------------------------------------------------------
Based on:
1."Bayesian inference with subset simulation: strategies and improvements"
   Betz et al.
   Computer Methods in Applied Mechanics and Engineering 331 (2018) 72-93.
2."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
---------------------------------------------------------------------------
"""

def aBUS_SuS(N, p0, log_likelihood, distr):
    if (N*p0 != np.fix(N*p0)) or (1/p0 != np.fix(1/p0)):
        raise RuntimeError('N*p0 and 1/p0 must be positive integers. Adjust N and p0 accordingly')

    # initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):    # use Nataf transform (dependence)
        n   = len(distr.Marginals)+1   # number of random variables + p Uniform variable of BUS
        u2x = lambda u: distr.U2X(u)   # from u to x

    elif isinstance(distr[0], ERADist):   # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise or use an ERANataf object
        n   = len(distr)+1                # number of random variables + p Uniform variable of BUS
        u2x = lambda u: distr[0].icdf(sp.stats.norm.cdf(u))   # from u to x
    else:
        raise RuntimeError('Incorrect distribution. Please create an ERADist/Nataf object!')

    # limit state funtion for the observation event (Ref.1 Eq.12)
    # log likelihood in standard space
    log_L_fun = lambda u: log_likelihood(u2x(u[0:n-1]).flatten())

    # LSF
    h_LSF = lambda pi_u, logl_hat, log_L: np.log(sp.stats.norm.cdf(pi_u)) + logl_hat - log_L
    # note that h_LSF = log(pi) + l(i) - leval
    # where pi = normcdf(u_j(end,:)) is the standard uniform variable of BUS

    # initialization of variables
    i        = 0                         # number of conditional level
    lam      = 0.6                       # initial scaling parameter \in (0,1)
    max_it   = 20                        # maximum number of iterations
    samplesU = {'seeds': list(),
                'total': list()}
    samplesX = list()
    #
    geval = np.empty(N)                # space for the LSF evaluations
    leval = np.empty(N)                # space for the LSF evaluations
    h     = np.empty(max_it)           # space for the intermediate leveles
    prob  = np.empty(max_it)           # space for the failure probability at each level

    # aBUS-SuS procedure
    # initial MCS step
    print('Evaluating log-likelihood function ...\t', end='')
    u_j   = np.random.normal(size=(n,N))  # N samples from the prior distribution
    leval = [log_L_fun(u_j[:,i]) for i in range(N)]
    leval = np.array(leval)
    print('Done!')
    logl_hat = max(leval)   # =-log(c) (Ref.1 Alg.5 Part.3)
    print('Initial maximum log-likelihood: ', logl_hat)

    # SuS stage
    h[i] = np.inf
    while h[i] > 0:
        # increase counter
        i += 1

        # compute the limit state function (Ref.1 Eq.12)
        geval = h_LSF(u_j[-1,:], logl_hat, leval)   # evaluate LSF (Ref.1 Eq.12)

        # sort values in ascending order
        idx = np.argsort(geval)
        # gsort[j,:] = geval[idx]

        # order the samples according to idx
        u_j_sort = u_j[:,idx]
        samplesU['total'].append(u_j_sort.T)   # store the ordered samples

        # intermediate level
        h[i] = np.percentile(geval, p0*100, interpolation='midpoint')

        # number of failure points in the next level
        nF = int(sum(geval <= max(h[i],0)))

        # assign conditional probability to the level
        if h[i] < 0:
            h[i]      = 0
            prob[i-1] = nF/N
        else:
            prob[i-1] = p0
        print('\n-Threshold level ', i, ' = ', h[i])

        # select seeds and randomize the ordering (to avoid bias)
        seeds     = u_j_sort[:,:nF]
        idx_rnd   = np.random.permutation(nF)
        rnd_seeds = seeds[:,idx_rnd]            # non-ordered seeds
        samplesU['seeds'].append(seeds.T)         # store seeds

        # sampling process using adaptive conditional sampling
        u_j, leval, lam, sigma, accrate = aCS_aBUS(N, lam, h[i], rnd_seeds, log_L_fun, logl_hat, h_LSF)
        print('\t*aCS lambda =', lam, '\t*aCS sigma =', sigma[0], '\t*aCS accrate =', accrate)

        # update the value of the scaling constant (Ref.1 Alg.5 Part.4d)
        l_new    = max(logl_hat, max(leval))
        h[i]     = h[i] - logl_hat + l_new
        logl_hat = l_new
        print(' Modified threshold level ', i, '= ', h[i])

        # decrease the dependence of the samples (Ref.1 Alg.5 Part.4e)
        p = np.random.uniform(low=np.zeros(N),
                              high=np.min([np.ones(N), np.exp(leval - logl_hat + h[i])], axis=0))
        u_j[-1,:] = sp.stats.norm.ppf(p)   # to the standard space

    # number of intermediate levels
    m = i

    # store final posterior samples
    samplesU['total'].append(u_j.T)  # store final failure samples (non-ordered)

    # delete unnecesary data
    if m < max_it:
        prob = prob[:m]
        h    = h[:m+1]

    # acceptance probability and evidence (Ref.1 Alg.5 Part.6and7)
    log_p_acc = np.sum(np.log(prob))
    c     = 1/np.exp(logl_hat)         # l = -log(c) = 1/max(likelihood)
    logcE    = log_p_acc+logl_hat     # exp(l) = max(likelihood)

    # transform the samples to the physical (original) space
    for i in range(m+1):
        pp = sp.stats.norm.cdf(samplesU['total'][i][:,-1])
        samplesX.append(np.concatenate((u2x(samplesU['total'][i][:,:-1]), pp.reshape(-1,1)), axis=1))

    return h, samplesU, samplesX, logcE, c, sigma
