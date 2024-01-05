import numpy as np
import scipy as sp
np.seterr(all='ignore')

# Make sure to include aCS.py in your workspace
from aCS import aCS

# Make sure ERADist, ERANataf classes are in the path
# https://www.bgu.tum.de/era/software/eradist/
from ERADist import ERADist
from ERANataf import ERANataf

"""
---------------------------------------------------------------------------
BUS with subset simulation
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
* N               : number of samples per level
* p0              : conditional probability of each subset
* c               : scaling constant of the BUS method
* log_likelihood  : log likelihood function of the problem at hand
* T_nataf         : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* h        : intermediate levels of the subset simulation method
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* logcE    : model log-evidence
* sigma    : final spread of the proposal
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SubSim"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
---------------------------------------------------------------------------
"""

def BUS_SuS(N, p0, c, log_likelihood, distr):
    if (N*p0 != np.fix(N*p0)) or (1/p0 != np.fix(1/p0)):
        raise RuntimeError('N*p0 and 1/p0 must be positive integers. Adjust N and p0 accordingly')

    # %% initial check if there exists a Nataf object
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

    # limit state function in the standard space
    ell   = np.log(1/c)
    h_LSF = lambda u: np.log(sp.stats.norm.cdf(u[-1])) + ell - log_likelihood(u2x(u[0:n-1]).flatten())

    # initialization of variables
    j        = 0                         # number of conditional level
    lam      = 0.6                       # initial scaling parameter \in (0,1)
    max_it   = 20                        # maximum number of iterations
    samplesU = {'seeds': list(),
                'total': list()}
    samplesX = list()
    #
    geval = np.empty(N)            # space for the LSF evaluations
    gsort = np.empty((max_it,N))   # space for the sorted LSF evaluations
    h     = np.empty(max_it)       # space for the intermediate leveles
    prob  = np.empty(max_it)       # space for the failure probability at each level

    # BUS-SuS procedure
    # initial MCS step
    print('Evaluating performance function:\t', end='')
    u_j   = np.random.normal(0,1,size=(n,N))      # samples in the standard space
    geval = [h_LSF(u_j[:,i]) for i in range(N)]
    geval = np.array(geval)
    print('OK!')

    # SuS stage
    h[j] = np.inf
    while (h[j] > 0 and j < max_it):
        # next level
        j += 1

        # sort values in ascending order
        idx        = np.argsort(geval)
        gsort[j,:] = geval[idx]

        # order the samples according to idx
        u_j_sort = u_j[:,idx]
        samplesU['total'].append(u_j_sort.T)   # store the ordered samples

        # intermediate level
        h[j] = np.percentile(geval, p0*100, interpolation='midpoint')

        # number of failure points in the next level
        nF = int(sum(geval <= max(h[j],0)))

        # assign conditional probability to the level
        if h[j] <= 0:
            h[j]      = 0
            prob[j-1] = nF/N
        else:
            prob[j-1] = p0
        print('\n-Threshold intermediate level ', j-1, ' = ', h[j])

        # select seeds and randomize the ordering (to avoid bias)
        seeds     = u_j_sort[:,:nF]
        idx_rnd   = np.random.permutation(nF)
        rnd_seeds = seeds[:,idx_rnd]            # non-ordered seeds
        samplesU['seeds'].append(seeds.T)         # store seeds

        # sampling process using adaptive conditional sampling
        u_j, geval, lam, sigma, accrate = aCS(N, lam, h[j], rnd_seeds, h_LSF)
        print('\t*aCS lambda =', lam, '\t*aCS sigma =', sigma[0], '\t*aCS accrate =', accrate)

    m = j
    samplesU['total'].append(u_j.T)  # store final failure samples (non-ordered)

    # delete unnecesary data
    if m < max_it:
        gsort = gsort[:m,:]
        prob  = prob[:m]
        h     = h[:m+1]

    # acceptance probability and evidence
    log_p_acc = np.sum(np.log(prob))
    logcE    = log_p_acc-np.log(c)

    # transform the samples to the physical (original) space
    for i in range(m+1):
        pp = sp.stats.norm.cdf(samplesU['total'][i][:,-1])
        samplesX.append(np.concatenate((u2x(samplesU['total'][i][:,:-1]), pp.reshape(-1,1)), axis=1))

    return h, samplesU, samplesX, logcE, sigma
