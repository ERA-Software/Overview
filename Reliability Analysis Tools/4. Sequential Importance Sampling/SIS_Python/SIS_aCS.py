import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
"""
---------------------------------------------------------------------------
Sequential importance sampling with adaptive conditional sampling
---------------------------------------------------------------------------
Created by:
Sebastian Geyer
Max Ehre
Iason Papaioannou
Daniel Straub

Assistant Developers:
Matthias Willer
Peter Kaplan
Luca Sardi
Daniel Koutas
Ivan Olarte Rodriguez

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Current version 2023-04
* Modification of Sensitivity Analysis Calls
---------------------------------------------------------------------------
Comments:
* The SIS-method in combination with the adaptive conditional MH sampler
  (aCS) performs well in high-dimensions. For low-dimensional problems, the
  Gaussian mixture proposal should be chosen over aCS.
* The way the initial standard deviation is computed can be changed in line 79.
  By default we use option 'a' (it is equal to one).
  In option 'b', it is computed from the seeds.
---------------------------------------------------------------------------
Input:
* N                    : number of samples per level
* p                    : N/number of chains per level
* g_fun                : limit state function
* distr                : Nataf distribution object or
                         marginal distribution object of the input variables
* burn                 : burn-in period
* tarCoV               : target coefficient of variation of the weights
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples
---------------------------------------------------------------------------
Output:
* Pr       : probability of failure
* l_tot    : total number of levels
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* W_final  : final weights
* f_s_iid  : Independent Identically Distributed samples generated from last step
---------------------------------------------------------------------------
Based on:
1. "Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
---------------------------------------------------------------------------
"""

def SIS_aCS(N, p, g_fun, distr, burn, tarCoV, samples_return):
    if (N*p != np.fix(N*p)) or (1/p != np.fix(1/p)):
        raise RuntimeError('N*p and 1/p must be positive integers. Adjust N and p accordingly')

    # initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):   # use Nataf transform (dependence)
        dim = len(distr.Marginals)     # number of random variables (dimension)
        u2x = lambda u: distr.U2X(u)   # from u to x

    elif isinstance(distr[0], ERADist):   # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise or create an ERANataf object
        dim = len(distr)                    # number of random variables (dimension)
        u2x = lambda u: distr[0].icdf(sp.stats.norm.cdf(u))   # from u to x
    else:
        raise RuntimeError('Incorrect distribution. Please create an ERADist/Nataf object!')

    # LSF in standard space
    g = lambda u: g_fun(np.array(u2x(u),ndmin=2))

    # Initialization of variables and storage
    max_it   = 100              # estimated number of iterations
    m        = 0                # counter for number of levels
    samplesU = []               # space for samples in U-space

    # Properties of SIS
    nsamlev  = N                    # number of samples
    nchain   = int(nsamlev*p)       # number Markov chains
    lenchain = int(nsamlev/nchain)  # number of samples per Markov chain

    # parameters for adaptive MCMC
    opc        = 'a'   # way to estimate the std for the aCS (see comments)
    adapflag   = 1
    adapchains = int(np.ceil(100*nchain/nsamlev))
    lam        = 0.6

    # initialize samples
    accrate = np.zeros([max_it])          # space for acceptance rate
    Sk      = np.ones([max_it])           # space for expected weights
    sigmak  = np.zeros([max_it])          # space for sigmak

    # Step 1
    # Perform the first Monte Carlo simulation
    uk = sp.stats.norm.rvs(size=(nsamlev,dim))    # initial samples
    
    gk = g(uk)                                  # evaluations of g

    # save samples
    samplesU.append(uk)

    # set initial subset and failure level
    gmu = np.mean(gk)

    # Iteration
    for m in range(max_it):
        # Step 2 and 3: compute sigma and weights
        if m == 0:
            func        = lambda x: abs(np.std(sp.stats.norm.cdf(-gk/x))/ \
                                        np.mean(sp.stats.norm.cdf(-gk/x))-tarCoV)
            sigma2      = sp.optimize.fminbound(func, 0, 10.0*gmu)
            sigmak[m+1] = sigma2
            wk          = sp.stats.norm.cdf(-gk/sigmak[m+1])
        else:
            func        = lambda x: abs(np.std(sp.stats.norm.cdf(-gk/x)/sp.stats.norm.cdf(-gk/sigmak[m]))/ \
                                        np.mean(sp.stats.norm.cdf(-gk/x)/sp.stats.norm.cdf(-gk/sigmak[m]))-tarCoV)
            sigma2      = sp.optimize.fminbound(func, 0, sigmak[m])
            sigmak[m+1] = sigma2
            wk          = sp.stats.norm.cdf(-gk/sigmak[m+1])/sp.stats.norm.cdf(-gk/sigmak[m])

        # Step 4: compute estimate of expected w
        Sk[m] = np.mean(wk)
        # Exit algorithm if no convergence is achieved
        if Sk[m] == 0:
            break
        wnork = wk/Sk[m]/nsamlev    # compute normalized weights

        # Step 5: resample
        # seeds for chains
        ind = np.random.choice(np.arange(nsamlev),nchain,True,wnork)
        gk0 = gk[ind]
        uk0 = uk[ind,:]

        # Step 6: perform aCS
        # compute the standard deviation
        if opc == 'a': # 1a. sigma = ones(n,1)
            sigmaf = 1
        elif opc == 'b': # 1b. sigma = sigma_hat (sample standard deviations)
            muf    = np.mean(np.repmat(wnork,dim,1)*uk,1)
            sigmaf = np.zeros((1,dim))
            for k in range(nsamlev):
                sigmaf = sigmaf + wnork[k]*(uk[k,:]-muf)**2
        else:
            raise RuntimeError('Choose a or b')

        # compute parameter rho
        sigmafk = min(lam*sigmaf, 1)
        rhok    = np.sqrt(1-sigmafk**2)
        counta  = 0
        count   = 0

        # initialize chain acceptance rate
        alphak = np.zeros([nchain])         # initialize chain acceptance rate
        gk     = np.zeros([nsamlev])        # delete previous samples
        uk     = np.zeros([nsamlev,dim])    # delete previous samples
        for k in range(nchain):
            # set seed for chain
            u0 = uk0[k,:]
            g0 = gk0[k]
            for i in range(lenchain+burn):
                if i == burn:
                    count = count-burn

                # get candidate sample from conditional normal distribution
                ucand = np.random.normal(loc=rhok*u0, scale=sigmafk)
                #ucand = np.random.multivariate_normal(u0*lam, np.identity(dim) * (1 - lam ** 2))

                # Evaluate limit-state function
                gcand = g(ucand)

                # compute acceptance probability
                alpha     = min(1,sp.stats.norm.cdf(-gcand/sigmak[m+1])/sp.stats.norm.cdf(-g0/sigmak[m+1]))
                alphak[k] = alphak[k] + alpha/(lenchain+burn)

                # check if sample is accepted
                uhelp = sp.stats.uniform.rvs()
                if uhelp <= alpha:
                    uk[count,:] = ucand
                    gk[count]   = gcand
                    u0 = ucand
                    g0 = gcand
                else:
                    uk[count,:] = u0
                    gk[count]   = g0

                count += 1

            # adapt the chain correlation
            if adapflag == 1:
                #check whether to adapt now
                if (k+1) % adapchains == 0:

                    # mean acceptance rate of last adap_chains
                    alpha_mu = np.mean(alphak[k-adapchains+1:k+1])
                    counta   = counta+1
                    gamma    = counta**(-0.5)
                    lam      = min(np.exp(np.log(lam)+gamma*(alpha_mu-0.44)), 1)

                    # compute parameter rho
                    sigmafk = min(lam*sigmaf,1)
                    rhok    = np.sqrt(1-sigmafk**2)

        uk = uk[:nsamlev,:]
        gk = gk[:nsamlev]

        # samples return - all / all by default
        if samples_return not in [0, 1]:
            samplesU.append(uk)

        # compute mean acceptance rate of all chains in level m
        accrate[m] = np.mean(alphak)

        if sigmak[m+1] == 0:
            COV_Sl = np.nan
        else:
            COV_Sl = np.std( (gk < 0)/sp.stats.norm.cdf(-gk/sigmak[m+1]))/ \
                     np.mean((gk < 0)/sp.stats.norm.cdf(-gk/sigmak[m+1]))

        print('\nCOV_Sl =', COV_Sl)
        print('\t*aCS sigma =', sigmafk, '\t*aCS accrate =', accrate[m])
        if COV_Sl < tarCoV:
            # samples return - last
            if samples_return == 1:
                samplesU[0] = uk
            break

    # Samples return - all by default message
    if samples_return not in [0, 1, 2]:
        print("\n-Invalid input for samples return, all samples are returned by default")

    # required steps
    l_tot = m + 2

    # Calculation of the Probability of failure
    const = np.prod(Sk)
    I_final = gk <= 0
    W_final = 1 / sp.stats.norm.cdf(-gk / sigmak[m + 1])
    Pr = const * np.mean(I_final * W_final)

    # transform the samples to the physical/original space
    if samples_return != 0:
        samplesX = [u2x(samplesU[i][:,:]) for i in range(len(samplesU))]

    # resample 10000 failure samples with final weights W
    weight_id = np.random.choice(list(np.nonzero(I_final))[0], 10000, list(W_final[I_final]))
    f_s_iid = samplesX[-1][weight_id, :]

    if samples_return == 0:
        samplesU = list()  # empty return samples U
        samplesX = list()  # and X

    # Convergence is not achieved message
    if m == max_it:
        print("\n-Exit with no convergence at max iterations \n")

    return Pr, l_tot, samplesU, samplesX, W_final, f_s_iid
