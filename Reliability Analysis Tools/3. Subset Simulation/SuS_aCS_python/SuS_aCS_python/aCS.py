import numpy as np
#import scipy as sp
import warnings
"""
---------------------------------------------------------------------------
Adaptive conditional sampling algorithm
---------------------------------------------------------------------------
Created by:
Matthias Willer
Felipe Uribe
Iason Papaioannou
Luca Sardi

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2021-03
* input dimensions of limit state function changed to rows*columns =
  samples*dimensions (adaptation to new ERANataf class)
---------------------------------------------------------------------------
Input:
* N       : number of samples to be generated
* old_lam : scaling parameter lambda
* b       : actual intermediate level
* u_j     : seeds used to generate the new samples
* G_LSF   : limit state function in the standard space
---------------------------------------------------------------------------
Output:
* u_jp1      : next level samples
* geval      : limit state function evaluations of the new samples
* new_lambda : next scaling parameter lambda
* sigma      : spread of the proposal
* accrate    : acceptance rate of the method
---------------------------------------------------------------------------
NOTES
* The way the initial standard deviation is computed can be changed in line 67.
By default we use option 'a' (it is equal to one).
In option 'b', it is computed from the seeds.
* The final accrate might differ from the target 0.44 when no adaptation is
performed. Since at the last level almost all seeds are already in the failure
domain, only a few are selected to complete the required N samples. The final
accrate is computed for those few samples
---------------------------------------------------------------------------
Based on:
1."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
"""

def aCS(N, old_lam, b, u_j, G_LSF):
    # %% Initialize variables
    n  = np.size(u_j,axis=1)     # number of uncertain parameters
    Ns = np.size(u_j,axis=0)     # number of seeds
    Na = int(np.ceil(100*Ns/N))  # number of chains after which the proposal is adapted

    # number of samples per chain
    Nchain = np.ones(Ns,dtype=int)*int(np.floor(N/Ns))
    Nchain[:np.mod(N,Ns)] = Nchain[:np.mod(N,Ns)]+1

    # initialization
    u_jp1  = np.zeros((N,n))                       # generated samples
    geval  = np.zeros(N)                           # store lsf evaluations
    acc    = np.zeros(N,dtype=int)                 # store acceptance
    mu_acc = np.zeros(int(np.floor(Ns/Na)+1))      # store acceptance
    hat_a  = np.zeros(int(np.floor(Ns/Na)+1))      # average acceptance rate of the chains
    lam    = np.zeros(int(np.floor(Ns/Na)+1))      # scaling parameter \in (0,1)

    # %% 1. compute the standard deviation
    opc = 'a'
    if opc == 'a': # 1a. sigma = ones(n,1)
        sigma_0 = np.ones(n)
    elif opc == 'b': # 1b. sigma = sigma_hat (sample standard deviations)
        mu_hat  = np.mean(u_j,axis=1)    # sample mean
        var_hat = np.zeros(n)            # sample std
        for i in range(n):  # dimensions
            for k in range(Ns):  # samples
                var_hat[i] = var_hat[i] + (u_j[k,i]-mu_hat[i])**2
            var_hat[i] = var_hat[i]/(Ns-1)

        sigma_0 = np.sqrt(var_hat)
    else:
        raise RuntimeError('Choose a or b')

    # %% 2. iteration
    star_a = 0.44      # optimal acceptance rate
    lam[0] = old_lam   # initial scaling parameter \in (0,1)

    # a. compute correlation parameter
    i         = 0                                        # index for adaptation of lambda
    sigma     = np.minimum(lam[i]*sigma_0, np.ones(n))   # Ref. 1 Eq. 23
    rho       = np.sqrt(1-sigma**2)                      # Ref. 1 Eq. 24
    mu_acc[i] = 0

    # b. apply conditional sampling
    for k in range(1, Ns+1):
        idx          = sum(Nchain[:k-1])     # beginning of each chain index
        #acc[idx]     = 1                    # seed acceptance
        u_jp1[idx,:] = u_j[k-1,:]            # pick a seed at random
        geval[idx]   = G_LSF(u_jp1[idx,:])   # store the lsf evaluation

        for t in range(1, Nchain[k-1]):
            # generate candidate sample
            v = np.random.normal(loc=rho*u_jp1[idx+t-1,:], scale=sigma)
            #v = sp.stats.norm.rvs(loc=rho*u_jp1[:,idx+t-1], scale=sigma)
            #v = sp.stats.multivariate_normal.rvs(mean=rho*u_jp1[:,idx+t-1], cov=np.diag(sigma**2))

            # accept or reject sample
            Ge = G_LSF(v)
            if Ge <= b:
                u_jp1[idx+t,:] = v         # accept the candidate in failure region
                geval[idx+t]   = Ge        # store the lsf evaluation
                acc[idx+t]     = 1         # note the acceptance
            else:
                u_jp1[idx+t,:] = u_jp1[idx+t-1,:]   # reject the candidate and use the same state
                geval[idx+t]   = geval[idx+t-1]     # store the lsf evaluation
                acc[idx+t]     = 0                  # note the rejection

        # average of the accepted samples for each seed 'mu_acc'
        # here the warning "Mean of empty slice" is not an issue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mu_acc[i] = mu_acc[i] + np.minimum(1, np.mean(acc[idx+1:idx+Nchain[k-1]]))

        if np.mod(k,Na) == 0:
            if Nchain[k-1] > 1:
                # c. evaluate average acceptance rate
                hat_a[i] = mu_acc[i]/Na   # Ref. 1 Eq. 25

                # d. compute new scaling parameter
                zeta     = 1/np.sqrt(i+1)   # ensures that the variation of lambda(i) vanishes
                lam[i+1] = np.exp(np.log(lam[i]) + zeta*(hat_a[i]-star_a))  # Ref. 1 Eq. 26

                # update parameters
                sigma = np.minimum(lam[i+1]*sigma_0, np.ones(n))  # Ref. 1 Eq. 23
                rho   = np.sqrt(1-sigma**2)                       # Ref. 1 Eq. 24

                # update counter
                i = i+1

    # next level lambda
    new_lambda = lam[i]

    # compute mean acceptance rate of all chains
    if i != 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            accrate = np.mean(hat_a[:i-1])

    else:   # no adaptation
        accrate = sum(acc[:np.mod(N,Ns)])/np.mod(N,Ns)   # only for the few generated samples

    return u_jp1, geval, new_lambda, sigma, accrate
# %%END
