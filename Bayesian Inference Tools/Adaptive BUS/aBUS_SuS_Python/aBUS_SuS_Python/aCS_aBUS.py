import numpy as np
import warnings
np.seterr(all='ignore')

"""
---------------------------------------------------------------------------
adaptive conditional sampling algorithm for aBUS
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu (fonglin.wu@tum.de)
Matthias Willer (matthias.willer@tum.de)
Felipe Uribe (felipe.uribe@tum.de)
Iason Papaioannou (iason.papaioannou@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-05
---------------------------------------------------------------------------
Input:
* N         : number of samples to be generated
* lam       : scaling parameter lambda
* h         : current intermediate level
* u_j       : seeds used to generate the new samples
* log_L_fun : log-likelihood function
* l         : =-log(c) ~ scaling constant of BUS for the current level
* gl        : limit state function in the standard space
---------------------------------------------------------------------------
Output:
* u_jp1      : next level samples
* leval      : log-likelihood function of the new samples
* new_lambda : next scaling parameter lambda
* sigma      : spread of the proposal
* accrate    : acceptance rate of the samples
---------------------------------------------------------------------------
NOTES
* The way the initial standard deviation is computed can be changed in line 69.
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

def aCS_aBUS(N, lambd_old, tau, theta_seeds, log_L_fun, logl_hat, h_LSF):
    n, Ns = theta_seeds.shape   # dimension and number of seeds
    
    # number of samples per chain
    Nchain = np.ones(Ns,dtype=int)*int(np.floor(N/Ns))
    Nchain[:np.mod(N,Ns)] = Nchain[:np.mod(N,Ns)]+1

    # initialization
    theta_chain = np.zeros((n,N))         # generated samples 
    leval       = np.zeros(N)             # store lsf evaluations
    acc         = np.zeros(N,dtype=int)   # store acceptance

    # initialization
    Na      = int(np.ceil(100*Ns/N))             # number of chains after which the proposal is adapted 
    mu_acc  = np.zeros(int(np.floor(Ns/Na)+1))   # store acceptance
    hat_acc = np.zeros(int(np.floor(Ns/Na)))     # average acceptance rate of the chains
    lambd   = np.zeros(int(np.floor(Ns/Na)+1))   # scaling parameter \in (0,1)

    # 1. compute the standard deviation
    opc = 'a'
    if opc == 'a': # 1a. sigma = ones(n,1)
        sigma_0 = np.ones(n)
    elif opc == 'b': # 1b. sigma = sigma_hat (sample standard deviations)
        sigma_0 = np.std(theta_seeds, axis=1)
    else:
        raise RuntimeError('Choose a or b')

    # 2. iteration
    star_a   = 0.44     # optimal acceptance rate 
    lambd[0] = lambd_old # initial scaling parameter \in (0,1)

    # a. compute correlation parameter
    i         = 0                                          # index for adaptation of lambda
    sigma     = np.minimum(lambd[i]*sigma_0, np.ones(n))   # Ref. 1 Eq. 23
    rho       = np.sqrt(1-sigma**2)                        # Ref. 1 Eq. 24
    mu_acc[i] = 0 

    # b. apply conditional sampling
    for k in range(1, Ns+1):
        idx = sum(Nchain[:k-1])   # beginning of each chain total index

        # initial chain values
        theta_chain[:,idx] = theta_seeds[:,k-1]

        # initial log-like evaluation
        leval[idx] = log_L_fun(theta_chain[:,idx])
        
        for t in range(1, Nchain[k-1]): 
            # current state
            theta_t = theta_chain[:,idx+t-1]

            # generate candidate sample        
            v_star = np.random.normal(loc=rho*theta_t, scale=sigma)    
            
            # evaluate loglikelihood function
            log_l_star = log_L_fun(v_star)

            # evaluate limit state function 
            heval = h_LSF(v_star[-1].reshape(-1), logl_hat, log_l_star)   # evaluate limit state function    

            # accept or reject sample
            if heval <= tau:
                theta_chain[:,idx+t] = v_star       # accept the candidate in observation region           
                leval[idx+t]         = log_l_star   # store the loglikelihood evaluation
                acc[idx+t]           = 1            # note the acceptance
            else:
                theta_chain[:,idx+t] = theta_t          # reject the candidate and use the same state
                leval[idx+t]         = leval[idx+t-1]   # store the loglikelihood evaluation    
                acc[idx+t]           = 0                # note the rejection

        # average of the accepted samples for each seed 'mu_acc'
        # here the warning "Mean of empty slice" is not an issue        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mu_acc[i] = mu_acc[i] + np.minimum(1, np.mean(acc[idx+1:idx+Nchain[k-1]])) 
        
        if np.mod(k,Na) == 0:
            if Nchain[k-1] > 1:
                # c. evaluate average acceptance rate
                hat_acc[i] = mu_acc[i]/Na   # Ref. 1 Eq. 25
            
                # d. compute new scaling parameter
                zeta       = 1/np.sqrt(i+1)   # ensures that the variation of lambda(i) vanishes
                lambd[i+1] = np.exp(np.log(lambd[i]) + zeta*(hat_acc[i]-star_a))  # Ref. 1 Eq. 26
            
                # update parameters
                sigma = np.minimum(lambd[i+1]*sigma_0, np.ones(n))  # Ref. 1 Eq. 23
                rho   = np.sqrt(1-sigma**2)                       # Ref. 1 Eq. 24
            
                # update counter
                i += 1

    # next level lambda
    new_lambda = lambd[i]

    # compute mean acceptance rate of all chains
    accrate = np.sum(acc)/(N-Ns)

    return theta_chain, leval, new_lambda, sigma, accrate