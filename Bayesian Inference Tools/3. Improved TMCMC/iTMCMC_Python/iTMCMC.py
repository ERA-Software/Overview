import numpy as np
import scipy as sp
from resampling_index import resampling_index
np.seterr(all='ignore')

# Make sure ERADist, ERANataf classes are in the path
# https://www.bgu.tum.de/era/software/eradist/
from ERADist import ERADist
from ERANataf import ERANataf

"""
---------------------------------------------------------------------------
improved Transitional Markov Chain Monte Carlo 
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
* Use of log evidence
---------------------------------------------------------------------------
Input:
* Ns             : number of samples per level
* Nb             : number of samples for burn-in
* log_likelihood : log-likelihood function of the problem at hand
* T_nataf        : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* q        : array of tempering parameter for each level
* logcE    : model log-evidence
---------------------------------------------------------------------------
References:
1."Transitional Markov Chain Monte Carlo: Observations and improvements".
   Wolfgang Betz et al.
   Journal of Engineering Mechanics. 142.5 (2016) 04016016.1-10
2."Transitional Markov Chain Monte Carlo method for Bayesian model updating,
   model class selection and model averaging".
   Jianye Ching & Yi-Chun Chen
   Journal of Engineering Mechanics. 133.7 (2007) 816-832
---------------------------------------------------------------------------
"""

def iTMCMC(Ns, Nb, log_likelihood, T_nataf):  
    # initial check if there exists a Nataf object
    if isinstance(T_nataf, ERANataf):    # use Nataf transform (dependence)
        d   = len(T_nataf.Marginals)     # number of random variables
        u2x = lambda u: T_nataf.U2X(u)   # from u to x
        PDF = lambda t: T_nataf.pdf(t)

    elif isinstance(T_nataf[0], ERADist):   # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise or use an ERANataf object   
        d   = len(T_nataf)                # number of random variables
        u2x = lambda u: T_nataf[0].icdf(sp.stats.norm.cdf(u))   # from u to x
        PDF = lambda t: T_nataf[0].pdf(t)

    else:
        raise RuntimeError('Incorrect distribution. Please create an ERADist/Nataf object!')

    # some constants and initialization
    beta     = 2.4/np.sqrt(d)           # prescribed scaling factor (recommended choice)
    t_acr    = 0.21/d + 0.23            # target acceptance rate
    Na       = 100                      # number of chains to adapt
    thres_p  = 1                        # threshold for the c.o.v (100# recommended choice)
    max_it   = 50                       # max number of iterations (for allocation)
    logS_j   = np.ones(max_it)          # space for factors logS_j
    q        = np.zeros(max_it)         # store tempering parameters
    j        = 0                        # initialize counter for intermediate levels
    samplesU = list()
    samplesX = list()

    # 1. Obtain N samples from the prior pdf and evaluate likelihood
    u_j     = np.random.normal(0,1, size=(Ns,d))   # u_0 (Nxd matrix)
    theta_j = u2x(u_j)                             # theta_0 (transform the samples)
    samplesU.append(np.array(u_j))           # store initial level samples
    samplesX.append(np.array(theta_j))       # store initial level samples    
    #
    logL_j = np.array([log_likelihood(theta_j[i,:]) for i in range(Ns)])

    # iTMCMC
    while q[j] < 1 and j < max_it:   # adaptively choose q
        j += 1
        print('\niTMCMC intermediate level j = ', j-1,', with q_{j} = ', q[j-1])

        # 2. Compute tempering parameter p_{j+1}
        # e   = p_{j+1}-p_{j}
        # w_j = likelihood^(e), but we are using the log_likelihood, then:
        fun = lambda e: np.std(np.exp(np.abs(e)*logL_j)) - thres_p*np.mean(np.exp(np.abs(e)*logL_j))   # c.o.v equation
        e   = sp.optimize.fsolve(fun, 0)
        if e != np.nan:
            q[j] = np.minimum(1, q[j-1]+e)
        else:
            q[j] = 1
            print('Variable q was set to ', q[j], ', since it is not possible to find a suitable value')
    
        # 3. Compute log-'plausibility weights' w(theta_j) and factors 'logS_j' for the evidence
        logw_j    = (q[j]-q[j-1])*logL_j    # [Ref. 2 Eq. 12]
        logwsum = logsumexp(logw_j)
        logS_j[j-1] = logwsum-np.log(Ns)                    # [Ref. 2 Eq. 15]
        
        # 4. Metropolis resampling step to obtain N samples from f_{j+1}(theta)
        # weighted sample mean
        w_j_norm = np.exp(logw_j-logwsum)             # normalized weights
        mu_j     = np.matmul(u_j.T,w_j_norm)   # sum inside [Ref. 2 Eq. 17]
        
        # Compute scaled sample covariance matrix of f_{j}(theta)
        Sigma_j = np.zeros((d,d))
        for k in range(Ns):
            tk_mu   = u_j[k,:] - mu_j
            tmp     = w_j_norm[k]*(np.matmul(tk_mu.reshape(-1,1),tk_mu.reshape(1,-1)))
            Sigma_j = Sigma_j + tmp   # [Ref. 2 Eq. 17]
        
        # target pdf \propto prior(\theta)*likelihood(\theta)^p(j)
        level_post = lambda t, log_L: PDF(t)*np.exp(q[j]*log_L) # [Ref. 2 Eq. 11]
        
        # Start N different Markov chains
        print(' M-H sampling...')
        u_c      = u_j
        theta_c  = theta_j
        logL_c   = logL_j
        Nadapt   = 1
        na, acpt = 0, 0
        for k in range(Ns+Nb):      
            # select index l with probability w_j
            l = resampling_index(np.exp(logw_j))
            
            # sampling from the proposal and evaluate likelihood
            u_star     = sp.stats.multivariate_normal.rvs(mean=u_c[l,:],cov=(beta**2)*Sigma_j)
            theta_star = u2x(u_star)  # transform the sample
            logL_star  = log_likelihood(theta_star.flatten())
            
            # compute the Metropolis ratio
            ratio = level_post(theta_star, logL_star)/ level_post(theta_c[l,:], logL_c[l])
            
            # accept/reject step
            rv = sp.stats.uniform.rvs()
            if rv <= ratio:
                u_c[l,:]     = u_star
                theta_c[l,:] = theta_star
                logL_c[l]    = logL_star
                acpt        += 1
            
            if k > Nb:   # (Ref. 1 Modification 2: burn-in period)
                u_j[k-Nb,:]     = u_c[l,:]
                theta_j[k-Nb,:] = theta_c[l,:]
                logL_j[k-Nb]    = logL_c[l]
            
            # recompute the weights (Ref. 1 Modification 1: update sample weights)
            logw_j[l] = (q[j]-q[j-1])*logL_c[l]
            
            # adapt beta (Ref. 1 Modification 3: adapt beta)
            na += 1
            if na >= Na:
                p_acr    = acpt/Na
                ca       = (p_acr - t_acr)/np.sqrt(Nadapt)
                beta     = beta*np.exp(ca)
                Nadapt  += 1
                na, acpt = 0, 0  
        
        # store samples
        samplesU.append(np.array(u_j))
        samplesX.append(np.array(theta_j))
 
    print('\niTMCMC intermediate level j = ', j,', with q_{j} = ', q[j])
    
    # delete unnecessary data
    if j < max_it:
        q = q[:j+1]
        logS_j = logS_j[:j]
    
    # evidence (normalization constant in Bayes' theorem)
    logcE = np.sum(logS_j)   # [Ref. 2 Eq. 17]

    return samplesU, samplesX, q, logcE

#------------------------------------------------------------------------------
# nested function
    
def logsumexp(x):
    """Compute log(sum(exp(x))) while avoiding numerical underflow.
        By default dim = 1 (columns).
        Written by Michael Chen (sth4nth@gmail.com).
        Adapted for the use within iTMCMC."""
    
    # subtract the largest in each column
    y = np.max(x,0)
    x = x-y
    s = y + np.log(np.sum(np.exp(x),0));
    i = np.where(~np.isfinite(y));
    if i[0].size:
        s[i] = y[i]
    
    return s