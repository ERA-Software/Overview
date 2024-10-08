'''
Import main libraries
'''

import numpy as np
import scipy as sp
from ERANataf import ERANataf
from ERADist import ERADist
from EMGM import EMGM
from EMGM_log import EMGM_log

"""
Comments:
---------------------------------------------------------------------------
Bayesian Updating and Marginal Likelihood Estimation by Cross Entropy based 
Importance Sampling
---------------------------------------------------------------------------

Created by:
Michael Engel
Oindrila Kanjilal 
Iason Papaioannou
Daniel Straub

Assistant Developers:
Daniel Koutas
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group
Technische Universitaet Muenchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2023-10
* 
---------------------------------------------------------------------------
Comments:
* Adopted draft scripts from Michael Engel. Reconstruction of the code made 
  to comply with the style of the published codes

---------------------------------------------------------------------------
Input:
* N                    : number of samples per level
* log_likelihood       : log-likelihood function
* max_it               : maximum number of iterations
* distr                : Nataf distribution object or marginal distribution object of the input variables
* CV_target            : taeget correlation of variation of weights
* k_init               : initial number of Gaussians in the mixture model
* samples_return       : return of samples: 0 - none, 1 - final sample, 2 - all samples

Optional:
* N_last                : set a number of independent identically 
                         distributed of samples to generate at the end
---------------------------------------------------------------------------
Output:
* samplesU      : list of the samples in the standard normal space
* samplesX      : list of the samples in the original space
* v_tot         : list of distribution parameters
* beta_tot      : array of tempering parameters
* k_fin         : final number of Gaussians in the mixture
* evidence      : evidence
* W_last_normed : normed weights of the last level samples
* f_s_iid       : iid samples
---------------------------------------------------------------------------
Based on:
1. Engel, M., Kanjilal, O., Papaioannou, I., Straub, D. (2022)
   Bayesian Updating and Marginal Likelihood Estimation by 
   Cross Entropy based Importance Sampling.(Preprint submitted to Journal 
   of Computational Physics) 
---------------------------------------------------------------------------
"""

def CEBU_GM(N:int, log_likelihood, distr:ERANataf, max_it:int, CV_target:float, 
            k_init:int, samples_return:int=1, N_last:int=10000):
    # %% Input Check
    if not isinstance(N,int) and not N>0:
        raise RuntimeError(
            "N must be a positive integer"
        )
    
    if not isinstance(k_init,int) and not k_init>0:
        raise RuntimeError(
            "k_init must be a positive integer"
        )

    # initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):  # use Nataf transform (dependence)
        dim = len(distr.Marginals)  # number of random variables (dimension)
        u2x = lambda u: distr.U2X(u)  # from u to x

    elif isinstance(
        distr[0], ERADist
    ):  # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise
        dim = len(distr)  # number of random variables (dimension)
        u2x = lambda u: distr[0].icdf(sp.stats.norm.cdf(u))  # from u to x
    else:
        raise RuntimeError(
            "Incorrect distribution. Please create an ERADist/Nataf object!"
        )

    #if dim < 2:
    #    raise RuntimeError("Sorry, the GM-model can only be applied to d > 1!")
    
    # %% Set LSF in standard space
    loglike_fun = lambda u: log_likelihood(u2x(u))

    # Initialization of variables and storage

    N_tot:int  = 0 # total number of samples

    mu_init:np.ndarray = np.zeros((k_init,dim))  # Initial mean sampled on unit hypersphere
    
    Si_init = np.zeros((dim,dim,k_init))  # covariance
    for i in range(k_init):
        Si_init[:,:,i] = Si_init[:,:,i]+ 1*np.eye(dim)

    Pi_init = np.ones((k_init,1))/k_init
    
    
    # Initial Beta
    beta_init:float = 0
                    
    samplesU:list = []           # space for the samples in the standard space
    # %% initialize storage parameters

    mu_hat:np.ndarray = mu_init  # Initial mean sampled on unit hypersphere
    Si_hat:np.ndarray = Si_init  # Concentration parameter
    Pi_hat:np.ndarray = Pi_init

    # Generate structure array to store the parameters
    v_tot:list = []

    v_tot.append({"mu_hat":mu_hat,"Si_hat":Si_hat,"Pi_hat":Pi_hat})

    beta_hat = beta_init

    beta_tot = np.zeros(max_it)
    cv_tot = np.zeros(max_it)

    # %% Iteration Section
    # Start the counter
    for j in range(max_it): 
        print('\n\nstep ',j,' for beta=',beta_hat)
        
        # Generate samples
        uk:np.ndarray = GM_sample(v_tot[j]['mu_hat'], v_tot[j]['Si_hat'],
                                  v_tot[j]['Pi_hat'], N)
        samplesU.append(uk)

        # Add the number of samples generated
        N_tot = N_tot+N
        
        # evaluation of the limit state function
        logLk = loglike_fun(uk)

        # Replace the "NaN" values from the estimation of the limit state
        # function
        logLk[np.isnan(logLk)] = np.nanmin(logLk.ravel())  

        # Compute the prior pdf function
        logPriorpdf_ = jointpdf_prior(uk, dim)
        
        # Update beta (tempering parameter)
        beta_hat = betanew(beta_hat, uk, v_tot[j], logLk, logPriorpdf_, N, CV_target)
        
        #Handler (if beta is greater than 1, set to 1)
        if beta_hat > 1:
            beta_hat = 1

        #Save the new beta (tempering parameter)
        beta_tot[j+1] = beta_hat

        #likelihood ratio
        logWconst_ = log_W_const(uk, v_tot[j], logPriorpdf_)
        logW = logwl(beta_hat, np.transpose(logLk), logWconst_); 
        
        cv_tot[j] = np.std(np.exp(logW))/np.mean(np.exp(logW)) # only for diagnostics, hence not optimized for precision
        
        # update parameter
        v_new = vnew(uk, logW, k_init)
        v_tot.append(v_new)
        
        if beta_hat>=1:
            break
    
    # %% Adjust storing variables
    beta_tot = beta_tot[:j+2]
    cv_tot = cv_tot[:j+1]

    # Calculation of the Probability of failure
    #Compute weights
    W = np.exp(logW)

    #Produce new failure samples after end step
    ulast = GM_sample(v_tot[-1]['mu_hat'], v_tot[-1]['Si_hat'], v_tot[-1]['Pi_hat'], N_last)
    samplesU.append(ulast)

    N_tot = N_tot + N_last

    logWlast = logwl2(ulast,v_tot[-1],loglike_fun,1)
    Wlast_normed = np.exp(logWlast-logsumexp(logWlast,0))

    # Compute nESS
    nESS = np.exp(2*logsumexp(logWlast)-logsumexp(2*logWlast)-np.log(N_last))

    evidence = np.exp(logsumexp(logW)-np.log(N_last)).squeeze() # Evidence

    xlast:np.ndarray = u2x(ulast)

    k_fin = len(v_tot[-1]["mu_hat"])

    # Use default numpy sampling generator
    weights_id = np.random.choice(np.arange(np.size(Wlast_normed)), N_last, True, Wlast_normed)
    f_s_iid = xlast[weights_id,:]
    print('\nfinished after ', j, ' steps at beta=', beta_hat, '\n')

    # transform the samples to the physical/original space
    samplesX = list()
    if samples_return != 0:
        for i in range(len(samplesU)):
            samplesX.append(u2x(samplesU[i][:,:]))
    
    if samples_return == 1:
        samplesU = [samplesU[-1]]
        samplesX = [samplesX[-1]]

    if samples_return == 0:
        samplesU = list()  # empty return samples U
        samplesX = list()  # and X

    # Convergence is not achieved message
    if j == max_it:
        print("\n-Exit with no convergence at max iterations \n")

    return samplesU, samplesX, v_tot, beta_tot, k_fin, evidence, Wlast_normed, f_s_iid 


# ===========================================================================
# =============================AUX FUNCTIONS=================================
# ===========================================================================
def GM_sample(mu:np.ndarray, si:np.ndarray, pi:np.ndarray, N:int):
    # ---------------------------------------------------------------------------
    # Algorithm to draw samples from a Gaussian-Mixture (GM) distribution
    # ---------------------------------------------------------------------------
    # Input:
    # * mu : [npi x d]-array of means of Gaussians in the Mixture
    # * Si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
    # * Pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1)
    # * N  : number of samples to draw from the GM distribution
    # ---------------------------------------------------------------------------
    # Output:
    # * X  : samples from the GM distribution
    # ---------------------------------------------------------------------------
    if np.size(mu, axis=0) == 1:
        mu = mu.squeeze()
        si = si.squeeze()
        X = sp.stats.multivariate_normal.rvs(mean=mu, cov=si, size=N)
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)
    else:
        # Determine number of samples from each distribution
        z:np.ndarray = np.round(pi * N)
        if np.sum(z) != N:
            dif = np.sum(z) - N
            ind = np.argmax(z)
            z[ind] = z[ind] - dif

        z = z.astype(int)  # integer conversion

        # Generate samples
        d = np.size(mu, axis=1)
        X = np.zeros([N, d])
        ind = 0
        for p in range(len(pi)):
            X[ind : ind + z.ravel()[p], :] = sp.stats.multivariate_normal.rvs(
                mean=mu[p, :], cov=si[:, :, p].squeeze(), size=z.ravel()[p]
            ).reshape(-1,d)
            ind = int(ind + z.ravel()[p])

    return X


# ===========================================================================
# --------------------------------------------------------------------------
# Returns the log of the gaussian pdf
# --------------------------------------------------------------------------
def loggausspdf(X:np.ndarray, mu:np.ndarray, Sigma:np.ndarray):

    d = np.size(X, axis=0)
    X = X-mu.reshape(-1,1)
    U = np.linalg.cholesky(Sigma).T.conj()
    Q = np.linalg.solve(U.T, X)
    q = np.sum(Q*Q, axis=0)      # quadratic term (M distance)
    c = d*np.log(2*np.pi)+2*np.sum(np.log(np.diag(U)))   # normalization constant
    y = -(c+q)/2
    
    return y


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


# ===========================================================================
# --------------------------------------------------------------------------
# Computes the joint pdf of the prior distribution
# --------------------------------------------------------------------------
def jointpdf_prior(X:np.ndarray,dim):

    pdf = loggausspdf(X.T,np.zeros((dim,1)),np.eye(dim))

    return pdf

# ===========================================================================
# --------------------------------------------------------------------------
# Computes the Gaussian Mixture log pdf
# --------------------------------------------------------------------------

def mixture_log_pdf(X:np.ndarray, mu:np.ndarray, si:np.ndarray, ki:np.ndarray):
    # Input:
    #   X   samples             [dim x Nsamples]
    #   mu  mean                [Nmodes x dim]
    #   si  covariance          [dim x dim x Nmodes
    #   ki  weights of modes    [1 x Nmodes]
    # Output:
    #   h   logpdf of samples   [1 x Nsamples]

    # Invert the number of samples
    [dim,N] = np.shape(X)

    if dim > N:
        X = X.T # transpose the sample array

    N = np.size(X,1); # number of samples
    k_tmp = np.size(ki); # number of modes
    if k_tmp == 1:
        si_squeezed = np.squeeze(si)
        if si_squeezed.ndim == 0:
            si_squeezed = np.eye(1)*si_squeezed
        h = loggausspdf(X,mu.T,si_squeezed)
    else:
        h_pre = np.zeros((k_tmp,N))
        for q in range(k_tmp):
            mu_ = mu[q,:]
            si_ = si[:,:,q]
            si_squeezed = np.squeeze(si_)
            if si_squeezed.ndim == 0:
                si_squeezed = np.eye(1)*si_squeezed
            h_pre[q,:] = np.log(ki.ravel()[q]) + loggausspdf(X,mu_.T,si_squeezed)

        h = logsumexp(h_pre,0)


    return h

#=========================================================================
# ------------------------------------------------------------------------
# Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
#By default dim = 1 (columns).
#Written by Michael Chen (sth4nth@gmail.com).
#   
#Adapted by Michael Engel such that log(sum(exp(x)*b,dim)) and
# negative results are supported.
# Only recommended for b working as a sign-vector of exp(x).
# ------------------------------------------------------------------------
def signedlogsumexp(x, dim=0, b=[]):
   
    if np.size(b)==0:
        b = np.ones((1,np.shape(x)[dim]))

    # subtract the largest in each column (rescaling x to (0,1] 
    # where log offers better precision)
    y:np.ndarray = np.argmax(x,axis = dim)
    y = np.expand_dims(y,axis=dim)
    x:np.ndarray = np.transpose(np.subtract(x,y))
    term:np.ndarray = np.ravel(np.sum(np.multiply(np.exp(x),b),axis=dim))
    signn = np.ones(term.shape)
    signn[term<0] = -1

    # return nonfinite value if existing
    s = y + np.log(np.abs(term))
    i = np.where(np.invert(np.isfinite(y).squeeze()))
    if np.size(i)!=0:
       s[i] = y[i]

    return s,signn



# ===========================================================================
# --------------------------------------------------------------------------
# Updates the tempering parameter (beta)
# --------------------------------------------------------------------------
def betanew(betaold:float,samples_U:np.ndarray,v:list,logLikelihood_,logpriorpdf_:np.ndarray,
            Nstep:int,cvtarget:float):
      
    # Compute the logarithmic weights
    logWconst_ = log_W_const(samples_U,v,logpriorpdf_); 

    # Necessary as new set of v given as computation after parameter update
    logstep_ = logstep(Nstep,cvtarget); # could be computed beforehand (once) but computational cost is negligible

    #variant a: optimize beta with respect to the ISdensity -> depending on parameters
    #logESSoptimize_ = @(dbeta) logESSoptimize(betaold+abs(dbeta),logLikelihood_,logWconst_,logstep_);
    #variant b: optimize beta with respect to the likelihood ratio assuming a 
    #perfect fit of the previous ISdensity -> not depending on the parameters
    logESSoptimize_ = lambda dbeta: logESSoptimize2(np.abs(dbeta),logLikelihood_,logstep_)

    opts:dict = {'maxiter':1000}

    res:sp.optimize.OptimizeResult = sp.optimize.minimize_scalar(logESSoptimize_,bounds=[-1,1e-6],
                                                                 method='bounded',options=opts)
    dbeta = np.ravel(res.x)                                    
    dbeta = np.abs(dbeta)
    newbeta = betaold+dbeta

    # for convergence
    if newbeta >=1-1e-3:
        newbeta=1


    return newbeta

# ===========================================================================
# --------------------------------------------------------------------------
# Updates the posterior distribution parameters (based on VFN Mixture)
# --------------------------------------------------------------------------
def vnew(X:np.ndarray,logWold,k):

    # RESHAPE STEP (avoid confusion on row or column vectors)
    [N,dim] = np.shape(X)
    if N > dim:
        X = X.T
        dim = np.shape(X)[0]
    #END RESHAPE STEP (avoid confusion on row or column vectors)
    if k == 1:
        print('\n...using analytical formulas')
        sumwold = np.exp(logsumexp(logWold,1))

        dim = np.size(X,0)
    
        signu = np.sign(X)
        logu = np.log(np.abs(X))
        
        [mu_unnormed,signmu] = signedlogsumexp(logu+logWold,1,signu)
        mu = np.dot(np.exp(mu_unnormed-logsumexp(logWold,1)),signmu)
       
        
        v0 = mu.T
        v1 = np.zeros((dim,dim,k))
        sqrtR = np.exp(0.5*logWold)
        Xo = np.subtract(X,np.transpose(v0[0,:]))
        Xo = np.multiply(Xo,sqrtR)

        v1[:,:,0] = np.divide(np.matmul(Xo,np.transpose(Xo)),sumwold)
        v1[:,:,0] = v1[:,:,0] + np.eye(dim)*(1e-06)


        v2 = np.array([1])
       
        newv_var = (v0.T,v1,v2)
    else:
        if np.all(X==0):
            print('issue with samples')

        print('\nusing EMGM_log ')
        # for sake of consistency I let m and omega computed in the same order 
        # as within the EMvMFNM-script -> therefore the transposed digits
        #[v0,v1,v2] = EMGM_log(X,logWold,k)
        [v0, v1, v2] = EMGM(X, np.exp(logWold), k)
        newv_var = (v0.T,v1,v2)
        #newv_var = {v0_1.T,v1_1,v2_1,v3_1,v4_1}


    # Restructure the output as a dynamic memory variable (STRUCTURE)
    newv = {"mu_hat":newv_var[0],"Si_hat":newv_var[1],"Pi_hat":newv_var[2]}


    return newv

# =========================================================================
# --------------------------------------------------------------------------
# Optimization function to compute the new tempering parameter
# --------------------------------------------------------------------------

def logESSoptimize2(dbeta:float,logLikelihood:np.ndarray,logstep):
    logw = np.multiply(dbeta,logLikelihood)
    logdiff = np.power(np.subtract(2*logsumexp(logw)-logsumexp(2*logw),logstep),2)
    return np.ravel(logdiff)

# =========================================================================
# --------------------------------------------------------------------------
#
# --------------------------------------------------------------------------
def log_W_const(samples_U,v,logpriorpdf):
    logconstw = np.subtract(logpriorpdf,mixture_log_pdf(samples_U,v['mu_hat'],
                                            v['Si_hat'], v['Pi_hat']))

    return np.ravel(logconstw)

# =========================================================================
# --------------------------------------------------------------------------
#
# --------------------------------------------------------------------------
def logstep(Nstep:int,cvtarget:float):
    logS = np.log(Nstep)-np.log(1+cvtarget**2)
    return logS


# =========================================================================
# --------------------------------------------------------------------------
# Compute the log weights
# --------------------------------------------------------------------------
def logwl(beta,logL,logWconst):
    # Compute the logarithm of the transition weights (W_t)
    logW = np.add(np.multiply(beta,logL),logWconst)

    return logW

# =========================================================================
# --------------------------------------------------------------------------
# Compute the log weights (for the end)
#  =========================================================================
def logwl2(X:np.ndarray,v:dict,loglike_fun_U,beta:float):

    # Perform evaluation of the function
    logL_ = loglike_fun_U(X)

    # Correct the evaluation
    logL_[np.isnan(logL_)] = np.nanmin(logL_) 

    print('\nmaximum of final logLikelihood: ',np.max(logL_))
    print('\nminimum of final logLikelihood: ',np.min(logL_))


    [_,dim] = np.shape(X)
    logP_ = jointpdf_prior(X,dim)
    logwconst_ = log_W_const(X,v,logP_)

    logw = np.add(np.multiply(beta,logL_.ravel()),logwconst_.ravel())

    return logw



