'''
Import main libraries
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ERANataf import ERANataf
from ERADist import ERADist
from EMvMFNM import EMvMFNM
from EMvMFNM_log import EMvMFNM_log

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

def CEBU_vMFNM(N:int, log_likelihood, distr:ERANataf, max_it:int, CV_target:float, 
               k_init:int, samples_return:int, N_last:int=10000):
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

    if dim < 2:
        raise RuntimeError("Sorry, the vMFN-model can only be applied to d > 1!")
    
    # %% Set LSF in standard space
    loglike_fun = lambda u: log_likelihood(u2x(u))

    # Initialization of variables and storage

    N_tot:int  = 0 # total number of samples

    mu_init:np.ndarray = (1/np.sqrt(dim))*np.ones((k_init,dim))  # Initial mean sampled on unit hypersphere
    kappa_init:np.ndarray = np.zeros((k_init,1))  # Concentration parameter (zeroes for uniform distribution)
    omega_init:np.ndarray = np.ones((k_init,1))*(dim) # spread parameter
    m_init:np.ndarray = np.ones((k_init,1))*(dim/2) # shape parameter
    alpha_init:np.ndarray = np.ones((k_init,1))/k_init; # normalized weight of the mixtures
    beta_init:float = 0
                    
    samplesU:list  = []           # space for the samples in the standard space

    # %% initialize storage parameters

    mu_hat:np.ndarray = mu_init  # Initial mean sampled on unit hypersphere
    kappa_hat:np.ndarray = kappa_init  # Concentration parameter
    omega_hat:np.ndarray = omega_init  # spread parameter
    m_hat:np.ndarray = m_init # shape parameter
    alpha_hat:np.ndarray = alpha_init # normalized weight of the mixtures

    # Generate structure array to store the parameters
    v_tot:list = []

    v_tot.append({"mu_hat":mu_hat,"kappa_hat":kappa_hat,"omega_hat":omega_hat,"m_hat":m_hat,
                  "alpha_hat":alpha_hat})

    beta_hat = beta_init

    beta_tot = np.zeros(max_it,)
    cv_tot = np.zeros(max_it)

    # %% Iteration Section
    # Start the counter
    for j in range(max_it): 
        print('\n\nstep ',j,' for beta=',beta_hat)
        
        # Generate samples
        uk = vMFNM_sample(v_tot[j]['mu_hat'],v_tot[j]['kappa_hat'],v_tot[j]['omega_hat'],
                         v_tot[j]['m_hat'],v_tot[j]['alpha_hat'],N)
        samplesU.append(uk)

        #plt.scatter(U[:,0],U[:,1])


        # Add the number of samples generated
        N_tot = N_tot+N
        
        # evaluation of the limit state function
        logLk = loglike_fun(uk)

        # Replace the "NaN" values from the estimation of the limit state
        # function
        logLk[np.isnan(logLk)] = np.nanmin(logLk)  


        # Compute the prior pdf function
        logPriorpdf_ = jointpdf_prior(uk,dim)
        
        # Update beta (tempering parameter)
        beta_hat = betanew(beta_hat,uk,v_tot[j],logLk,logPriorpdf_,N,CV_target)
        
        #Handler (if beta is greater than 1, set to 1)
        if beta_hat > 1:
            beta_hat = 1

        #Save the new beta (tempering parameter)
        beta_tot[j+1] = beta_hat
        
        #likelihood ratio
        logWconst_ = log_W_const(uk,v_tot[j],logPriorpdf_)
        logW = logwl(beta_hat,np.transpose(logLk),logWconst_); 
        
        cv_tot[j] = np.std(np.exp(logW))/np.mean(np.exp(logW)) # only for diagnostics, hence not optimized for precision
        
        # update parameter
        v_new = vnew(uk,logW,k_init)
        v_tot.append(v_new)
        
        if beta_hat>=1:
            break
    
    
    #plt.show()
    # %% Adjust storing variables
    beta_tot = beta_tot[:j+2]
    cv_tot = cv_tot[:j+1]

    # Calculation of the Probability of failure
    #Compute weights
    W =  np.exp(logW)

    # results
    #Produce new failure samples after end step
    ulast = vMFNM_sample(v_tot[-1]['mu_hat'],v_tot[-1]['kappa_hat'],
                         v_tot[-1]['omega_hat'],v_tot[-1]['m_hat'],v_tot[-1]['alpha_hat'],
                         N_last)
    samplesU.append(ulast)


    N_tot = N_tot+ N_last

    logWlast = logwl2(ulast,v_tot[-1],loglike_fun,1)
    Wlast = np.exp(logWlast)
    Wlast_normed = np.exp(logWlast-logsumexp(logWlast))

    # Compute nESS
    nESS = np.exp(2*logsumexp(logWlast)-logsumexp(2*logWlast)-np.log(N_last))

    evidence = np.exp(logsumexp(logW)-np.log(N_last)).squeeze() # Evidence

    k_fin = len(v_tot[-1]["mu_hat"])

    xlast:np.ndarray = u2x(ulast)


    # Use default MATLAB Sampling generator
    weights_id = np.random.choice(np.arange(np.size(Wlast_normed)), N_last,True,Wlast_normed)
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


    #return v_tot,beta_tot,cv_tot,N_tot,xlast,Wlast_normed,evidence, randsamples,nESS
    return samplesU, samplesX, v_tot, beta_tot, k_fin, evidence, Wlast_normed, f_s_iid 


# ===========================================================================
# =============================AUX FUNCTIONS=================================
# ===========================================================================
# --------------------------------------------------------------------------
# Returns uniformly distributed samples from the surface of an
# n-dimensional hypersphere
# --------------------------------------------------------------------------
# N: # samples
# n: # dimensions
# R: radius of hypersphere
# --------------------------------------------------------------------------
def hs_sample(N:int, n, R):

    Y = sp.stats.norm.rvs(size=(n, N))  # randn(n,N)
    Y = Y.T
    norm = np.tile(np.sqrt(np.sum(Y ** 2, axis=1)), [1, n])
    X = Y / norm * R  # X = np.matmul(Y/norm,R)

    return X


# ===========================================================================
# --------------------------------------------------------------------------
# Returns samples from the von Mises-Fisher-Nakagami mixture
# --------------------------------------------------------------------------
def vMFNM_sample(mu:np.ndarray, kappa:np.ndarray, omega:np.ndarray, m:np.ndarray, alpha:np.ndarray, 
                 N:int):

    k = np.size(mu,0)
    dim = np.size(mu,1)

    if k == 1:
        # sampling the radius
        #     pd=makedist('Nakagami','mu',m,'omega',omega)
        #     R=pd.random(N,1)
        R = np.sqrt(sp.stats.gamma.rvs(a=m.ravel(), 
                                       scale=np.divide(omega.ravel(),m.ravel()), 
                                       size=[N, 1]))

        # sampling on unit hypersphere
        X_norm = vsamp(mu.T, kappa.ravel(), N)

    else:
        # Determine number of samples from each distribution
        z = np.sum(dummyvar(np.random.choice(np.arange(k), N, True, alpha.ravel())), axis=0)
        k = len(z)

        # Generation of samples
        R = np.zeros([N, 1])
        R_last = 0
        X_norm = np.zeros([N, dim])
        X_last = 0

        for p in range(k):
            # sampling the radius
            R[R_last : R_last + z[p]] = np.sqrt(
                sp.stats.gamma.rvs(
                    a=m.ravel()[p], scale=np.true_divide(omega.ravel()[p] ,m.ravel()[p]), size=[z[p], 1]
                )
            )
            R_last = R_last + z[p]

            # sampling on unit hypersphere
            X_norm[X_last : X_last + z[p], :] = vsamp(mu[p, :].T, kappa.ravel()[p], z[p])
            X_last = X_last + z[p]

            # clear pd

    # Assign sample vector
    X = R * X_norm  # bsxfun(@times,R,X_norm)

    return X


# ===========================================================================
# --------------------------------------------------------------------------
# Returns samples from the von Mises-Fisher distribution
# --------------------------------------------------------------------------
def vsamp(center, kappa, n):

    d = np.size(center, axis=0)  # Dimensionality
    l = kappa  # shorthand
    t1 = np.sqrt(4 * l * l + (d - 1) * (d - 1))
    b = (-2 * l + t1) / (d - 1)
    x0 = (1 - b) / (1 + b)
    X = np.zeros([n, d])
    m = (d - 1) / 2
    c = l * x0 + (d - 1) * np.log(1 - x0 * x0)

    for i in range(n):
        t = -1000
        u = 1
        while t < np.log(u):
            z = sp.stats.beta.rvs(m, m)  # z is a beta rand var
            u = sp.stats.uniform.rvs()  # u is unif rand var
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            t = l * w + (d - 1) * np.log(1 - x0 * w) - c

        v = hs_sample(1, d - 1, 1)
        X[i, : d - 1] = (
            np.sqrt(1 - w * w) * v
        )  # X[i,:d-1] = np.matmul(np.sqrt(1-w*w),v.T)
        X[i, d - 1] = w

    [v, b] = house(center)
    Q = np.eye(d) - b * np.matmul(v, v.T)
    for i in range(n):
        tmpv = np.matmul(Q, X[i, :].T)
        X[i, :] = tmpv.T

    return X


# ===========================================================================
# --------------------------------------------------------------------------
# X,mu,kappa
# Returns the von Mises-Fisher mixture log pdf on the unit hypersphere
# --------------------------------------------------------------------------
def vMF_logpdf(X, mu, kappa):

    d = np.size(X, axis=0)
    n = np.size(X, axis=1)

    if kappa == 0:
        A = np.log(d) + np.log(np.pi ** (d / 2)) - sp.special.gammaln(d / 2 + 1)
        y = -A * np.ones([1, n])
    elif kappa > 0:
        c = (
            (d / 2 - 1) * np.log(kappa)
            - (d / 2) * np.log(2 * np.pi)
            - logbesseli(d / 2 - 1, kappa)
        )
        q = np.matmul((mu * kappa).T, X)  # bsxfun(@times,mu,kappa)'*X
        y = q + c.T  # bsxfun(@plus,q,c')
    else:
        raise ValueError("Kappa must not be negative!")

    return y


# ===========================================================================
# --------------------------------------------------------------------------
# Returns the value of the log-nakagami-pdf
# --------------------------------------------------------------------------
def nakagami_logpdf(X, m, om):

    y = (
        np.log(2)
        + m * (np.log(m) - np.log(om) - X ** 2 / om)
        + np.log(X) * (2 * m - 1)
        - sp.special.gammaln(m)
    )

    return y

# ===========================================================================
# --------------------------------------------------------------------------
# HOUSE Returns the householder transf to reduce x to b*e_n
#
# [V,B] = HOUSE(X)  Returns vector v and multiplier b so that
# H = eye(n)-b*v*v' is the householder matrix that will transform
# Hx ==> [0 0 0 ... ||x||], where  is a constant.
# --------------------------------------------------------------------------
def house(x):

    x = x.squeeze()
    n = len(x)
    s = np.matmul(x[: n - 1].T, x[: n - 1])
    v = np.concatenate([x[: n - 1], np.array([1.0])]).squeeze()
    if s == 0:
        b = 0
    else:
        m = np.sqrt(x[n - 1] * x[n - 1] + s)

        if x[n - 1] <= 0:
            v[n - 1] = x[n - 1] - m
        else:
            v[n - 1] = -s / (x[n - 1] + m)

        b = 2 * v[n - 1] * v[n - 1] / (s + v[n - 1] * v[n - 1])
        v = v / v[n - 1]

    v = v.reshape(-1, 1)

    return [v, b]


# ===========================================================================
# --------------------------------------------------------------------------
# log of the Bessel function, extended for large nu and x
# approximation from Eqn 9.7.7 of Abramowitz and Stegun
# http://www.math.sfu.ca/~cbm/aands/page_378.htm
# --------------------------------------------------------------------------
def logbesseli(nu, x):

    if nu == 0:  # special case when nu=0
        logb = np.log(sp.special.iv(nu, x))  # besseli
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


# ===========================================================================
# --------------------------------------------------------------------------
# Computes the joint pdf of the prior distribution
# --------------------------------------------------------------------------
def jointpdf_prior(X:np.ndarray,dim):


    # Convert the physical space from "rectilinear" to "spherical"
    R = np.sqrt(np.sum(X * X, axis=1)).reshape(-1, 1)

    # numerical stability; 
    # should actually be equal to number closest to zero without cancellation 
    # error
    R[R==0]=1e-300 

    # uniform hypersphere
    A   = np.log(dim)+dim/2*np.log(np.pi)-sp.special.gammaln(dim/2+1)
    f_u = -A

    # chi distribution
    f_chi = (
        np.log(2) * (1 - dim / 2)
        + np.log(R) * (dim - 1)
        - 0.5 * R ** 2
        - sp.special.gammaln(dim / 2)
    )

    # logpdf of the standard distribution (uniform combined with chi distribution)
    f_log = f_u + f_chi

    return f_log 

# ===========================================================================
# --------------------------------------------------------------------------
# Computes the von Mises Fisher Nakagami Mixture log pdf
# --------------------------------------------------------------------------

def mixture_log_pdf(X, mu, kappa, omega, m, alpha):

    k = len(alpha.ravel())
    [N, dim] = np.shape(X)
    R = np.sqrt(np.sum(X * X, axis=1)).reshape(-1, 1)
    if k == 1:
        # log pdf of vMF distribution
        logpdf_vMF = vMF_logpdf((X / R).T, mu.T, kappa.ravel()).T
        # log pdf of Nakagami distribution
        logpdf_N = nakagami_logpdf(R, m.ravel(), omega.ravel())
        # log pdf of weighted combined distribution
        h_log = logpdf_vMF + logpdf_N
    else:
        logpdf_vMF = np.zeros([N, k])
        logpdf_N = np.zeros([N, k])
        h_log = np.zeros([N, k])

        # log pdf of distributions in the mixture
        for p in range(k):
            # log pdf of vMF distribution
            logpdf_vMF[:, p] = vMF_logpdf((X / R).T, mu[p, :].T, kappa.ravel()[p]).squeeze()
            # log pdf of Nakagami distribution
            logpdf_N[:, p] = nakagami_logpdf(R, m.ravel()[p], omega.ravel()[p]).squeeze()
            # log pdf of weighted combined distribution
            h_log[:, p] = logpdf_vMF[:, p] + logpdf_N[:, p] + np.log(alpha.ravel()[p])

        # mixture log pdf
        h_log = logsumexp(h_log, 1)

    return h_log

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
    y:np.ndarray = np.max(x, axis = dim)
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
        
        # Transform coordinates
        R = np.sqrt(np.sum(X * X, axis=1)).reshape(-1, 1)

        logsumwold = logsumexp(logWold,1)
        
        # mean direction mu
        signX = np.sign(X)
        logX = np.log(np.abs(X))
        
        [mu_unnormed,signmu] = signedlogsumexp(np.add(logX,logWold),1,signX)
        lognorm_mu = 0.5*logsumexp(2*mu_unnormed,0)
        mu = np.multiply(np.exp(np.subtract(mu_unnormed,lognorm_mu)),signmu)
        mu[np.isinf(mu)] = np.sqrt(dim)
        
        v0 = mu
            
        #concentration parameter kappa
        xi = np.minimum(np.exp(lognorm_mu-logsumwold),0.999999999)
        kappa = np.abs((xi*dim-xi**3)/(1-xi**2)); 
            
        v1 = kappa

        # spread parameter omega
        # for numerical stability; 
        # should actually be equal to the number closest to zero without 
        # cancellation error by machine
        R[R==0]=1e-300; 
        
        logR = np.log(R)
        logRsquare = 2*logR
        omega = np.exp(logsumexp(logWold+logRsquare,1) - logsumwold)
        
        v2 = omega
        
        # shape parameter m
        logRpower = 4*logR
        mu4       = np.exp(logsumexp(logWold+logRpower,1) - logsumwold)
        m         = np.divide(np.power(omega,2),mu4-np.power(omega,2))
        
        m[m<0.5]  = dim/2
        m[np.isinf(m)] = dim/2
        
        v3 = m
        
        # distribution weights alpha
        alpha = np.array([1])
        
        v4 = alpha
        
        newv_var = (v0.T,v1,v2,v3,v4)
    else:
        print('\nusing EMvMFNM_log ')
        # for sake of consistency I let m and omega computed in the same order 
        # as within the EMvMFNM-script -> therefore the transposed digits
        [v0,v1,v3,v2,v4] = EMvMFNM_log(X,logWold,k); 

        # Normalize the weights
        #normW = np.exp(logWold-logsumexp(logWold))
        #[v0,v1,v3,v2,v4] = EMvMFNM(X,np.exp(logWold),k); 

        #Flips
        #v0_1=fliplr(v0_1);v1_1=fliplr(v1_1);v3_1=fliplr(v3_1);
        #v2_1=fliplr(v2_1);v4_1=fliplr(v4_1);
        newv_var = (v0.T,v1,v2,v3,v4)
        #newv_var = {v0_1.T,v1_1,v2_1,v3_1,v4_1}


    # Restructure the output as a dynamic memory variable (STRUCTURE)
    newv = {"mu_hat":newv_var[0],"kappa_hat":newv_var[1],"omega_hat":newv_var[2],
                "m_hat":newv_var[3],"alpha_hat":newv_var[4]}


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
                                            v['kappa_hat'], v['omega_hat'],v['m_hat'],v['alpha_hat']))

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
def logwl(beta:float,logL:np.ndarray,logWconst:np.ndarray):
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

