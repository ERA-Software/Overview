from argon2 import Type
import numpy as np
import scipy as sp
np.seterr(all='ignore')
"""
---------------------------------------------------------------------------
Perform soft EM algorithm for fitting the Gaussian mixture model
---------------------------------------------------------------------------
Created by:
Michael Engel (m.engel@tum.de)

Based upon the code of:
Sebastian Geyer (s.geyer@tum.de)
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2023-10
---------------------------------------------------------------------------
Input:
* X   : data matrix (dimensions x Number of samples)
* logW : vector of log-likelihood-ratios for weighted samples
* nGM : number of Gaussians in the Mixture
---------------------------------------------------------------------------
Output:
* mu : [npi x d]-array of means of Gaussians in the Mixture
* si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
* pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1) 
---------------------------------------------------------------------------
Based on:
1. "EM Demystified: An Expectation-Maximization Tutorial"
   Yihua Chen and Maya R. Gupta
   University of Washington, Dep. of EE (Feb. 2010)
---------------------------------------------------------------------------
"""
def EMGM_log(X:np.ndarray, logW:np.ndarray, nGM:int,
             tol:float=1e-5,maxiter:int=500):
    # reshaping just to be sure
    logW = logW.reshape(1,-1)

    # initialization
    logR = initialization(X, nGM)

    llh       = np.full([maxiter],-np.inf)
    converged = False
    t         = 0

    # soft EM algorithm
    while (not converged) and (t+1 < maxiter):
        t = t+1   

        label = np.argmax(logR, axis=1)
        u = np.unique(label)    # non-empty components
        if np.size(logR, axis=1) != np.size(u):
            logR = logR[:,u]          # remove empty components
           
        
        [mu, si, pi] = maximization(X,logW,logR)
        [logR, llh[t]]  = expectation(X, logW, mu, si, pi)

        if t > 1:
            converged = abs(llh[t]-llh[t-1]) < tol*np.abs(llh[t])

    if converged:
        print('Converged in', t,'steps.')
    else:
        print('Not converged in ', maxiter, ' steps.')

    return mu, si, pi

# --------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------
def initialization(X:np.ndarray, nGM):
    
    # Random initialization
    n:int = np.shape(X)[1]

    # Use the K-Means algorithm from Scipy
    codebook, distortion = sp.cluster.vq.kmeans(X.T,k_or_guess=nGM,thresh=1e-16)
    label,dist = sp.cluster.vq.vq(X.T,codebook)
    # build M
    logR:np.ndarray = np.ones((n,nGM))*(-1e200)
    for i in range(n):
        logR[i,label[i]]=0



    return logR



# --------------------------------------------------------------------------
# Expectation
# --------------------------------------------------------------------------
def expectation(X, logW, mu, si, pi):

    n      = np.size(X,1)
    k      = np.size(mu,1)
    logpdf = np.zeros((n,k))

    for i in range(k):
        logpdf[:,i] = np.ravel(loggausspdf(X,np.squeeze(mu[:,i]),np.squeeze(si[:,:,i])))
    
    logpdf = logpdf + np.log(pi)
    T      = logsumexp(logpdf,1)
    llh    = sum(np.transpose(logW)+T)-logsumexp(logW,1)

    logR = logpdf-T
    logR = logR-logsumexp(logR,1)

    if np.any(np.isnan(logR)):
        print('\nlogR is Nan')


    return logR, llh


# --------------------------------------------------------------------------
# Maximization
# --------------------------------------------------------------------------
def maximization(X, logW, logR):
    logRW     = np.transpose(logW)+logR
    d = np.size(X,0)
    k = np.size(logRW,1)
    nk = np.exp(logsumexp(logRW,0))
    lognk = logsumexp(logRW,0)
    if np.any(nk == 0): # numerical stability
        nk = nk + 1e-6
    
    sumW = np.exp(logsumexp(logW,1))
    w = np.true_divide(nk,sumW)
    # w = np.exp(logsumexp(logRW,0)-logsumexp(logW,1));

    signX = np.sign(X)
    logX = np.log(abs(X))

    mu = np.zeros((d,k))
    for i in range(k):
        [mu_unnormed,signmu] = signedlogsumexp(logX+np.transpose(logRW[:,i]),1,signX)
        mu[:,i] = np.multiply(np.exp(mu_unnormed-logsumexp(np.transpose(logRW[:,i]))),signmu).ravel()
    

    Sigma = np.zeros((d,d,k))
    sqrtRW = np.exp(0.5*logRW)
    # linear-version as log-version slow
    for i in range(k):
        Xo = X-mu[:,i].reshape((-1,1))
        Xo = np.multiply(Xo,np.transpose(sqrtRW[:,i]))
   
        Sigma[:,:,i] = np.true_divide((Xo @ np.transpose(Xo)),nk.ravel()[i])
        Sigma[:,:,i] = Sigma[:,:,i]+np.eye(d)*(1e-5)  # add a prior for numerical stability
    
        
    if np.any(np.isnan(Sigma)):
        print('\nSigma is Nan')


    if np.any(np.isnan(Sigma)):
        print('\nAlpha is Inf')
    



    return mu, Sigma, w

# --------------------------------------------------------------------------
# Returns the log of the gaussian pdf
# --------------------------------------------------------------------------
def loggausspdf(X, mu, Sigma):
    d = np.size(X, axis=0)
    X = X-mu.reshape(-1,1)
    U = np.linalg.cholesky(Sigma).T.conj()
    Q = np.linalg.solve(U.T, X)
    q = np.sum(Q*Q, axis=0)      # quadratic term (M distance)
    c = d*np.log(2*np.pi)+2*np.sum(np.log(np.diag(U)))   # normalization constant
    y = -(c+q)/2
    return y

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

# --------------------------------------------------------------------------
# Translation of the Matlab-function "dummyvar()" to Python
# --------------------------------------------------------------------------
def dummyvar(idx):
    n = np.max(idx)+1
    d = np.zeros([len(idx),n],int)
    for i in range(len(idx)):
        d[i,idx[i]] = 1
    return d


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
    y:np.ndarray = np.max(x,axis = dim)
    y = np.expand_dims(y,axis=dim)
    x:np.ndarray = np.subtract(x,y) # bsxfun(@minus,x,y)
    term:np.ndarray = np.sum(np.multiply(np.exp(x),b),axis=dim)
    term = np.expand_dims(term,axis=dim)
    signn = np.ones(term.shape)
    signn[term.ravel()<0] = -1

    # return nonfinite value if existing
    s = y + np.log(np.abs(term))
    i = np.where(np.invert(np.isfinite(y).squeeze()))
    
    if np.size(i)!=0:
       s[i] = y[i]

    return s,signn