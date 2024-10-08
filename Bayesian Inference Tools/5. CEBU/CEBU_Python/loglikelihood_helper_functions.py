# Import main libraries
import numpy as np
import scipy as sp


"""
This script is meant to include helper functions to compute the loglikelihood functions

Created by:
Ivan Olarte-Rodriguez

Assistant Developers:
Daniel Koutas

Based on the work by:
Michael Engel


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
"""



def loglikelihood(u:np.ndarray,sigma:float,dim:int,fraction:float,rho:float):
    
    if dim < 2:
        print('\ndim has to be larger than 2!')

    if np.size(u,0) > dim:
        u = np.transpose(u)

    N = np.size(u,1)
    mu1 = 0.5*np.ones((dim,1))
    si1 = np.eye(dim)
    si1[0,1] = rho
    si1[1,0] = rho
    cov1 = np.multiply(sigma**2,si1)
    print(cov1)
    fraction1 = fraction
    mu2 = -0.5*np.ones((dim,1))
    si2 = np.eye(dim)
    si2[0,1] = -rho
    si2[1,0] = -rho
    cov2 = np.multiply(sigma**2,si2)
    print(cov2)
    fraction2 = 1-fraction
    h_pre = np.zeros((N,2))
    h_pre[:,0] = np.add(np.log(fraction1), loggausspdf(u,mu1,cov1) )
    h_pre[:,1] = np.add( np.log(fraction2), loggausspdf(u,mu2,cov2) ) 
    h = logsumexp(h_pre,1)

    return h


# ===========================================================================
# --------------------------------------------------------------------------
# Computes ME Fraction given set of samples, sigma and fraction
# --------------------------------------------------------------------------
def MEfraction(u,sigma,fraction):
    
    N = np.size(u, axis = 1)
    dim = np.size(u, axis = 0)
    mu1 = 0.5*np.ones((dim,1))
    cov1 = sigma**2*np.eye(dim)
    fraction1 = fraction
    mu2 = -0.5*np.ones((dim,1))
    cov2 = cov1
    fraction2 = 1-fraction
    
    counter1 = 0
    index = np.zeros((1,N),dtype=bool)
    for i in range(N):
        gauss1 = np.log(fraction1) + loggausspdf(u[:,i],mu1,cov1)
        gauss2 = np.log(fraction2) + loggausspdf(u[:,i],mu2,cov2)
        if gauss1 > gauss2:
            counter1 = counter1 + 1
            index[i] = True
    
    fraction1 = np.divide(counter1,N)
    if np.any(np.isnan(u)):
        fraction1 = np.nan
        index = np.nan
    
    return [fraction1,index]


# ===========================================================================
# --------------------------------------------------------------------------
# Computes ME Mean
# --------------------------------------------------------------------------

def MEmean(samples,d,dim,sigma,fraction):
    [_,index] = MEfraction(samples,sigma,fraction)
    if d==1:
        mean = np.nanmean(samples[dim,index],1)
    elif d==2:
        mean = np.nanmean(samples[dim,np.logical_not(index)],1)


    return mean


# ===========================================================================
# --------------------------------------------------------------------------
# Computes ME Mean-max
# --------------------------------------------------------------------------
def MEmeanmax(u):
    N:int = np.size(u,1)
    maxvec = np.argmax(u,axis=0)
    mm = (1/N)*np.sum(maxvec)
    return mm


# ===========================================================================
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

# ===========================================================================
# --------------------------------------------------------------------------
# Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
#   By default dim = 0 (columns).
# Written by Michael Chen (sth4nth@gmail.com).
# --------------------------------------------------------------------------
def logsumexp(x, dim=0):

    # subtract the largest in each column
    y = np.max(x, axis=dim).reshape(-1, 1)
    x = x - y
    s = y + np.log(np.sum(np.exp(x), axis=dim)).reshape(-1, 1)
    # ===========================================================================
    i = np.where(np.invert(np.isfinite(y).squeeze()))
    s[i] = y[i]

    return s