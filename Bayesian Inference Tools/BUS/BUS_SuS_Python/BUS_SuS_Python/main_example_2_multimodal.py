"""
---------------------------------------------------------------------------
Example 2: multi-modal posterior
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu
Felipe Uribe
Luca Sardi

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Last Version 2021-03
* Adaptation to new ERANataf class
---------------------------------------------------------------------------
References:
1."Asymptotically independent Markov sampling: a new MCMC scheme for Bayesian inference"
   James L. Beck and Konstantin M. Zuev
   International Journal for Uncertainty Quantification, 3.5 (2013) 445-474.
2."Bayesian inference with subset simulation: strategies and improvements"
   Wolfgang Betz et al.
   Computer Methods in Applied Mechanics and Engineering 331 (2018) 72-93.
---------------------------------------------------------------------------
"""

#=================================================================
# import libraries
import numpy as np
import numpy.matlib
import scipy as sp
from scipy.stats import multivariate_normal as mvn
import matplotlib.pylab as plt

# Make sure ERADist, ERANataf classes are in the path
# https://www.bgu.tum.de/era/software/eradist/
from ERADist import ERADist
from ERANataf import ERANataf
from BUS_SuS import BUS_SuS
plt.close('all')
np.random.seed(123)   # fix seed

#=================================================================
# define the prior (bivariate uniform)
d     = 2      # number of dimensions (uncertain parameters)
a     = 10
bound = np.matlib.repmat([0, a],d,1)   # boundaries of the uniform box

# define the prior
prior_pdf_1 = ERADist('uniform','PAR',[bound[0,0],bound[0,1]])
prior_pdf_2 = ERADist('uniform','PAR',[bound[1,0],bound[1,1]])
prior_pdf   = [prior_pdf_1, prior_pdf_2]

# correlation matrix
R = np.eye(d)   # independent case

# object with distribution information
prior_pdf = ERANataf(prior_pdf,R)

#=================================================================
# define likelihood
# data set
m     = 10                          # number of data points
mu    = prior_pdf.random(m)       # Here using ERANataf.random
Sigma = (0.1**2)*np.eye(d)
w     = 0.1*np.ones(m)

# define likelihood
realmin        = np.finfo(np.double).tiny  # to prevent log(0)
likelihood_opt = lambda theta: sum(w[i] * mvn.pdf(theta[i],mu[i], Sigma) for i in range(0,m))   # Modified likelihood function for sum
likelihood     = lambda theta: sum(w[i] * mvn.pdf(theta, mu[i], Sigma) for i in range(0,m))
log_likelihood = lambda theta: np.log(likelihood(theta.T) + realmin)

#=================================================================
# find scale constant c
method = 1
# use MLE to find c
if method == 1: 
    f_start = np.log(mu + realmin)
    fun     = lambda lnF: -np.log( likelihood_opt(np.exp(f_start)) + realmin)
    MLE_ln  = sp.optimize.fmin(func=fun, x0=f_start)
    MLE_ln  = np.reshape(MLE_ln,[10,2])
    MLE     = np.exp(MLE_ln)          # likelihood(MLE) = 1
    c       = 1/likelihood_opt(MLE)
        
# some likelihood evaluations to find c
elif method == 2:
    K  = 5e3                          # number of samples
    u  = np.random.normal(size=(d,K)) # samples in standard space
    x  = prior_pdf.U2X(u)             # samples in physical space
    # likelihood function evaluation
    L_eval = np.zeros(K,1)
    for i in range(K):
        L_eval[i] = likelihood(np.transpose(x[:,i]))
    c = 1/max(L_eval)                 # Ref. 1 Eq. 34
    
# use approximation to find c
elif method == 3:
    print('This method requires a large number of measurements')
    m = len(mu)
    p = 0.05
    c = 1/np.exp(-0.5*sp.stats.chi2.ppf(p,df=m))     # Ref. 1 Eq. 38
        
else:
    raise RuntimeError('Finding the scale constant c requires -method- 1, 2 or 3')

#=================================================================
# BUS-SuS step
N  = int(3e3)    # number of samples per level
p0 = 0.1         # probability of each subset

print('\nBUS with SUBSET SIMULATION: \n')
h, samplesU, samplesX, logcE, sigma = BUS_SuS(N, p0, c, log_likelihood, prior_pdf)

#=================================================================
# organize samples and show results
nsub        = len(h.flatten())  # number of levels + final posterior
u1p,u2p,u0p = list(),list(),list()
x1p,x2p,pp  = list(),list(),list()
#
for i in range(nsub):
   # samples in standard
   u1p.append(samplesU['total'][i][:,0])  
   u2p.append(samplesU['total'][i][:,1])
   u0p.append(samplesU['total'][i][:,2])
   # samples in physical
   x1p.append(samplesX[i][:,0])
   x2p.append(samplesX[i][:,1])
   pp.append( samplesX[i][:,2])
#=================================================================
# show results
print('\nModel evidence aBUS-SuS =', np.exp(logcE), '\n')
print('Mean value of x_1 =', np.mean(x1p[-1]))
print('Std of x_1 =', np.std(x1p[-1]),'\n')
print('Mean value of x_2 =', np.mean(x2p[-1])) 
print('Std of x_2 =', np.std(x2p[-1]))

#=================================================================
# Plots
# Options for font-family and font-size
plt.rc('font', size=12)
plt.rc('axes', titlesize=20)    # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('figure', titlesize=20)  # fontsize of the figure title

# plot samples in standard normal space
plt.figure()
plt.suptitle('Standard space')
for i in range(nsub):
   plt.subplot(2,3,i+1) 
   plt.plot(u1p[i],u2p[i],'r.') 
   plt.xlabel('$u_1$')
   plt.ylabel('$u_2$')
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()

# plot samples in original space
plt.figure()
plt.suptitle('Original space')
for i in range(nsub):
   plt.subplot(2,3,i+1) 
   plt.plot(x1p[i],x2p[i],'b.') 
   plt.xlabel('$x_1$')
   plt.ylabel('$x_2$')
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()