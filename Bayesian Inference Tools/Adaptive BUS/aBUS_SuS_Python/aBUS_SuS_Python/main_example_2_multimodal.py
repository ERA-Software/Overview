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
----------------------------------------------------
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
from aBUS_SuS import aBUS_SuS
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
m     = 10                        # number of data points
mu    = prior_pdf.random(m)       # Here using ERANataf.random
Sigma = (0.1**2)*np.eye(d)
w     = 0.1*np.ones(m)

# define likelihood
realmin        = np.finfo(np.double).tiny  # to prevent log(0)
likelihood     = lambda theta: sum(w[i] * mvn.pdf(theta, mu[i], Sigma) for i in range(0,m))
log_likelihood = lambda theta: np.log(likelihood(theta.T ) + realmin)

#=================================================================
# aBUS-SuS step
N  = int(3e3)    # number of samples per level
p0 = 0.1         # probability of each subset

print('\naBUS with SUBSET SIMULATION: \n')
h, samplesU, samplesX, logcE, c, sigma = aBUS_SuS(N, p0, log_likelihood, prior_pdf)

#=================================================================
# organize samples and show results
nsub        = len(h.flatten())  # number of levels
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
