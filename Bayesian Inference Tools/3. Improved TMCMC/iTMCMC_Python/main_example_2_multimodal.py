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
---------------------------------------------------------------------------
Last Version 2021-03
* Adaptation to new ERANataf class
---------------------------------------------------------------------------
References:
1."Transitional Markov Chain Monte Carlo: Observations and improvements".
   Wolfgang Betz et al.
   Journal of Engineering Mechanics. 142.5 (2016) 04016016.1-10
2."Asymptotically independent Markov sampling: a new MCMC scheme for Bayesian inference"
   James L. Beck and Konstantin M. Zuev
   International Journal for Uncertainty Quantification, 3.5 (2013) 445-474.
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
from iTMCMC import iTMCMC
plt.close('all')
np.random.seed(2021)   # fix seed

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
mu    = prior_pdf.random(m)         # Here using ERANataf.random
Sigma = (0.1**2)*np.eye(d)
w     = 0.1*np.ones(m)

# define likelihood
realmin        = np.finfo(np.double).tiny  # to prevent log(0)
likelihood     = lambda theta: sum(w[i] * mvn.pdf(theta, mu[i], Sigma) for i in range(0,m))
log_likelihood = lambda theta: np.log(likelihood(theta.T) + realmin)

#=================================================================
# iTMCMC step
Ns = int(1e3)      # number of samples per level
Nb = int(0.1*Ns)   # burn-in period

print('\n\niTMCMC: ')
samplesU, samplesX, q, logcE = iTMCMC(Ns, Nb, log_likelihood, prior_pdf)

#=================================================================
m = len(q)   # number of stages (intermediate levels)

# show results
print('\nModel evidence =', np.exp(logcE),'\n')
print('Mean value of x_1 =', np.mean(samplesX[-1][:,0]))
print('Std of x_1 =', np.std(samplesX[-1][:,0]),'\n')
print('Mean value of x_2 =', np.mean(samplesX[-1][:,1])) 
print('Std of x_2 =', np.std(samplesX[-1][:,1]))

#=================================================================
# Plots
# Options for font-family and font-size
plt.rc('font', size=12)
plt.rc('axes', titlesize=20)    # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('figure', titlesize=20)  # fontsize of the figure title

# plot q values
plt.figure()
plt.plot(np.arange(0,m),q,'ro-')
plt.xlabel(r'Intermediate levels $j$') 
plt.ylabel(r'$q_j$')
plt.tight_layout()
   
# plot samples increasing q
idx = np.array([0, int(np.round((m-1)/3)), int(np.round(2*(m-1)/3)), m-1])
plt.figure()
for i in range(4):
   plt.subplot(2,2,i+1) 
   plt.plot(samplesX[idx[i]][:,0], samplesX[idx[i]][:,1],'b.')
   plt.title(r'$p_j=' + str(np.round(q[idx[i]],5)) + r'$')
   plt.xlabel(r'$x_1$')
   plt.ylabel(r'$x_2$')
plt.tight_layout()

plt.show()