"""
---------------------------------------------------------------------------
Example 1: 1D posterior
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
1."Transitional Markov Chain Monte Carlo: Observations and improvements".
   Wolfgang Betz et al.
   Journal of Engineering Mechanics. 142.5 (2016) 04016016.1-10
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
from iTMCMC import iTMCMC
plt.close('all')

#=================================================================
# define the prior
d           = 1        # number of dimensions (uncertain parameters)
prior_pdf_1 = ERADist('normal','PAR',[0,1])
prior_pdf   = [prior_pdf_1]

# correlation matrix
# R = np.eye(d)   # independent case

# object with distribution information
# prior_pdf = ERANataf(prior_pdf,R)

#=================================================================
# define likelihood
mu_obs    = 5
sigma_obs = 0.2
#
realmin        = np.finfo(np.double).tiny  # to prevent log(0)
likelihood     = lambda theta: mvn.pdf(theta, mu_obs, sigma_obs**2)
log_likelihood = lambda theta: np.log(likelihood(theta) + realmin)

#=================================================================
# iTMCMC step
Ns = int(1e3)      # number of samples per level
Nb = int(0.1*Ns)   # burn-in period

print('\n\niTMCMC: ')
samplesU, samplesX, q, logcE = iTMCMC(Ns, Nb, log_likelihood, prior_pdf)

#=================================================================
# reference and numerical solutions
mu_exact    = 4.81     
sigma_exact = 0.196
cE_exact    = 2.36e-6
m           = len(q)   # number of stages (intermediate levels)

# show results
print('\nExact model evidence = ', cE_exact)
print('Model evidence iTMCMC = ', np.exp(logcE))
print('\nExact posterior mean = ', mu_exact)
print('Mean value of samples = ', np.mean(samplesX[-1]))
print('\nExact posterior std = ', sigma_exact)
print('Std of samples = ', np.std(samplesX[-1]))

#=================================================================
# Plots
# Options for font-family and font-size
plt.rc('font', size=12)
plt.rc('axes', titlesize=20)    # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('figure', titlesize=20)  # fontsize of the figure title

# plot q values
plt.figure()
plt.plot(np.arange(0,m), q, 'r*-')
plt.xlabel(r'Intermediate levels $j$') 
plt.ylabel(r'$q_j$')
plt.tight_layout()
plt.show()