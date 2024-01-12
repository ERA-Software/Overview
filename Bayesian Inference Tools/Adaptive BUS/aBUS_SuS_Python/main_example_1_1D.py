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
1."Bayesian inference with subset simulation: strategies and improvements"
   Wolfgang Betz et al.
   Computer Methods in Applied Mechanics and Engineering 331 (2018) 72-93.
---------------------------------------------------------------------------
"""

#=================================================================
# import libraries
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal as mvn
import matplotlib.pylab as plt

# Make sure ERADist, ERANataf classes are in the path
# https://www.bgu.tum.de/era/software/eradist/
from ERADist import ERADist
from ERANataf import ERANataf
from aBUS_SuS import aBUS_SuS
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
# aBUS-SuS step
N  = int(3e3)    # number of samples per level
p0 = 0.1         # probability of each subset

print('\naBUS with SUBSET SIMULATION: \n')
h, samplesU, samplesX, logcE, c, sigma = aBUS_SuS(N, p0, log_likelihood, prior_pdf)

#=================================================================
# organize samples and show results
nsub    = len(h.flatten())  # number of levels
u1p,u0p = list(),list()
x1p,pp  = list(),list()
#
for i in range(nsub):
   # samples in standard
   u1p.append(samplesU['total'][i][:,0])
   u0p.append(samplesU['total'][i][:,1])
   # samples in physical
   x1p.append(samplesX[i][:,0])
   pp.append(samplesX[i][:,1])

#=================================================================
# reference and BUS solutions
mu_exact    = 4.81
sigma_exact = 0.196
cE_exact    = 2.36e-6

# show results
print('\nExact model evidence = ', cE_exact)
print('Model evidence aBUS-SuS = ', np.exp(logcE))
print('\nExact posterior mean = ', mu_exact)
print('Mean value of samples = ', np.mean(x1p[-1]))
print('\nExact posterior std = ', sigma_exact)
print('Std of samples = ', np.std(x1p[-1]),'\n')
