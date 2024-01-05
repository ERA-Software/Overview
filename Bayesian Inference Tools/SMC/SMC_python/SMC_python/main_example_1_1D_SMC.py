import numpy as np
import scipy as sp
from scipy import stats

## example 1: 1D posterior
"""
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
Contact: Antonis Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Version 2021-04
---------------------------------------------------------------------------
References:
1."Bayesian inference with subset simulation: strategies and improvements"
   Wolfgang Betz et al.
   Computer Methods in Applied Mechanics and Engineering 331 (2018) 72-93.
---------------------------------------------------------------------------

% add path to ERADist and ERANataf classes
% Source: https://www.bgu.tum.de/era/software/eradist/
Here we assume ERADist and ERANataf are stored in the same directory in folder
called "Distributions"
"""
from Distributions.ERADist import ERADist
from SMC_aCS import SMC_aCS
from SMC_GM import SMC_GM


## initial data
d = 1           # number of dimensions (number of uncertain parameters)

## prior PDF for theta
prior_pdf = ERADist('normal','PAR',[0,1])

## likelihood PDF (mixture of gaussians in 2D)
np.random.seed(2021)            # fixed the seed to get the same data points

#
mu             = 5
sigma          = 0.2
likelihood     = lambda theta: stats.norm.pdf(theta, mu, sigma)
log_likelihood = lambda theta: np.log(likelihood(theta)+np.finfo(float).tiny)    # tiny to avoid Inf values in log(0)

## SMC-SuS
N  = int(1e3)           # number of samples per level
p = 1                   # N/number of chains per level
burn = int(2)           # Burn-in length (per chain)
tarCoV = 1.5            # target coefficient of variation of the weights
k_init = int(2)

print('\nSequential Monte Carlo with aCS: \n\n')
[samplesU, samplesX, q, logcE] = SMC_aCS(N, p, log_likelihood, prior_pdf, burn, tarCoV)
#print('\nSequential Monte Carlo with GM: \n\n')
#[samplesU, samplesX, q, k_fin, logcE] = SMC_GM(N, p, log_likelihood, prior_pdf, k_init, burn, tarCoV)

## extract the samples
nsub = len(q)
u1p  = []
u0p  = []
x1p  = []
pp   = []

for i in range(nsub):
    # samples in standard
    u1p.append(samplesU[i])
    # samples in physical
    x1p.append(samplesX[i])

## reference and aBUS solutions
mu_exact    = 4.81
sigma_exact = 0.196
cE_exact    = 2.36e-6

# show results
print('\nExact model evidence = {:.3g}'.format(cE_exact))
print('\nModel evidence SMC-aCS = {:.3g}\n'.format(np.exp(logcE)))
print('\nExact posterior mean = {:.2f}'.format(mu_exact))
print('\nMean value of samples = {:.2f}\n'.format(np.mean(x1p[-1])))
print('\nExact posterior std = {:.3f}'.format(sigma_exact))
print('\nStd of samples = {:.3f}\n\n'.format(np.nanstd(x1p[-1])))
##END
