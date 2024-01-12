import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ERANataf import ERANataf
from ERADist import ERADist
from SIS_GM import SIS_GM
from SIS_aCS import SIS_aCS
plt.close('all')
"""
---------------------------------------------------------------------------
Sequential importance sampling: Ex. 2 Ref. 2 - linear function of independent exponential
---------------------------------------------------------------------------
Created by:
Sebastian Geyer
Iason Papaioannou
Daniel Straub

Assistant Developers:
Matthias Willer
Peter Kaplan
Luca Sardi
Daniel Koutas

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2022-04
* inlcusion of sensitivity analysis
---------------------------------------------------------------------------
Based on:
1. "Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
2."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
"""

# %% definition of the random variables
# d      = 100         # number of dimensions
d      = 2         # number of dimensions
pi_pdf = [ERADist('exponential', 'PAR', 1) for i in range(d)] # n independent rv

# correlation matrix
#R = np.eye(d)   # independent case

# object with distribution information
#pi_pdf = ERANataf(pi_pdf,R)    # if you want to include dependence

# %% limit state function
# Ca = 140
Ca = 10
g = lambda x: Ca - np.sum(x, axis=1)

# %% Sequential Importance Sampling
N      = 2000     # Total number of samples for each level
p      = 0.1      # N/number of chains per level
k_init = 3        # Initial number of Gaussians in the Mixture Model (GM)
burn   = 0        # Burn-in period
tarCOV = 1.5      # Target COV of weights

# %% Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 1

# Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1

print('\nSIS stage: ')
method = 'aCS'
if method == 'GM':      # gaussian mixture
    [Pf_SIS, l, samplesU, samplesX, k_fin, S_F1] = SIS_GM(N, p, g, pi_pdf, k_init, burn, tarCOV, sensitivity_analysis, samples_return)
elif method == 'aCS':   # adaptive conditional sampling
    [Pf_SIS, l, samplesU, samplesX, S_F1] = SIS_aCS(N, p, g, pi_pdf, burn, tarCOV, sensitivity_analysis, samples_return)
else:
    print('\nChoose GM, or aCS methods')

# Reference values
# The reference values for the first order indices
S_F1_ref   = [0.2021, 0.1891]

# Print reference values for the first order indices
print("\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# exact solution
lam      = 1
pf_ex    = 1 - sp.stats.gamma.cdf(Ca, a=d, scale=lam)
Pf_exact = lambda gg: 1-sp.stats.gamma.cdf(Ca-gg, a=d, scale=lam)
gg       = np.linspace(0,30,300)

# show p_f results
print('\n***Exact Pf: ', pf_ex, ' ***')
print('\n***SIS Pf: ', Pf_SIS, ' ***\n')
