import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ERANataf import ERANataf
from ERADist import ERADist
from SIS_GM import SIS_GM
from SIS_aCS import SIS_aCS
from SIS_vMFNM import SIS_vMFNM
from Sim_Sensitivity import Sim_Sensitivity
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
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Current Version 2023-10
* Modification of Sensitivity Analysis Calls
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

# Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1

print('\nSIS stage: ')
method = 'vMFNM'
if method == 'GM':      # gaussian mixture
    [Pf_SIS, l, samplesU, samplesX, k_fin, W_final, fs_iid] = SIS_GM(N, p, g, pi_pdf, k_init, burn, tarCOV , 
                                                                     samples_return)
elif method == 'aCS':   # adaptive conditional sampling
    [Pf_SIS, l, samplesU, samplesX, W_final, fs_iid] = SIS_aCS(N, p, g, pi_pdf, burn, tarCOV,  samples_return)
elif method == 'vMFNM':      # Von Mises-Fisher-Nakagami mixture
    [Pf_SIS, l, samplesU, samplesX, W_final, k_fin, fs_iid] = SIS_vMFNM(N, p, g, pi_pdf, k_init, burn, tarCOV , 
                                                                        samples_return)
else:
    print('\nChoose GM, vMFNM or aCS methods')

# %% Implementation of sensitivity analysis

# Computation of Sobol Indices
compute_Sobol = True

# Computation of EVPPI (based on standard cost of failure (10^8) and cost
# of replacement (10^5)
compute_EVPPI = True

[S_F1, S_EVPPI] = Sim_Sensitivity(fs_iid, Pf_SIS, pi_pdf, compute_Sobol, compute_EVPPI)

# %% Reference values
# The reference values for the first order indices
S_F1_ref   = [0.2021, 0.1891]

# Print reference values for the first order indices
print("\n\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# exact solution
lam      = 1
pf_ex    = 1 - sp.stats.gamma.cdf(Ca, a=d, scale=lam)
Pf_exact = lambda gg: 1-sp.stats.gamma.cdf(Ca-gg, a=d, scale=lam)
gg       = np.linspace(0,30,300)

# show p_f results
print('\n***Exact Pf: ', pf_ex, ' ***')
print('\n***SIS Pf: ', Pf_SIS, ' ***\n')
