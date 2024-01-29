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
Sequential importance sampling: Ex. 1 Ref. 2 - strongly nonlinear and non-monotonic LSF
---------------------------------------------------------------------------
Created by:
Daniel Koutas

Additional Developers:
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2022-04
---------------------------------------------------------------------------
Current Version 2023-10
* Modification of Sensitivity Analysis Calls
---------------------------------------------------------------------------
Comments:
* The SIS method in combination with a Gaussian Mixture model can only be
  applied for low-dimensional problems, since its accuracy decreases
  dramatically in high dimensions.
---------------------------------------------------------------------------
Based on:
1."Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
2."Global reliability sensitivity estimation based on failure samples"
   Luyi et al.
   Structural Safety 81 (2019) 101871
---------------------------------------------------------------------------
"""

# %% definition of the random variables
d      = 3          # number of dimensions
pi_pdf = [ERADist('standardnormal', 'PAR', np.nan) for _ in range(d)] # n independent rv

# correlation matrix
R = np.eye(d)   # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)    # if you want to include dependence

# %% limit state function
g    = lambda x: x[:,0]**3 + 10*x[:,1]**2 + 0.1*np.sin(np.pi*x[:,1]) + 10*x[:,2]**2 + 40*np.sin(np.pi*x[:,2]) + 38

# Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1
# %% Sequential Importance Sampling
N      = 8000     # Total number of samples for each level
p      = 0.1      # N/number of chains per level
k_init = 3        # Initial number of Gaussians in the Mixture Model (GM)
burn   = 0        # Burn-in period
tarCOV = 1.5      # Target COV of weights

print('\nSIS method: \n')
method = 'vMFNM'
if method == 'GM':      # gaussian mixture
    [Pf_SIS, l, samplesU, samplesX, k_fin, fs_iid] = SIS_GM(N, p, g, pi_pdf, k_init, burn, tarCOV , samples_return)
elif method == 'aCS':   # adaptive conditional sampling
    [Pf_SIS, l, samplesU, samplesX, fs_iid] = SIS_aCS(N, p, g, pi_pdf, burn, tarCOV,  samples_return)
elif method == 'vMFNM':      # Von Mises-Fisher-Nakagami mixture
    [Pf_SIS, l, samplesU, samplesX, k_fin, fs_iid] = SIS_vMFNM(N, p, g, pi_pdf, k_init, burn, tarCOV , samples_return)
else:
    print('\nChoose GM, vMFNM or aCS methods')

# %% Implementation of sensitivity analysis

# Computation of Sobol Indices
compute_Sobol = True

# Computation of EVPPI (based on standard cost of failure (10^8) and cost
# of replacement (10^5)
compute_EVPPI = True

[S_F1, S_EVPPI] = Sim_Sensitivity(fs_iid, Pf_SIS, pi_pdf, compute_Sobol, compute_EVPPI)

# %% reference solution
pf_ref    = 0.0062
MC_S_F1  = [0.0811, 0.0045, 0.0398] # approximately read and extracted from paper

# show results
print('\n\n***Reference Pf: ***', pf_ref)
print('\n***SIS Pf: ***', Pf_SIS)

print('\n\n***MC sensitivity indices:')
print(MC_S_F1)
print('\n***SIS sensitivity indices:')
print(S_F1)
