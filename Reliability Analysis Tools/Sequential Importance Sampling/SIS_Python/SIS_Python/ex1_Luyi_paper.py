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
Sequential importance sampling: Ex. 1 Ref. 2 - strongly nonlinear and non-monotonic LSF
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2022-04
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

# %% Sequential Importance Sampling
N      = 8000     # Total number of samples for each level
p      = 0.1      # N/number of chains per level
k_init = 3        # Initial number of Gaussians in the Mixture Model (GM)
burn   = 0        # Burn-in period
tarCOV = 1.5      # Target COV of weights

print('\nSIS method: \n')
method = 'GM'

if method == 'GM':
    [Pf_SIS, lv, samplesU, samplesX, k_fin, S_F1] = SIS_GM(N, p, g, pi_pdf, k_init, burn, tarCOV)
elif method == 'aCS':
    [Pf_SIS, lv, samplesU, samplesX, S_F1] = SIS_aCS(N, p, g, pi_pdf, burn, tarCOV)
else:
    raise RuntimeError('Choose GM or aCS methods')


# refernce solution
pf_ref    = 0.0062
MC_S_F1  = [0.0811, 0.0045, 0.0398] # approximately read and extracted from paper

# show results
print('\n***Reference Pf: ***', pf_ref)
print('\n***SIS Pf: ***', Pf_SIS)

print('\n\n***MC sensitivity indices:')
print(MC_S_F1)
print('\n***SIS sensitivity indices:')
print(S_F1)
