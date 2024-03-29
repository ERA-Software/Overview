import numpy as np
import matplotlib.pyplot as plt
from ERANataf import ERANataf
from ERADist import ERADist
from SuS import SuS
from Sim_Sensitivity import Sim_Sensitivity
plt.close('all')
"""
---------------------------------------------------------------------------
Subset Simulation: Ex. 1 Ref. 2  - Strongly nonlinear and non-monotonic LSF
---------------------------------------------------------------------------
Created by:
Daniel Koutas

Developed by:
Ivan Olarte-Rodriguez
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2023-12
* Modification in Sensitivity Analysis calls
---------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SuS"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
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

# %% Samples Return
samples_return = 1
# %% subset simulation
N  = 2000        # Total number of samples for each level
p0 = 0.1         # Probability of each subset, chosen adaptively

print('\n\nSUBSET SIMULATION: ')
[Pf_SuS, delta_SuS, b, Pf, b_sus, pf_sus, samplesU, samplesX, fs_iid] = SuS(N, p0, g, pi_pdf, samples_return)

# %% Implementation of sensitivity analysis

# Computation of Sobol Indices
compute_Sobol = True

# Computation of EVPPI (based on standard cost of failure (10^8) and cost
# of replacement (10^5)
compute_EVPPI = False

[S_F1, _] = Sim_Sensitivity(fs_iid, Pf_SuS, pi_pdf, compute_Sobol, compute_EVPPI)

# %% reference solution
pf_ref    = 0.0062
MC_S_F1  = [0.0811, 0.0045, 0.0398] # approximately read and extracted from paper

# %% show results
print('\n\n***Reference Pf: ***', pf_ref)
print('\n***SuS Pf: ***', Pf_SuS)

print('\n\n***MC Sobol indices:')
print(MC_S_F1)
print('\n***SuS Sobol indices:')
print(S_F1)
