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
Sequential importance sampling: Ex. 1 Ref. 2 - Steel column
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
2."Variance-based reliability sensitivity analysis and the FORM a-factors."
   Papaioannou & Straub
   Reliability Engineering and System Safety 210 (2021) 107496
---------------------------------------------------------------------------
"""

# %% Initialization
b   = 250
h   = 250
t_b = 15
t_h = 10
L   = 7.5e+3

A_s = 2*b*t_b + h*t_h
W_s = (h*t_h**3)/(6*b) + (t_b*b**2)/3
I_s = (h*t_h**3)/12 + (t_b*b**3)/6

# %% definition of the random variables
d       = 5          # number of dimensions
P_p     = ERADist('normal','MOM',[200e+3, 20e+3])
P_e     = ERADist('gumbel','MOM',[400e+3, 60e+3])
delta_0 = ERADist('normal','MOM',[30, 10])
f_y     = ERADist('lognormal','MOM',[400, 32])
E       = ERADist('lognormal','MOM',[2.1e+5, 8.4e+3])
pi_pdf  = [P_p, P_e, delta_0, f_y, E]

# correlation matrix
R = np.eye(d)   # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)    # if you want to include dependence

# %% limit state function in the original space
g  = lambda x: 1 - ((x[:,0] + x[:,1]) / (x[:,3]*A_s) + (x[:,0]+x[:,1])*x[:,2]/(x[:,3]*W_s) *
                     np.pi**2*(x[:,4]*I_s/L**2) / (np.pi**2*x[:,4]*I_s/L**2 - (x[:,0] + x[:,1])))

#print(g(np.asarray([[0,1,2,3,4],[1,2,3,4,5],[2,3,4,5,6],[0.2,0.2,0.002,0.002,0.002]])))

# Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1
# %% Sequential Importance Sampling
N      = 3000     # Total number of samples for each level
p      = 0.1      # Number of chains per level
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

# %% MC solution given in paper
# The MC results for S_F1_MC have the following COVs in the given order:
# [16.1%, 0.2%, 1.8%, 7.4%, 15.1%]
# Hence the first order indices (except for the second one) have quite high
# uncertainty and should not be considered as exact.
S_F1_MC   = [1.7e-4, 0.1974, 0.0044, 4.8e-4, 1.6e-4]

# The MC results for the total-effect indices have all COVs <= 0.2% so they
# can be considered as more accurate than the first-order indices
S_F1_T_MC = [0.2365, 0.9896, 0.7354, 0.3595, 0.2145]

# MC probability of failure
Pf_MC = 8.35e-4

# show results
print('\n***Refernce Pf: ***', Pf_MC)
print('\n***SIS Pf: ***', Pf_SIS)

print('\n\n***MC sensitivity indices:')
print(S_F1_MC)
print('\n***SIS sensitivity indices:')
print(S_F1)
