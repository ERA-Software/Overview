import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ERANataf import ERANataf
from ERADist import ERADist

# improved version
from iCE_SG import iCE_SG
from iCE_GM import iCE_GM
from iCE_vMFNM import iCE_vMFNM

# original version
from CEIS_SG import CEIS_SG
from CEIS_GM import CEIS_GM
from CEIS_vMFNM import CEIS_vMFNM

# Sensitivity Analysis
from Sim_Sensitivity import Sim_Sensitivity

plt.close('all')
"""
---------------------------------------------------------------------------
Improved cross entropy method: Ex. 1 Ref. 3 - strongly nonlinear and non-monotonic LSF
---------------------------------------------------------------------------
Created by:
Daniel Koutas

Assistant Developers:
Ivan Olarte-Rodriguez 

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2023-12
* Modification of Sensitivity Analysis Calls
---------------------------------------------------------------------------
Based on:
1. Papaioannou, I., Geyer, S., & Straub, D. (2019).
   Improved cross entropy-based importance sampling with a flexible mixture model.
   Reliability Engineering & System Safety, 191
2. Geyer, S., Papaioannou, I., & Straub, D. (2019).
   Cross entropy-based importance sampling using Gaussian densities revisited.
   Structural Safety, 76, 15–27
3. Papaioannou, I.,  & Straub, D. (2021)
   Variance-based reliability sensitivity analysis and the FORM a-factors."
   Reliability Engineering and System Safety 210 107496
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

# Definition of additional values
max_it    = 100   # maximum number of iteration steps per simulation
N         = 4000  # definition of number of samples per level
CV_target = 2.0   # target CV

# CE method
p      = 0.1  # quantile value to select samples for parameter update
k_init = 1    # initial number of distributions in the Mixture models (GM/vMFNM)

# %% Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1

# %% Run methods
print("\nCE-based IS stage: ")
#method = "iCE_SG"
method = "iCE_GM"
#method = "iCE_vMFNM"
#method = "CE_SG"
# method = "CE_GM"
#method = "CE_vMFNM"

if method == "iCE_SG": # improved CE with single gaussian
    [Pf_CE, lv, N_tot, samplesU, samplesX, W_final, fs_iid] = iCE_SG(N, p, g, pi_pdf, max_it, CV_target, samples_return)

elif method == "iCE_GM": # improved CE with gaussian mixture
    [Pf_CE, lv, N_tot, samplesU, samplesX, k_fin, W_final, fs_iid] = iCE_GM(N, p, g, pi_pdf, max_it, CV_target, k_init, samples_return)

elif method == "iCE_vMFNM": # improved CE with adaptive vMFN mixture
    [Pf_CE, l, N_tot, samplesU, samplesX, k_fin, W_final, fs_iid] = iCE_vMFNM(N, p, g, pi_pdf, max_it, CV_target, k_init, samples_return)

elif method == "CE_SG": # single gaussian
    [Pf_CE, l, N_tot, gamma_hat, samplesU, samplesX, k_fin, W_final, fs_iid] = CEIS_SG(N, p, g, pi_pdf, samples_return)

elif method == "CE_GM": # gaussian mixture
    [Pf_CE, l, N_tot, gamma_hat, samplesU, samplesX, k_fin, W_final, fs_iid] = CEIS_GM(N, p, g, pi_pdf, k_init, samples_return)

elif method == "CE_vMFNM": # adaptive vMFN mixture
    [Pf_CE, l, N_tot, gamma_hat, samplesU, samplesX, k_fin, W_final, fs_iid] = CEIS_vMFNM(N, p, g, pi_pdf, k_init, samples_return)

else:
    print("\nChoose SG, GM, or vMFNM methods")

# %% Implementation of sensitivity analysis

# Computation of Sobol Indices
compute_EVPPI = True

# Computation of EVPPI (based on standard cost of failure (10^8) and cost
# of replacement (10^5)
compute_EVPPI = False

[S_F1, S_EVPPI] = Sim_Sensitivity(fs_iid, Pf_CE, pi_pdf, compute_EVPPI, compute_EVPPI)

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
print('\n\n***Reference Pf: ***', Pf_MC)
print('\n***CE Pf: ***', Pf_CE)

print('\n\n***MC sensitivity indices:')
print(S_F1_MC)
print('\n***CE sensitivity indices:')
print(S_F1)
