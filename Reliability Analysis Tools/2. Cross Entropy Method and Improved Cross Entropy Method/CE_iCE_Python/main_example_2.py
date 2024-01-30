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

"""
---------------------------------------------------------------------------
Improved cross entropy method: Ex. 2 Ref. 2 - linear function of independent exponential
---------------------------------------------------------------------------
Created by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Fong-Lin Wu
Matthias Willer
Peter Kaplan
Luca Sardi
Daniel Koutas
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
   Structural Safety, 76, 15-27
---------------------------------------------------------------------------
"""

# %% definition of the random variables
# d = 100  # number of dimensions
d = 2  # number of dimensions
pi_pdf = [ERADist("exponential", "PAR", 1) for _ in range(d)]  # n independent rv

# correlation matrix
R = np.eye(d)  # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)  # if you want to include dependence

# limit state function
# Ca = 140
Ca = 10
g = lambda x: Ca - np.sum(x, axis=1)

# %% Definition of additional values
max_it    = 100   # maximum number of iteration steps per simulation
N         = 1000  # definition of number of samples per level
CV_target = 2.5   # target CV

# CE method
p      = 0.1  # quantile value to select samples for parameter update
k_init = 1    # initial number of distributions in the Mixture Model (GM/vMFNM)

# Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 2

# %% Run methods
print("\nCE-based IS stage: ")
#method = "iCE_SG"
# method = "iCE_GM"
#method = "iCE_vMFNM"
method = "CE_SG"
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
compute_Sobol = True

# Computation of EVPPI (based on standard cost of failure (10^8) and cost
# of replacement (10^5)
compute_EVPPI = True

[S_F1, S_EVPPI] = Sim_Sensitivity(fs_iid, Pf_CE, pi_pdf, compute_Sobol, compute_EVPPI)

# %% Reference values
# The reference values for the first order indices
S_F1_ref   = [0.2021, 0.1891]

# Print reference values for the first order indices
print("\n\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# exact solution
lam = 1
pf_ex = 1 - sp.stats.gamma.cdf(Ca, a=d)

# show p_f results
print("\n***Exact Pf: ", pf_ex, " ***")
print("***CE-based IS Pf: ", Pf_CE, " ***\n")
