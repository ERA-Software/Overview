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

plt.close('all')
"""
---------------------------------------------------------------------------
Improved cross entropy method: Ex. 1 Ref. 3 - strongly nonlinear and non-monotonic LSF
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2022-04
---------------------------------------------------------------------------
Based on:
1. Papaioannou, I., Geyer, S., & Straub, D. (2019).
   Improved cross entropy-based importance sampling with a flexible mixture model.
   Reliability Engineering & System Safety, 191
2. Geyer, S., Papaioannou, I., & Straub, D. (2019).
   Cross entropy-based importance sampling using Gaussian densities revisited.
   Structural Safety, 76, 15–27
3. Li, L., Papaioannou, I., & Straub, D. (2019)
   Global reliability sensitivity estimation based on failure samples"
   Structural Safety 81 101871
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

# Definition of additional values
max_it    = 100   # maximum number of iteration steps per simulation
N         = 4000  # definition of number of samples per level
CV_target = 2.0   # target CV

# CE method
p      = 0.1  # quantile value to select samples for parameter update
k_init = 4    # initial number of distributions in the Mixture models (GM/vMFNM)

print("\nCE-based IS stage: ")
method = "iCE_SG"
# method = "iCE_GM"
# method = "iCE_vMFNM"
# method = "CE_SG"
# method = "CE_GM"
# method = "CE_vMFNM"

if method == "iCE_SG": # improved CE with single gaussian
    [Pr, l, N_tot, samplesU, samplesX, S_F1] = iCE_SG(N, p, g, pi_pdf, max_it, CV_target)

elif method == "iCE_GM": # improved CE with gaussian mixture
    [Pr, l, N_tot, samplesU, samplesX, k_fin, S_F1] = iCE_GM(N, p, g, pi_pdf, max_it, CV_target, k_init)

elif method == "iCE_vMFNM": # improved CE with adaptive vMFN mixture
    [Pr, l, N_tot, samplesU, samplesX, k_fin, S_F1] = iCE_vMFNM(N, p, g, pi_pdf, max_it, CV_target, k_init)

elif method == "CE_SG": # single gaussian
    [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin, S_F1] = CEIS_SG(N, p, g, pi_pdf)

elif method == "CE_GM": # gaussian mixture
    [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin, S_F1] = CEIS_GM(N, p, g, pi_pdf, k_init)

elif method == "CE_vMFNM": # adaptive vMFN mixture
    [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin, S_F1] = CEIS_vMFNM(N, p, g, pi_pdf, k_init)

else:
    print("\nChoose SG, GM, or vMFNM methods")


# reference solution
pf_ref    = 0.0062
MC_S_F1  = [0.0811, 0.0045, 0.0398] # approximately read and extracted from paper

# show results
print('\n***Reference Pf: ***', pf_ref)
print('\n***CE Pf: ***', Pr)

print('\n\n***MC sensitivity indices:')
print(MC_S_F1)
print('\n***CE sensitivity indices:')
print(S_F1)
