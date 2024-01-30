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
Improved cross entropy method: Ex. 2 Ref. 2 - parabolic/concave limit state function
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
d = 2  # number of dimensions
pi_pdf = [
    ERADist("standardnormal", "PAR", [0, 1]) for i in range(d)
]  # n independent rv

# correlation matrix
R = np.identity(d)   # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)    # if you want to include dependence

# %% limit state function
b = 5.0
kappa = 0.5
e = 0.1
g = lambda u: b - u[:, 1] - kappa * (u[:, 0] - e) ** 2

# %%Definition of additional values
max_it    = 100   # maximum number of iteration steps per simulation
N         = 1000  # definition of number of samples per level
CV_target = 2.0   # target CV

# CE method
p      = 0.1  # quantile value to select samples for parameter update
k_init = 2    # initial number of distributions in the Mixture models (GM/vMFNM)

# Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1

# %% Run methods
print("\nCE-based IS stage: ")
#method = "iCE_SG"
# method = "iCE_GM"
method = "iCE_vMFNM"
#method = "CE_SG"
#method = "CE_GM"
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

[S_F1, S_EVPPI] = Sim_Sensitivity(fs_iid, Pf_CE, pi_pdf,compute_Sobol, compute_EVPPI)

# Reference values
# The reference values for the first order indices
S_F1_ref   = [0.4377, 0.0078]

# Print reference values for the first order indices
print("\n\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# reference solution
pf_ref = 3.01e-3

# show p_f results
print("\n***Reference Pf: ", pf_ref, " ***")
print("***CE-based IS Pf: ", Pf_CE, " ***\n")

# Plot samples
if samplesU:
    if d == 2:
        nnp = 200
        xx = np.linspace(-6, 6, nnp)
        [X, Y] = np.meshgrid(xx, xx)
        xnod = np.vstack([X.T.ravel(), Y.T.ravel()]).T

        Z = g(xnod)
        Z = Z.reshape(X.shape).T

        fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
        ax.contour(X, Y, Z, [0], colors="r", linewidths=3)  # LSF
        for sample in samplesU:
            ax.plot(*sample.T, ".", markersize=2)
        plt.xlabel(r'$u_1$')
        plt.ylabel(r'$u_2$', rotation=0)
        plt.show()
