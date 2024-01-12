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

"""
---------------------------------------------------------------------------
Improved cross entropy method: Ex. 1 Ref. 2 - linear function of independent standard normal
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

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2022-04
* inlcusion of sensitivity analysis
---------------------------------------------------------------------------
Based on:
1. Papaioannou, I., Geyer, S., & Straub, D. (2019).
   Improved cross entropy-based importance sampling with a flexible mixture model.
   Reliability Engineering & System Safety, 191
2. Geyer, S., Papaioannou, I., & Straub, D. (2019).
   Cross entropy-based importance sampling using Gaussian densities revisited.
   Structural Safety, 76, 15–27
---------------------------------------------------------------------------
"""

# definition of the random variables
d = 2  # number of dimensions
pi_pdf = [
    ERADist("standardnormal", "PAR", [0, 1]) for i in range(d)
]  # n independent rv

# correlation matrix
R = np.eye(d)  # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)  # if you want to include dependence

# limit state function
beta = 3.5
g = lambda x: -np.sum(x, axis=1) / np.sqrt(d) + beta

# Definition of additional values
max_it    = 100   # maximum number of iteration steps per simulation
N         = 1000  # definition of number of samples per level
CV_target = 2.0   # target CV

# CE method
p      = 0.1  # quantile value to select samples for parameter update
k_init = 1    # initial number of distributions in the Mixture models (GM/vMFNM)

# Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 1

# Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1

print("\nCE-based IS stage: ")
# method = "iCE_SG"
# method = "iCE_GM"
method = "iCE_vMFNM"
# method = "CE_SG"
# method = "CE_GM"
# method = "CE_vMFNM"

if method == "iCE_SG": # improved CE with single gaussian
    [Pr, l, N_tot, samplesU, samplesX, S_F1] = iCE_SG(N, p, g, pi_pdf, max_it, CV_target, sensitivity_analysis, samples_return)

elif method == "iCE_GM": # improved CE with gaussian mixture
    [Pr, l, N_tot, samplesU, samplesX, k_fin, S_F1] = iCE_GM(N, p, g, pi_pdf, max_it, CV_target, k_init, sensitivity_analysis, samples_return)

elif method == "iCE_vMFNM": # improved CE with adaptive vMFN mixture
    [Pr, l, N_tot, samplesU, samplesX, k_fin, S_F1] = iCE_vMFNM(N, p, g, pi_pdf, max_it, CV_target, k_init, sensitivity_analysis, samples_return)

elif method == "CE_SG": # single gaussian
    [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin, S_F1] = CEIS_SG(N, p, g, pi_pdf, sensitivity_analysis, samples_return)

elif method == "CE_GM": # gaussian mixture
    [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin, S_F1] = CEIS_GM(N, p, g, pi_pdf, k_init, sensitivity_analysis, samples_return)

elif method == "CE_vMFNM": # adaptive vMFN mixture
    [Pr, l, N_tot, gamma_hat, samplesU, samplesX, k_fin, S_F1] = CEIS_vMFNM(N, p, g, pi_pdf, k_init, sensitivity_analysis, samples_return)

else:
    print("\nChoose SG, GM, or vMFNM methods")

# Reference values
# The reference values for the first order indices
S_F1_ref   = [0.0315, 0.0315]

# Print reference values for the first order indices
print("\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# exact solution
pf_ex = sp.stats.norm.cdf(-beta)

# show p_f results
print("\n***Exact Pf: ", pf_ex, " ***")
print("***CE-based IS Pf: ", Pr, " ***\n")

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
        plt.show()
