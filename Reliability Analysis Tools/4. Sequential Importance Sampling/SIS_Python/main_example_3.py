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
Sequential importance sampling: Ex. 1 Ref. 1 - convex limit state function
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
---------------------------------------------------------------------------
"""

# %% definition of the random variables
d      = 2          # number of dimensions
pi_pdf = [ERADist('standardnormal', 'PAR', np.nan) for _ in range(d)] # n independent rv

# correlation matrix
R = np.eye(d)   # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)    # if you want to include dependence

# limit state function
g = lambda u: np.amin(np.array([3.2 + (u[:, 0] + u[:, 1]) / np.sqrt(2),
            0.1 * (u[:, 0] - u[:, 1]) ** 2 - (u[:, 0] + u[:, 1]) / np.sqrt(2) + 2.5,
        ]),axis=0)

# %% Sequential Importance Sampling
N      = 2000     # Total number of samples for each level
p      = 0.1      # N/number of chains per level
k_init = 3        # Initial number of Gaussians in the Mixture Model (GM)
burn   = 0        # Burn-in period
tarCOV = 1.5      # Target COV of weights

# %% Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 1

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
S_F1_ref   = [0.0526, 0.0526]

# Print reference values for the first order indices
print("\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# reference solution
pf_ref = 4.90e-3

# show p_f results
print('\n***Reference Pf: ', pf_ref, ' ***')
print('***SIS Pf: ', Pf_SIS, ' ***\n')

# Options for font-family and font-size
plt.rc('font', size=12)
plt.rc('axes', titlesize=20)    # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('figure', titlesize=20)  # fontsize of the figure title

# %% Plot samples
if samplesU:
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
