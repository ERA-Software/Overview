import numpy as np
import matplotlib.pyplot as plt
from ERANataf import ERANataf
from ERADist import ERADist
from SuS import SuS
from Sim_Sensitivity import Sim_Sensitivity
plt.close('all')
"""
---------------------------------------------------------------------------
Subset Simulation: Ex. 3 Ref. 3 - system reliability limit state function
---------------------------------------------------------------------------
Created by:
Matthias Willer
Felipe Uribe
Luca Sardi
Daniel Koutas
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
2."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
3."Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
---------------------------------------------------------------------------
"""

# %% definition of the random variables
d      = 2          # number of dimensions
pi_pdf = list()
for i in range(d):
    pi_pdf.append(ERADist('standardnormal', 'PAR', np.nan)) # n independent rv

# correlation matrix
R = np.eye(d)   # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)    # if you want to include dependence

# %% limit-state function
g = lambda u: np.amin(np.array([ 0.1 * (u[:, 0] - u[:, 1]) ** 2 - (u[:, 0]  + u[:, 1]) / np.sqrt(2) + 3,
            0.1 * (u[:, 0]  - u[:, 1]) ** 2 + (u[:, 0]  + u[:, 1]) / np.sqrt(2) + 3,
            u[:, 0]  - u[:, 1] + 7 / np.sqrt(2),
            u[:, 1] - u[:, 0]  + 7 / np.sqrt(2),]),axis=0,)

# %% subset simulation
N  = 2000        # Total number of samples for each level
p0 = 0.1         # Probability of each subset, chosen adaptively

# Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1

print('\n\nSUBSET SIMULATION: ')
[Pf_SuS, delta_SuS, b, Pf, b_sus, pf_sus, samplesU, samplesX, fs_iid] = SuS(N, p0, g, pi_pdf, samples_return)

# %% Implementation of sensitivity analysis

# Computation of Sobol Indices
compute_Sobol = True

# Computation of EVPPI (based on standard cost of failure (10^8) and cost
# of replacement (10^5)
compute_EVPPI = True

[S_F1, S_EVPPI] = Sim_Sensitivity(fs_iid, Pf_SuS, pi_pdf, compute_Sobol, compute_EVPPI)

# %% Reference values
# The reference values for the first order indices
S_F1_ref   = [0.0481, 0.0481]

# Print reference values for the first order indices
print("\n\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# %% reference solution
pf_ref = 2.2e-3

# show p_f results
print('\n***Reference Pf: ', pf_ref, ' ***')
print('***SuS Pf: ', Pf_SuS, ' ***\n\n')

# %% Plots
# Options for font-family and font-size
plt.rc('font', size=12)
plt.rc('axes', titlesize=20)    # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('figure', titlesize=20)  # fontsize of the figure title

# Plot samples
if samplesU['total']:
    xx = np.linspace(-7, 7, 300)
    [X, Y] = np.meshgrid(xx, xx)
    xnod = np.vstack([X.T.ravel(), Y.T.ravel()]).T

    Z = g(xnod)
    Z = Z.reshape(X.shape)

    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    ax.contour(X, Y, Z, [0], colors="r", linewidths=3)  # LSF
    for sample in samplesU['total']:
        ax.plot(*sample.T, ".", markersize=2)
        plt.xlabel(r'$u_1$')
        plt.ylabel(r'$u_2$', rotation=0)

# Plot failure probability: Exact
plt.figure()
plt.yscale('log')
plt.title('Failure probability estimate')
plt.xlabel(r'Limit state function $g$')
plt.ylabel(r'Failure probability $P_f$')

# Plot failure probability: SuS
plt.plot(b_sus,pf_sus,'r--', label='SuS')           # curve
plt.plot(b,Pf,'ko', label='Intermediate levels',
                    markersize=8,
                    markerfacecolor='none')         # points
plt.plot(0,Pf_SuS,'b+', label='Pf SuS',
                        markersize=10,
                        markerfacecolor='none')
plt.plot(0,pf_ref,'ro', label='Pf Ref.',
                       markersize=10,
                       markerfacecolor='none')
plt.legend()
plt.tight_layout()
plt.show()
# %%END
