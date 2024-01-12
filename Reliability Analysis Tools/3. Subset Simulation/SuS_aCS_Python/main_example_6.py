import numpy as np
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from SuS import SuS
plt.close('all')
"""
---------------------------------------------------------------------------
Subset Simulation: Ex. 5 Ref. 2 - points outside of hypersphere
---------------------------------------------------------------------------
Created by:
Matthias Willer
Felipe Uribe
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
1."Estimation of small failure probabilities in high dimentions by SuS"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277.
2."Bayesian inference of engineering models"
   Wolfgang Betz
   PhD thesis.
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

# %% limit state function
r = 5.26
m = 1
g = lambda u: 1 - (np.sqrt(np.sum(u ** 2, 1)) / r) ** 2- (u[:, 0] / r) * ((1 - (np.sqrt(np.sum(u ** 2, 1)) / r) ** m) / (1 + (np.sqrt(np.sum(u ** 2, 1)) / r) ** m))
# %% subset simulation
N  = 2000        # Total number of samples for each level
p0 = 0.1         # Probability of each subset, chosen adaptively

# Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 0

# Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1

print('\n\nSUBSET SIMULATION: ')
[Pf_SuS, delta_SuS, b, Pf, b_sus, pf_sus, samplesU, samplesX, S_F1] = SuS(N, p0, g, pi_pdf, sensitivity_analysis, samples_return)

# Reference values
# The reference values for the first order indices
S_F1_ref   = [0.1857, 0.1857]

# Print reference values for the first order indices
print("\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# reference solution
pf_ref = 1e-6

# show p_f results
print('\n***Reference Pf: ', pf_ref, ' ***')
print('***SuS Pf: ', Pf_SuS, ' ***\n\n')

# %% Plots
# Options for font-family and font-size
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('font', size=12)
plt.rc('axes', titlesize=20)    # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('figure', titlesize=20)  # fontsize of the figure title

# Plot samples
if samplesU['total']:
    xx = np.linspace(-6, 6, 300)
    [X, Y] = np.meshgrid(xx, xx)
    xnod = np.vstack([X.T.ravel(), Y.T.ravel()]).T

    Z = g(xnod)
    Z = Z.reshape(X.shape).T

    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    ax.contour(X, Y, Z, [0], colors="r", linewidths=3)  # LSF
    for sample in samplesU['total']:
        ax.plot(*sample.T, ".", markersize=2)

# Plot failure probability: Exact
plt.figure()
plt.yscale('log')
plt.title('Failure probability estimate')
plt.xlabel('Limit state function, $g$')
plt.ylabel('Failure probability, $P_f$')

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
