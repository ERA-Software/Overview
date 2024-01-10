import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from SuS import SuS
plt.close('all')
"""
---------------------------------------------------------------------------
Subset Simulation: Ex. 1 Ref. 2 - linear function of independent standard normal
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
2."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
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
beta = 3.5
g = lambda x: -np.sum(x, axis=1) / np.sqrt(d) + beta

# %% subset simulation
# N  = 3000        # Total number of samples for each level
N  = 100000        # Total number of samples for each level
p0 = 0.1         # Probability of each subset, chosen adaptively

# Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 1

# Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 1

print('\n\nSUBSET SIMULATION: ')
[Pf_SuS, delta_SuS, b, Pf, b_sus, pf_sus, samplesU, samplesX, S_F1] = SuS(N, p0, g, pi_pdf, sensitivity_analysis, samples_return)

# Reference values
# The reference values for the first order indices
S_F1_ref   = [0.0315, 0.0315]

# Print reference values for the first order indices
print("\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# exact solution
pf_ex    = sp.stats.norm.cdf(-beta)
Pf_exact = lambda gg: sp.stats.norm.cdf(gg,beta,1)
gg       = np.linspace(0,7,140)

# show p_f results
print('\n***Exact Pf: ', pf_ex, ' ***')
print('***SuS Pf: ', Pf_SuS, ' ***\n')

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
    if d == 2:
        nnp = 200
        xx = np.linspace(-6, 6, nnp)
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
plt.plot(gg,Pf_exact(gg),'b-', label='Exact')
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
plt.plot(0,pf_ex,'ro', label='Pf Exact',
                       markersize=10,
                       markerfacecolor='none')
plt.legend()
plt.tight_layout()
plt.show()
# %%END