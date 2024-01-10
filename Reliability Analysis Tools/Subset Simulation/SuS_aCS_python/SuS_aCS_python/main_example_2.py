import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from SuS import SuS
plt.close('all')
"""
---------------------------------------------------------------------------
Subset Simulation: Ex. 2 Ref. 2 - linear function of independent exponential
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
#d      = 100         # number of dimensions
d      = 2         # number of dimensions
pi_pdf = list()
for i in range(d):
    pi_pdf.append(ERADist('exponential', 'PAR',1)) # n independent rv

# correlation matrix
# R = eye(d)   # independent case

# object with distribution information
# pi_pdf = ERANataf(pi_pdf,R)    # if you want to include dependence

# %% limit-state function
#Ca = 140
Ca = 10
g = lambda x: Ca - np.sum(x, axis=1)

# %% Subset simulation
N  = 2000         # Total number of samples for each level
p0 = 0.1          # Probability of each subset, chosen adaptively

# Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 1

# Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 2

print('\n\nSUBSET SIMULATION: ')
[Pf_SuS, delta_SuS, b, Pf, b_sus, pf_sus, samplesU, samplesX, S_F1] = SuS(N, p0, g, pi_pdf, sensitivity_analysis, samples_return)

# Reference values
# The reference values for the first order indices
S_F1_ref   = [0.2021, 0.1891]

# Print reference values for the first order indices
print("\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# exact solution
lam      = 1
pf_ex    = 1 - sp.stats.gamma.cdf(Ca, a=d)
Pf_exact = lambda gg: 1-sp.stats.gamma.cdf(Ca-gg, a=d)
gg       = np.linspace(0,30,300)

# show p_f results
print('\n***Exact Pf: ', pf_ex, ' ***')
print('***SuS Pf: ', Pf_SuS, ' ***\n')

# %% Plots
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
