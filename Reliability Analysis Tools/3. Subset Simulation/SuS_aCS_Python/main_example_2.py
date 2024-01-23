import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from SuS import SuS
from Sim_Sensitivity import Sim_Sensitivity
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
---------------------------------------------------------------------------
"""

# %% definition of the random variables
#d      = 100         # number of dimensions
d      = 2         # number of dimensions
pi_pdf = list()
for i in range(d):
    pi_pdf.append(ERADist('exponential', 'PAR',1)) # n independent rv

# correlation matrix
R = np.eye(d)   # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf,R)    # if you want to include dependence

# %% limit-state function
#Ca = 140
Ca = 10
g = lambda x: Ca - np.sum(x, axis=1)

# %% Subset simulation
N  = 2000         # Total number of samples for each level
p0 = 0.1          # Probability of each subset, chosen adaptively

# Samples return: 0 - none, 1 - final sample, 2 - all samples
samples_return = 2

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
S_F1_ref   = [0.2021, 0.1891]

# Print reference values for the first order indices
print("\n\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# exact solution
lam      = 1
pf_ex    = 1 - sp.stats.gamma.cdf(Ca, a=d)
Pf_exact = lambda gg: 1-sp.stats.gamma.cdf(Ca-gg, a=d)
gg       = np.linspace(0,30,300)

# show p_f results
print('\n***Exact Pf: ', pf_ex, ' ***')
print('***SuS Pf: ', Pf_SuS, ' ***\n')

# %% Plots

# Options for font-family and font-size
plt.rc('font', size=12)
plt.rc('axes', titlesize=20)    # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('figure', titlesize=20)  # fontsize of the figure title

# Plot failure probability: Exact
plt.figure()
plt.yscale('log')
plt.plot(gg,Pf_exact(gg),'b-', label='Exact')
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
plt.plot(0,pf_ex,'ro', label='Pf Exact',
                       markersize=10,
                       markerfacecolor='none')
plt.legend()
plt.tight_layout()
plt.show()
# %%END
