import numpy as np
from ERADist import ERADist
from ERANataf import ERANataf
from FORM_HLRF import FORM_HLRF
from FORM_fmincon import FORM_fmincon
from FORM_Sensitivity import FORM_Sensitivity

"""
---------------------------------------------------------------------------
FORM using fmincon: Ex. 1 Ref. 1 - steel column
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
First version: 2021-06
---------------------------------------------------------------------------
Based on:
1."Variance-based reliability sensitivity analysis and the FORM a-factors."
   Papaioannou & Straub
   Reliability Engineering and System Safety 210 (2021) 107496
---------------------------------------------------------------------------
"""

## Initialization
b   = 250
h   = 250
t_b = 15
t_h = 10
L   = 7.5e+3

A_s = 2*b*t_b + h*t_h
W_s = h*(t_h**3)/(6*b) + t_b*b**2/3
I_s = h*(t_h**3)/12 + t_b*(b**3)/6

## definition of the random variables
d       = 5          # number of dimensions
P_p     = ERADist('normal','MOM',[200e+3, 20e+3])
P_e     = ERADist('gumbel','MOM',[400e+3, 60e+3])
delta_0 = ERADist('normal','MOM',[30, 10])
f_y     = ERADist('lognormal','MOM',[400, 32])
E       = ERADist('lognormal','MOM',[2.1e+5, 8.4e+3])
pi_pdf  = [P_p, P_e, delta_0, f_y, E]


# correlation matrix
R = np.eye(d)      # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)        # if you want to include dependence

# limit state function in the original space
g  = lambda x:  1 - ((x[0]+x[1])/(x[3]*A_s) +
                (x[0]+x[1])*x[2]/(x[3]*W_s) *
                (np.pi**2)*x[4]*I_s/(L**2) / ((np.pi**2)*x[4]*I_s/L**2-(x[0]+x[1])))

#  Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 1

# OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
#[u_star_hlrf, x_star_hlrf, beta_hlrf, Pf_hlrf, S_F1_hlrf, S_F1_T_hlrf] = FORM_HLRF(g, [], pi_pdf,sensitivity_analysis)
[u_star_hlrf, x_star_hlrf, beta_hlrf, alpha_hlrf, Pf_hlrf] = FORM_HLRF(g, [], pi_pdf)

# OPC 2. FORM using Python scipy.optimize.minimize()
[u_star_fmc, x_star_fmc, beta_fmc, alpha_fmc, Pf_fmc] = FORM_fmincon(g, [], pi_pdf)

# Computation of Sobol Indices
compute_Sobol = True

# Computation of EVPPI (based on standard cost of failure (10^8) and cost
# of replacement (10^5)
compute_EVPPI = True

# using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[S_F1_hlrf, S_F1_T_hlrf, S_EVPPI] = FORM_Sensitivity(Pf_hlrf, pi_pdf, beta_hlrf, alpha_hlrf, 
                                                     compute_Sobol, compute_EVPPI)

# using MATLAB fmincon
[S_F1_fmc, S_F1_T_fmc, S_EVPPI] = FORM_Sensitivity(Pf_fmc, pi_pdf, beta_fmc, alpha_fmc, 
                                                   compute_Sobol, compute_EVPPI)


# MC solution given in paper
# The MC results for S_F1_MC have the following COVs in the given order:
# [16.1%, 0.2%, 1.8%, 7.4%, 15.1%]
# Hence the first order indices (except for the second one) have quite high
# uncertainty and should not be considered as exact.
S_F1_ref   = [1.7e-4, 0.1974, 0.0044, 4.8e-4, 1.6e-4]

# The MC results for the total-effect indices have all COVs <= 0.2% so they
# can be considered as more accurate than the first-order indices
S_F1_T_ref = [0.2365, 0.9896, 0.7354, 0.3595, 0.2145]

# MC probability of failure
Pf_ref = 8.35e-4

# Print reference values for the first order indices
print("\n***Reference first order Sobol' indices: ***\n", S_F1_ref)
print("\n***Reference total-effect indices: ***\n", S_F1_T_ref)

# show p_f results
print('***Exact Pf: ', Pf_ref, ' ***')
print('***FORM HLRF Pf: ', Pf_hlrf, ' ***')
print('***FORM fmincon Pf: ', Pf_fmc, ' ***\n')