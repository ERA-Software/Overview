import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ERANataf import ERANataf
from ERADist import ERADist
from FORM_HLRF import FORM_HLRF
from FORM_fmincon import FORM_fmincon
from FORM_Sensitivity import FORM_Sensitivity

plt.close('all')
"""
---------------------------------------------------------------------------
FORM using HLRF algorithm and fmincon: Ex. 1 Ref. 3 - linear function of independent standard normal
---------------------------------------------------------------------------
Created by:
Matthias Willer
Felipe Uribe
Luca Sardi
Daniel Koutas
Ivan Olarte-Rodriguez

Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2023-10
* Modification of Sensitivity Analysis Calls
---------------------------------------------------------------------------
Based on:
1."Structural reliability under combined random load sequences."
   Rackwitz, R., and B. Fiessler (1979).
   Computers and Structures, 9.5, pp 489-494
2."Lecture Notes in Structural Reliability"
   Straub (2016)
---------------------------------------------------------------------------
"""

#  definition of the random variables
d      = 2          # number of dimensions
pi_pdf = list()
for i in range(d):
    pi_pdf.append(ERADist('standardnormal', 'PAR', np.nan)) # n independent rv

# correlation matrix
R = np.eye(d)   # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)    # if you want to include dependence

#  limit state function and its gradient in the original space
beta = 3.5
g    = lambda x: -np.sum(x, axis=0)/np.sqrt(d) + beta
dg   = lambda x: np.tile(-1/np.sqrt(d),[d,1])

##  Solve the optimization problem of the First Order Reliability Method

# OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf, x_star_hlrf, beta_hlrf, alpha_hlrf, Pf_hlrf] = FORM_HLRF(g, dg, pi_pdf)

# OPC 2. FORM using Python scipy.optimize.minimize()
[u_star_fmc, x_star_fmc, beta_fmc, alpha_fmc, Pf_fmc] = FORM_fmincon(g, [], pi_pdf)

# OPC 3. FORM using Python scipy.optimize.minimize() with analytical gradients
#[u_star_fmc, x_star_fmc, beta_fmc, alpha_fmc, Pf_fmc] = FORM_fmincon(g, dg, pi_pdf)


## Implementation of sensitivity analysis

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


# Reference values
# The reference values for the first order indices
S_F1_ref   = [0.0315, 0.0315]

# Print reference values for the first order indices
print("\n\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# exact solution
pf_ex = sp.stats.norm.cdf(-beta)

# show p_f results
print('\n***Exact Pf: ', pf_ex, ' ***')
print('***FORM HLRF Pf: ', Pf_hlrf, ' ***')
print('***FORM fmincon Pf: ', Pf_fmc, ' ***\n')

# %% Plots
# Options for font-family and font-size
plt.rc('font', size=12)
plt.rc('axes', titlesize=20)    # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('figure', titlesize=20)  # fontsize of the figure title

#  Plot HLRF results
if d == 2:
    # grid points
    xx      = np.linspace(0,5,100)
    [X1,X2] = np.meshgrid(xx,xx)
    xnod    = np.array([X1,X2])
    ZX      = g(xnod)

    # figure
    plt.figure()
    plt.axes().set_aspect('equal','box')
    plt.pcolor(X1,X2,ZX)
    plt.contour(X1,X2,ZX,[0])
    plt.plot(0,0,'k.')                                               # origin in standard
    plt.plot([0, x_star_hlrf[0]],[0, x_star_hlrf[1]],label='HLRF')   # reliability index beta
    plt.plot(x_star_hlrf[0],x_star_hlrf[1],'r*')                     # design point in standard
    plt.plot([0, u_star_fmc[0]],[0, u_star_fmc[1]],label='SLSQP')    # reliability index beta
    plt.plot(u_star_fmc[0],u_star_fmc[1],'bo')                       # design point in standard
    plt.xlabel(r'$u_1$')
    plt.ylabel(r'$u_2$', rotation=0)
    plt.title('Standard space')
    plt.legend()
    plt.show()
# END
