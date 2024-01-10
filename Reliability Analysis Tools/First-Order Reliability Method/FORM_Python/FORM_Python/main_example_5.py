import numpy as np
import scipy as sp
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
from FORM_HLRF import FORM_HLRF
from FORM_fmincon import FORM_fmincon
plt.close('all')
"""
---------------------------------------------------------------------------
FORM using HLRF algorithm and fmincon: Ex. 3 Ref. 3 - series system reliability problem
---------------------------------------------------------------------------
Created by:
Matthias Willer
Felipe Uribe
Luca Sardi
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2022-04
* Inclusion of sensitivity analysis
---------------------------------------------------------------------------
Based on:
1."Structural reliability under combined random load sequences."
   Rackwitz, R., and B. Fiessler (1979).
   Computers and Structures, 9.5, pp 489-494
2."Lecture Notes in Structural Reliability"
   Straub (2016)
3. "Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
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
g = lambda u: np.amin(np.array([
                    0.1*(u[0]-u[1])**2-(u[0]+u[1])/np.sqrt(2)+3 ,
                    0.1*(u[0]-u[1])**2+(u[0]+u[1])/np.sqrt(2)+3 ,
                    u[0]-u[1]+7/np.sqrt(2) ,
                    u[1]-u[0]+7/np.sqrt(2) ]), axis=0)


#  Implementation of sensitivity analysis: 1 - perform, 0 - not perform
sensitivity_analysis = 1

#  Solve the optimization problem of the First Order Reliability Method

# OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf, x_star_hlrf, beta_hlrf, Pf_hlrf, S_F1_hlrf, S_F1_T_hlrf] = FORM_HLRF(g, [], pi_pdf, sensitivity_analysis)

# OPC 2. FORM using Python scipy.optimize.minimize()
[u_star_fmc, x_star_fmc, beta_fmc, Pf_fmc, S_F1_fmc, S_F1_T_fmc] = FORM_fmincon(g, [], pi_pdf,sensitivity_analysis)

# OPC 3. FORM using Python scipy.optimize.minimize() with analytical gradients
#[u_star_fmc, x_star_fmc, beta_fmc, Pf_fmc, S_F1_fmc, S_F1_T_fmc] = FORM_fmincon(g, dg, pi_pdf,sensitivity_analysis)

# Reference values
# The reference values for the first order indices
S_F1_ref   = [0.0481, 0.0481]

# Print reference values for the first order indices
print("\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# reference solution
pf_ref = 2.26e-3

# show p_f results
print('***Reference Pf: ', pf_ref, ' ***')
print('***FORM HLRF Pf: ', Pf_hlrf, ' ***')
print('***FORM fmincon Pf: ', Pf_fmc, ' ***\n')

#  Plot
if d == 2:
    # grid points
    xx      = np.linspace(-7,7,100)
    [X1,X2] = np.meshgrid(xx,xx)
    xnod    = np.array([X1,X2])
    ZX      = g(xnod)

    # figure
    plt.figure()
    plt.axes().set_aspect('equal', 'box')
    plt.pcolor(X1,X2,ZX)
    plt.contour(X1,X2,ZX,[0])
    plt.plot(0,0,'k.')                                               # origin in standard
    plt.plot([0, x_star_hlrf[0]],[0, x_star_hlrf[1]],label='HLRF')   # reliability index beta
    plt.plot(x_star_hlrf[0],x_star_hlrf[1],'r*')                     # design point in standard
    plt.plot([0, u_star_fmc[0]],[0, u_star_fmc[1]],label='SLSQP')    # reliability index beta
    plt.plot(u_star_fmc[0],u_star_fmc[1],'bo')                       # design point in standard
    plt.title('Standard space')
    plt.legend()
    plt.show()

# END
