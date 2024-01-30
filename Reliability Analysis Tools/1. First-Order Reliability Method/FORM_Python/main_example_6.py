try:
    import autograd.numpy as np
except:
    import numpy as np

import matplotlib.pyplot as plt
from ERANataf import ERANataf
from ERADist import ERADist
from FORM_HLRF import FORM_HLRF
from FORM_fmincon import FORM_fmincon
from FORM_Sensitivity import FORM_Sensitivity
plt.close('all')
"""
---------------------------------------------------------------------------
FORM using HLRF algorithm and fmincon: Ex. 5 Ref. 3 - points outside of hypersphere
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

#  limit state function
r = 5.26
m = 1


'''
def g(u:np.ndarray)->np.ndarray:
    # Reshape the array u
    u_st = np.reshape(u,(-1,d))

    # Generate output array
    g_out = np.zeros((u_st.shape[0],1))

    # Perform loop
    for ii in range(u_st.shape[0]):
        g_out[ii] = 1 - (np.sqrt(np.sum(u_st[ii,:]**2))/r)**2 -(u_st[ii,0]/r)*((1-(np.sqrt(np.sum(u_st[ii,:]**2))/r)**m)/(1+(np.sqrt(np.sum(u_st[ii,:]**2))/r)**m))
    return np.ravel(g_out)
'''

g  = lambda u: 1 - (np.sqrt(np.sum(u**2))/r)**2 - (u[0]/r)*((1-(np.sqrt(np.sum(u**2))/r)**m)/(1+(np.sqrt(np.sum(u**2))/r)**m))
dg = lambda u: np.array([(((u[0]**2 + u[1]**2)**(1/2)/r)**m - 1)/(r*(((u[0]**2 + u[1]**2)**(1/2)/r)**m + 1)) - (2*u[0])/r**2 + (m*u[0]**2*((u[0]**2 + u[1]**2)**(1/2)/r)**(m - 1))/(r**2*(u[0]**2 + u[1]**2)**(1/2)*(((u[0]**2 + u[1]**2)**(1/2)/r)**m + 1)) - (m*u[0]**2*((u[0]**2 + u[1]**2)**(1/2)/r)**(m - 1)*(((u[0]**2 + u[1]**2)**(1/2)/r)**m - 1))/(r**2*(u[0]**2 + u[1]**2)**(1/2)*(((u[0]**2 + u[1]**2)**(1/2)/r)**m + 1)**2), \
                          (m*u[0]*u[1]*((u[0]**2 + u[1]**2)**(1/2)/r)**(m - 1))/(r**2*(u[0]**2 + u[1]**2)**(1/2)*(((u[0]**2 + u[1]**2)**(1/2)/r)**m + 1)) - (2*u[1])/r**2 - (m*u[0]*u[1]*((u[0]**2 + u[1]**2)**(1/2)/r)**(m - 1)*(((u[0]**2 + u[1]**2)**(1/2)/r)**m - 1))/(r**2*(u[0]**2 + u[1]**2)**(1/2)*(((u[0]**2 + u[1]**2)**(1/2)/r)**m + 1)**2) ])

## OPC 1. FORM using Hasofer-Lind-Rackwitz-Fiessler algorithm HLRF (Ref.1 Pag.128)
[u_star_hlrf, x_star_hlrf, beta_hlrf, alpha_hlrf, Pf_hlrf] = FORM_HLRF(g, [], pi_pdf)

# OPC 2. FORM using Python scipy.optimize.minimize()
#[u_star_fmc, x_star_fmc, beta_fmc, alpha_fmc, Pf_fmc] = FORM_fmincon(g, [], pi_pdf)

# OPC 3. FORM using Python scipy.optimize.minimize() with analytical gradients
[u_star_fmc, x_star_fmc, beta_fmc, alpha_fmc, Pf_fmc] = FORM_fmincon(g, [], pi_pdf)


# Implementation of sensitivity analysis

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
S_F1_ref   = [0.1857, 0.1857]

# Print reference values for the first order indices
print("\n***Reference first order Sobol' indices: ***\n", S_F1_ref)

# reference solution
pf_ref = 1e-6

# show p_f results
print('***Reference Pf: ', pf_ref, ' ***')
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
    xx      = np.linspace(-6,6,100)
    [X1,X2] = np.meshgrid(xx,xx)
    xnod = np.array([np.reshape(X1,len(xx)**2),np.reshape(X2,len(xx)**2)])
    ZX = np.array([g(xnod[:,i]) for i in range(len(xx)**2)])
    ZX = np.reshape(ZX,[len(xx),len(xx)])

    # figure
    plt.figure()
    plt.axes().set_aspect('equal', 'box')
    plt.pcolor(X1,X2,ZX)
    plt.contour(X1,X2,ZX,[0])
    plt.plot(0,0,'k.')                                               # origin in standard
    plt.plot([0, u_star_hlrf[0]],[0, u_star_hlrf[1]],label='HLRF')   # reliability index beta
    plt.plot(u_star_hlrf[0],u_star_hlrf[1],'r*')                     # design point in standard
    plt.plot([0, u_star_fmc[0]],[0, u_star_fmc[1]],label='SLSQP')    # reliability index beta
    plt.plot(u_star_fmc[0],u_star_fmc[1],'bo')                       # design point in standard
    plt.xlabel(r'$u_1$')
    plt.ylabel(r'$u_2$', rotation=0)
    plt.title('Standard space')
    plt.legend()
    plt.show()

# END
