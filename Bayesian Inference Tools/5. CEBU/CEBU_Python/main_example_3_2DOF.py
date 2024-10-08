"""
---------------------------------------------------------------------------
Example 3: Parameter identification two-DOF shear building
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu
Matthias Willer   
Felipe Uribe
Luca Sardi

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Last Version 2021-03
* Adaptation to new ERANataf class
---------------------------------------------------------------------------
References:
1."Transitional Markov Chain Monte Carlo: Observations and improvements". 
   Wolfgang Betz et al.
   Journal of Engineering Mechanics. 142.5 (2016) 04016016.1-10
2."Transitional Markov Chain Monte Carlo method for Bayesian model updating, 
   model class selection and model averaging". 
   Jianye Ching & Yi-Chun Chen
   Journal of Engineering Mechanics. 133.7 (2007) 816-832
---------------------------------------------------------------------------
"""

#=================================================================
# import libraries
import sys
import numpy as np
import scipy as sp
import matplotlib.pylab as plt

# Make sure ERADist, ERANataf classes are in the path
# https://www.bgu.tum.de/era/software/eradist/
# Import In House Functions
from CEBU_GM import CEBU_GM
from CEBU_vMFNM import CEBU_vMFNM
from ERADist import ERADist
from ERANataf import ERANataf
from shear_building_2DOF import shear_building_2DOF
plt.close('all')

#=================================================================
# shear building data
m1 = 16.5e3     # mass 1st story [kg]
m2 = 16.1e3     # mass 2nd story [kg]
kn = 29.7e6     # nominal values for the interstory stiffnesses [N/m]

# prior PDF for X1 and X2 (product of lognormals)
mod_log_X1 = 1.3   # mode of the lognormal 1
std_log_X1 = 1.0   # std of the lognormal 1
mod_log_X2 = 0.8   # mode of the lognormal 2
std_log_X2 = 1.0   # std of the lognormal 2

# find lognormal X1 parameters
var_fun = lambda mu: std_log_X1**2 - (np.exp(mu-np.log(mod_log_X1))-1) \
                                     *np.exp(2*mu+(mu-np.log(mod_log_X1)))
mu_X1  = sp.optimize.fsolve(var_fun,1)      # mean of the associated Gaussian
std_X1 = np.sqrt(mu_X1-np.log(mod_log_X1))  # std of the associated Gaussian

# find lognormal X2 parameters
var_X2 = lambda mu: std_log_X2**2 - (np.exp(mu-np.log(mod_log_X2))-1) \
                                    *np.exp(2*mu+(mu-np.log(mod_log_X2)))
mu_X2  = sp.optimize.fsolve(var_X2,0)        # mean of the associated Gaussian
std_X2 = np.sqrt(mu_X2-np.log(mod_log_X2))   # std of the associated Gaussian

#=================================================================
# definition of the random variables
n       = 2   # number of random variables (dimensions)
dist_x1 = ERADist('lognormal','PAR',[mu_X1, std_X1])
dist_x2 = ERADist('lognormal','PAR',[mu_X2, std_X2])

# distributions
dist_X = [dist_x1, dist_x2]

# correlation matrix
R = np.eye(n)   # independent case

# object with distribution information
prior_pdf = ERANataf(dist_X, R)

#=================================================================
# likelihood function
lam     = np.array([1, 1])   # means of the prediction error
i       = 9                  # simulation level
var_eps = 0.5**(i-1)         # variance of the prediction error
f_tilde = np.array([3.13, 9.83])       # measured eigenfrequencies [Hz]

# shear building model 
f = lambda x: shear_building_2DOF(m1, m2, kn*x[0], kn*x[1])

# modal measure-of-fit function
J = lambda x: np.sum((lam**2)*(((f(x)**2)/f_tilde**2) - 1)**2)   

# likelihood function
likelihood     = lambda x: np.exp(-J(x)/(2*var_eps))

def log_likelihood(theta):
    llvec = np.zeros(theta.shape[0])
    for k in range(theta.shape[0]):
        llvec[k] = -J(theta[k,:])/(2*var_eps) + np.finfo(float).tiny #np.log(likelihood(theta[k,:]) + np.finfo(float).tiny)
    return llvec

#=================================================================
N = int(3e3)
Nlast = N
max_steps = 100
tarCoV = 1.5
k_init = 5


method = "vMFNM"
#method = "GM"

if method == "GM":
    samplesU, samplesX, v_tot, beta_tot, k_final, evidence, Wlast_normed, f_s_iid = CEBU_GM(N, log_likelihood, prior_pdf, max_steps, tarCoV, k_init, 2, Nlast)
elif method == "vMFNM":
    samplesU, samplesX, v_tot, beta_tot, k_final, evidence, Wlast_normed, f_s_iid = CEBU_vMFNM(N, log_likelihood, prior_pdf, max_steps, tarCoV, k_init, 2, Nlast)

## extract the samples
nsub = len(samplesU) 
if nsub == 0: 
    print("No samples returned, hence no visualization and reference solutions.")
    sys.exit()

u1p  = []
u2p  = []
x1p  = []
x2p  = []

for i in range(nsub):
    # samples in standard
    u1p.append(samplesU[i][:,0])
    u2p.append(samplesU[i][:,1])
    # samples in physical
    x1p.append(samplesX[i][:,0])
    x2p.append(samplesX[i][:,1])

#=================================================================
# reference solutions
mu_exact    = 1.12   # for x_1
sigma_exact = 0.66   # for x_1
cE_exact    = 1.52e-3

print('\nExact model evidence =', cE_exact)
print('Model evidence CEBU =', evidence, '\n')
print('Exact posterior mean x_1 =', mu_exact)
print('Mean value of x_1 =', np.mean(samplesX[-1][:,0]), '\n')
print('Exact posterior std x_1 = ', sigma_exact)
print('Std of x_1 =',np.std(samplesX[-1][:,0]),'\n')

#=================================================================
# Plots
# Options for font-family and font-size
plt.rc('font', size=12)
plt.rc('axes', titlesize=20)    # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('figure', titlesize=20)  # fontsize of the figure title

# plot intermediate levels
if nsub == beta_tot.size: 
    plt.figure()
    plt.plot(np.arange(0,nsub),beta_tot,'ro-')
    plt.xlabel(r'Intermediate levels $j$') 
    plt.ylabel(r'$\beta_j$')
    plt.tight_layout()
   
# plot samples
nrows = int(np.ceil(np.sqrt(nsub)))
ncols = int(np.ceil(nsub/nrows))

fig1 = plt.figure(figsize=(10,6))      # customize here your plots per row
for k in range(1, nsub+1):
    ax = fig1.add_subplot(nrows,ncols,k)
    ax.plot(u1p[k-1],u2p[k-1],'r.', markersize=3)
    ax.set_xlabel(r"$u_1$", fontsize=15)
    ax.set_ylabel(r"$u_2$", fontsize=15)
    ax.axis('equal')
    ax.title.set_text(f'Step {k}')
fig1.suptitle("Standard space", fontsize=13, fontweight='bold')
plt.rcParams.update({'font.size': 13})
plt.tight_layout(rect=[0, 0.0, 1, 0.95])

fig2 = plt.figure(figsize=(10,6))      # customize here your plots per row
for k in range(1, nsub+1):
    ax = fig2.add_subplot(nrows,ncols,k)
    ax.plot(x1p[k-1],x2p[k-1],'b.', markersize=3)
    ax.set_xlabel(r"$\theta_1$", fontsize=15)
    ax.set_ylabel(r"$\theta_2$", fontsize=15)
    ax.axis('equal')
    ax.title.set_text(f'Step {k}')
fig2.suptitle("Original space", fontsize=13,fontweight='bold')
plt.rcParams.update({'font.size': 13})
plt.tight_layout(rect=[0, 0.0, 1, 0.95])

plt.show()
