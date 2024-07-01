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
import numpy as np
import numpy.matlib
import scipy as sp
from scipy.stats import multivariate_normal as mvn
import matplotlib.pylab as plt

# Make sure ERADist, ERANataf classes are in the path
# https://www.bgu.tum.de/era/software/eradist/
from ERADist import ERADist
from ERANataf import ERANataf
from iTMCMC import iTMCMC
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
T_nataf = ERANataf(dist_X, R)

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
log_likelihood = lambda x: -J(x)/(2*var_eps)

#=================================================================
# iTMCMC
Ns = int(1e3)      # number of samples per level
Nb = int(0.1*Ns)   # burn-in period

print('\n\niTMCMC: ')
samplesU, samplesX, q, logcE = iTMCMC(Ns, Nb, log_likelihood, T_nataf)

#=================================================================
# reference solutions
mu_exact    = 1.12   # for x_1
sigma_exact = 0.66   # for x_1
cE_exact    = 1.52e-3
m           = len(q)   # number of stages (intermediate levels)

print('\nExact model evidence =', cE_exact)
print('Model evidence iTMCMC =', np.exp(logcE), '\n')
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

# plot p values
plt.figure()
plt.plot(np.arange(0,m),q,'ro-')
plt.xlabel(r'Intermediate levels $j$') 
plt.ylabel(r'$p_j$')
plt.tight_layout()
   
# plot samples increasing q
idx = np.array([0, int(np.round((m-1)/3)), int(np.round(2*(m-1)/3)), m-1])
plt.figure()
for i in range(4):
   plt.subplot(2,2,i+1) 
   plt.plot(samplesX[idx[i]][:,0],samplesX[idx[i]][:,1],'b.')
   plt.title(r'$p_j=' + str(np.round(q[idx[i]],5)) + r'$')
   plt.xlabel(r'$x_1$')
   plt.ylabel(r'$x_2$')
   plt.xlim([0, 3])
   plt.ylim([0, 2])
plt.tight_layout()

plt.show()