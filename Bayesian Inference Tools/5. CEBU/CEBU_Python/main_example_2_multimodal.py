"""
---------------------------------------------------------------------------
Example 2: multi-modal posterior
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu   
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
2."Asymptotically independent Markov sampling: a new MCMC scheme for Bayesian inference"
   James L. Beck and Konstantin M. Zuev
   International Journal for Uncertainty Quantification, 3.5 (2013) 445-474.
---------------------------------------------------------------------------
"""
import sys
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

# Import In House Functions
from CEBU_GM import CEBU_GM
from CEBU_vMFNM import CEBU_vMFNM
from ERADist import ERADist
from ERANataf import ERANataf

# initial data
d = 2       # number of dimensions (number of uncertain parameters)

## prior PDF for theta (bivariate uniform)
a     = 10
bound = np.tile([0, a], [d, 1])   # boundaries of the uniform distribution box

## define the prior
prior_pdf = []
prior_pdf.append(ERADist('uniform','PAR',[bound[0,0], bound[0,1]]))
prior_pdf.append(ERADist('uniform','PAR',[bound[1,0], bound[1,1]]))

# correlation matrix
R = np.identity(d)          # independent case

# object with distribution information
prior_pdf = ERANataf(prior_pdf, R)

## likelihood PDF (mixture of gaussians in 2D)
np.random.seed(2021)            # fixed the seed to get the same data points

M     = 10                          # number of mixture
mu    = prior_pdf.random(M)         # Here using ERDist.random
sigma = 0.1
Cov   = (sigma**2)*np.identity(d)
w     = 0.1*np.ones((M,1))


def likelihood(theta):
    array = np.zeros((M,2))
    for k in range(len(mu)):
        array[k,:] = stats.multivariate_normal.pdf(theta, mu[k,:], Cov)
    return np.sum(np.multiply(w, array))

def log_likelihood(theta):
    llvec = np.zeros(theta.shape[0])
    for k in range(theta.shape[0]):
        llvec[k] = np.log(likelihood(theta[k,:]) + np.finfo(float).tiny)
    return llvec

N = int(3e3)
Nlast = N
max_steps = 100
tarCoV = 1.5
k_init = 2

method = "vMFNM"

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

# show results
print(f'\nModel evidence CEBU = {evidence:.5f}')
print(f'\nMean value of x_1 = {np.mean(x1p[-1]):.5f}')
print(f'\tStd of x_1 = {np.std(x1p[-1]):.5f}')
print(f'\nMean value of x_2 = {np.mean(x2p[-1]):.5f}')
print(f'\tStd of x_2 = {np.std(x2p[-1]):.5f}\n')

## plot samples
nrows = int(np.ceil(np.sqrt(nsub)))
ncols = int(np.ceil(nsub/nrows))

fig1 = plt.figure(1, figsize=(10,6))      # customize here your plots per row
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

fig2 = plt.figure(2, figsize=(10,6))      # customize here your plots per row
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
##END
