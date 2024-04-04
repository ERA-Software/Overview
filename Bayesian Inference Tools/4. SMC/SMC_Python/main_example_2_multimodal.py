import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats


## example 2: multi-modal posterior
"""
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
Contact: Antonis Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Version 2021-04
---------------------------------------------------------------------------
References:
1."Asymptotically independent Markov sampling: a new MCMC scheme for Bayesian inference"
   James L. Beck and Konstantin M. Zuev
   International Journal for Uncertainty Quantification, 3.5 (2013) 445-474.
2."Bayesian inference with subset simulation: strategies and improvements"
   Wolfgang Betz et al.
   Computer Methods in Applied Mechanics and Engineering 331 (2018) 72-93.
---------------------------------------------------------------------------
"""

# add path to ERADist and ERANataf classes
# Source: https://www.bgu.tum.de/era/software/eradist/
# addpath('../../../ERA_Dist/ERADistNataf_MATLAB/')

from ERADist import ERADist
from ERANataf import ERANataf
from SMC_aCS import SMC_aCS
from SMC_GM import SMC_GM

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
    return np.log(likelihood(theta) + np.finfo(float).tiny)

## SMC-SuS
N  = int(1e3)           # number of samples per level
p = 1                   # N/number of chains per level
burn = int(2)           # Burn-in length (per chain)
tarCoV = 1.5            # target coefficient of variation of the weights
k_init = int(2)

#method = "aCS"
method = "GM"

if method == "aCS":
    print('\nSequential Monte Carlo with aCS: \n\n')
    [samplesU, samplesX, q, logcE] = SMC_aCS(N, p, log_likelihood, prior_pdf, burn, tarCoV)
elif method == "GM":
    print('\nSequential Monte Carlo with GM: \n\n')
    [samplesU, samplesX, q, k_fin, logcE] = SMC_GM(N, p, log_likelihood, prior_pdf, k_init, burn, tarCoV)

## extract the samples
nsub = len(q)
u1p  = []
u2p  = []
x1p  = []
x2p  = []

for i in range(nsub):
    # samples in standard
    u1p.append(samplesU[i][0,:])
    u2p.append(samplesU[i][1,:])
    # samples in physical
    x1p.append(samplesX[i][0, :])
    x2p.append(samplesX[i][1, :])

# show results
print('\nModel evidence BUS-SuS = {:.5f}'.format(np.exp(logcE)))
print('\nMean value of x_1 = {:.5f}'.format(np.mean(x1p[-1])))
print('\tStd of x_1 = {:.5f}'.format(np.std(x1p[-1])))
print('\nMean value of x_2 = {:.5f}'.format(np.mean(x2p[-1])))
print('\tStd of x_2 = {:.5f}\n'.format(np.std(x2p[-1])))

## plot samples
nrows = int(np.ceil(np.sqrt(nsub)))
ncols = int(np.ceil(nsub/nrows))

fig1 = plt.figure(1)      # customize here your plots per row
for k in range(1, nsub+1):
    ax = fig1.add_subplot(nrows,ncols,k)
    ax.plot(u1p[k-1],u2p[k-1],'r.', markersize=3)
    ax.set_xlabel(r"$u_1$", fontsize=15)
    ax.set_ylabel(r"$u_2$", fontsize=15)
    ax.axis('equal')
fig1.suptitle("Standard space", fontsize=13, fontweight='bold')
plt.rcParams.update({'font.size': 13})
plt.tight_layout(rect=[0, 0.0, 1, 0.95])

fig2 = plt.figure(2)      # customize here your plots per row
for k in range(1, nsub+1):
    ax = fig2.add_subplot(nrows,ncols,k)
    ax.plot(x1p[k-1],x2p[k-1],'b.', markersize=3)
    ax.set_xlabel(r"$\theta_1$", fontsize=15)
    ax.set_ylabel(r"$\theta_2$", fontsize=15)
    ax.axis('equal')
fig2.suptitle("Original space", fontsize=13,fontweight='bold')
plt.rcParams.update({'font.size': 13})
plt.tight_layout(rect=[0, 0.0, 1, 0.95])

plt.show()
##END
