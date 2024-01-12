## Nonlinear bi-dimensional limit-state function
"""
---------------------------------------------------------------------------
Created by:
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Muenchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2021-05
---------------------------------------------------------------------------
Based on:
1."Combination line sampling for structural reliability analysis"
   Iason Papaioannou & Daniel Straub.
   Structural Safety 88 (2021) 102025.
---------------------------------------------------------------------------
"""
import numpy as np
import scipy as sp
from scipy import stats

from Distributions.ERANataf import ERANataf
from Distributions.ERADist import ERADist
from CLS import CLS


## Fix Seed with True or deactivate with False
fix_seed = False
if fix_seed:
    np.random.seed(2021)

## definition of the random variables
d      = 6          # number of dimensions
pi_pdf = [ERADist('lognormal','MOM', [120, 8]),
          ERADist('lognormal','MOM', [120, 8]),
          ERADist('lognormal','MOM', [120, 8]),
          ERADist('lognormal','MOM', [120, 8]),
          ERADist('lognormal','MOM', [50, 10]),
          ERADist('lognormal','MOM', [40, 8])]

## correlation matrix
R = np.identity(d)      # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)

## limit state function
g = lambda x: x[:,0] + 2*x[:,1] + 2*x[:,2] + x[:,3] - 5*x[:,4] - 5*x[:,5] + 0.1*np.sum(np.sin(100*x[:,0:4]), 1)

## line sampling
N  = 100        # Total number of samples
u_star = np.array([[0.35, 0.24, 0.24, 0.35, 0.63, 0.50]])

print('LINE SAMPLING: \n')
[Pf_LS, delta_LS, u_new, x_new, n_eval] = CLS(u_star, N, g, pi_pdf)

# reference solution
pf_ref = 5.29e-4

# show p_f results
print('\n***Reference Pf: {:.6f} ***'.format(pf_ref))
print('\n***CLS Pf: {:.8f} ***\n\n'.format(float(Pf_LS)))
