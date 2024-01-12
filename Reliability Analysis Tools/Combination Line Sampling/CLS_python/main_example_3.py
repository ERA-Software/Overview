## Subset Simulation: Ex. 4 Ref. 3 - linear and convex limit state function
"""
---------------------------------------------------------------------------
Created by:
Iason Papaioannou
Daniel Koutas
Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Current version 2021-05
---------------------------------------------------------------------------
"""
import numpy as np

from Distributions.ERADist import ERADist
from Distributions.ERANataf import ERANataf
from CLS import CLS

## Fix Seed with True or deactivate with False
fix_seed = False
if fix_seed:
    np.random.seed(2021)

## definition of the random variables
d      = 2              # number of dimensions
pi_pdf = np.tile(ERADist('standardnormal','PAR'), [d])   # n independent rv

## correlation matrix
#R = np.identity(d)        # independent case
#
## object with distribution information
#pi_pdf = ERANataf(pi_pdf, R)      # if you want to include dependence

## limit state function
g = lambda x: 0.1*(x[:,0]-x[:,1])**2 - (x[:,0]+x[:,1])/np.sqrt(2) + 2.5

## line sampling
N  = 100        # Total number of samples
u_star = np.array([[0.5, 1]])

print('LINE SAMPLING: \n')
[Pf_LS, delta_LS, u_new, x_new, n_eval] = CLS(u_star, N, g, pi_pdf)

# reference solution
pf_ref = 4.21e-3

# show p_f results
print('\n***Reference Pf: {:.5f} ***'.format(pf_ref))
print('\n***CLS Pf: {:.8f} ***\n\n'.format(float(Pf_LS)))
