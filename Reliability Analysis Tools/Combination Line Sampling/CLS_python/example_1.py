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
Current version 2021-06
---------------------------------------------------------------------------
Based on:
1."Advanced Line Sampling for efficient robust reliability analysis"
   Marco de Angelis, Edoardo Patelli & Michael Beer.
   Structural Safety 52 (2015) 170–182.
---------------------------------------------------------------------------
"""
import numpy as np
import scipy as sp
from scipy import stats

from ERANataf import ERANataf
from ERADist import ERADist
from CLS import CLS


## Fix Seed with True or deactivate with False
fix_seed = False
if fix_seed:
    np.random.seed(2021)

## definition of the random variables
d      = 2          # number of dimensions
pi_pdf = [ERADist('normal','MOM', [5,2]),
          ERADist('normal','MOM', [2,2])]

## correlation matrix
R = np.identity(d)      # independent case

# object with distribution information
pi_pdf = ERANataf(pi_pdf, R)

## limit state function
# you can manually choose between examples 0 to 4
example = 0

if round(example) < 0 or round(example) > 4:
    raise RuntimeError("Selected example is not available. Please choose an integer between 0 and 4")

# vector of constants to choose from
a_vec = np.array([10, 10.2, 10.5, 12, 14])
# constant for the limit-state function
a = a_vec[int(round(example))]
# limit state function
g = lambda x: -np.sqrt(x[:,0]**2 + x[:,1]**2) + a

## line sampling
N  = 100        # Total number of samples
u_star = np.array([[0.5, 1]])

print('LINE SAMPLING: \n')
[Pf_LS, delta_LS, u_new, x_new, n_eval] = CLS(u_star, N, g, pi_pdf)

# vector of reference solutions
pf_ref_vec = np.array([1.49e-2, 1.16e-2, 7.40e-3, 7.06e-4, 1.42e-5])

# reference solution
pf_ref = pf_ref_vec[int(round(example))]

# show p_f results
print('\n***Reference Pf: {:.8f} ***'.format(pf_ref))
print('\n***CLS Pf: {:.8f} ***\n\n'.format(float(Pf_LS)))
