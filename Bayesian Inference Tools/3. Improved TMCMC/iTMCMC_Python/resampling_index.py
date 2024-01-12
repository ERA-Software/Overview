import numpy as np
import scipy as sp
"""
---------------------------------------------------------------------------
Basic resampling algorithm
---------------------------------------------------------------------------
Created by:
Matthias Willer (matthias.willer@tum.de)    
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Input:
* w : unnormalized plausability weights
---------------------------------------------------------------------------
Output:
* idx : index with the highest probability
---------------------------------------------------------------------------
"""

def resampling_index(w):
    N    = len(w)
    csum = np.cumsum(w)
    ssum = sum(w)
    lim  = ssum*sp.stats.uniform.rvs()
    #
    i = 1
    while csum[i] <= lim:
        i = i+1
        if i > N:
            i   = 1
            lim = ssum*sp.stats.uniform.rvs()
    idx = i

    return idx
## END