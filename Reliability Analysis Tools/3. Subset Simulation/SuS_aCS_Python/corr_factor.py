import numpy as np
"""
---------------------------------------------------------------------------
Computation of the correlation factor Ref. 1 Eq. 20
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
Input:
* I_Fj : matrix with the values of the indicator function in Ref. 1 Eq. 22
* p_j  : conditional probability of the level
* Ns   : number of samples simulated from each Markov chain (N/Nc)
* Nc   : number of Markov chains (seeds)
---------------------------------------------------------------------------
Output:
* gamma : correlation factor in Ref. 1 Eq. 20
---------------------------------------------------------------------------
Based on:
1."Bayesian post-processor and other enhancements of subset simulation for
   estimating failure probabities in high dimensions"
   Konstantin M. Zuev, James L. Beck, Siu-Kui Au & Lambros S. Katafygiotis.
   Computers and Structures 92-93 (2012) 283-296.
---------------------------------------------------------------------------
"""

def corr_factor(I_Fj, p_j, Ns, Nc):
    ## Initialize variables
    gamma = np.zeros(Ns-1)   # correlation factor
    R     = np.zeros(Ns-1)   # autocovariance sequence
    N     = Nc*Ns           # total number of samples

    ## correlation at lag 0
    sums = 0
    for k in range(Nc):
        for ip in range(Ns):
            sums = sums + (I_Fj[ip,k]*I_Fj[ip,k])   # sums inside (Ref. 1 Eq. 22)

    R_0 = (1/N)*sums - p_j**2   # autocovariance at lag 0 (Ref. 1 Eq. 22)

    ## correlation factor calculation
    for i in range(Ns-1):
        sums = 0
        for k in range(Nc):
            for ip in range(Ns-i-1):
                sums = sums + (I_Fj[ip,k]*I_Fj[ip+i+1,k])  # sums inside (Ref. 1 Eq. 22)

        R[i]     = (1/(N-(i+1)*Nc))*sums - p_j**2   # autocovariance at lag i (Ref. 1 Eq. 22)
        gamma[i] = (1-((i+1)/Ns))*(R[i]/R_0)        # correlation factor (Ref. 1 Eq. 20)

    gamma = 2*np.sum(gamma)   # small values --> higher efficiency  (Ref. 1 Eq. 20)

    return gamma
##END
