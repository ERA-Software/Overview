"""
--------------------------------------------------------------------------
Multi-index sequence for the computation of M-dimensional polynomials
--------------------------------------------------------------------------
Created by:
Fong-Lin Wu (fonglin.wu@tum.de)
Felipe Uribe (felipe.uribe@tum.de)
Technische Universitat Munchen
www.era.bgu.tum.de
--------------------------------------------------------------------------
Version: 
* 2019-08 Transition from Matlab
--------------------------------------------------------------------------
Input:
* M      : Number of K-L terms (number of random variables)
* p_order: Order of PC
--------------------------------------------------------------------------
Output:
* alpha  : Multi-index sequence
--------------------------------------------------------------------------
Based on:
1."Numerical methods for stochastic computations. A spectral method approach"
   D. Xiu. 2010. Princeton University Press.
2."Stochastic finite element methods and reliability"
   B. Sudret and A. Der Kiureghian. State of the art report.
--------------------------------------------------------------------------
"""

import numpy as np
import scipy as sp

def  multi_index(M,p):

   # procedure
   alpha     = []          # multi-index
   alpha.append( [0]*M )   # multi-index for p = 0

   if M==1: # dimension = 1
      for q in range(1,p+1):
         alpha.append( [q] )

   else:  # dimension>1
       for q in range(1,p+1):
           s          = [ sp.special.comb(x, M-1) for x in range(1,M+q) ]
           s1         = np.zeros((np.size(s,0),1))
           s2         = (M+q)+s1
           alpha.append( [ np.flipud(np.diff([s1, s, s2],1,2)) ] ) 
           if np.sum(alpha[q],2) != q*np.ones(sp.special.comb(M+q-1,M-1),1):
               raise NameError('The sum of each row has to be equal to q-th order')

   return np.array(alpha)