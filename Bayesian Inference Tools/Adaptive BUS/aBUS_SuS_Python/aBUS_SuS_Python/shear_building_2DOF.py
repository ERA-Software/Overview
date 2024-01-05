import numpy as np

"""
---------------------------------------------------------------------------
two-degree-of-freedom shear building model
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-03
---------------------------------------------------------------------------
EXAMPLE:
m1 = 16.531e3;   m2 = 16.131e3;
k1 = 29.7e6;     k2 = 29.7e6;
[w,Phi] = shear_building_2DOF(m1,m2,k1,k2)
---------------------------------------------------------------------------
References:
1."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2014) 1-13.
---------------------------------------------------------------------------
"""

def shear_building_2DOF(m1, m2, k1, k2):
    # mass and stiffness matrices
    M = np.array([ [m1, 0], [0, m2] ])          # mass matrix
    K = np.array([ [k1+k2, -k2], [-k2, k2] ])   # stiffness matrix

    # free response - Modal analysis 
    L, _ = np.linalg.eig(np.linalg.solve(M,K))   # eigenvalue solution
    w    = np.sqrt(L)/(2*np.pi)                  # natural frequencies [Hz]
    idx  = np.argsort(w)                         # ordering (ascendent)
    w    = w[idx]
    # Phi = V[idx]                                # vibration modes

    # normalizing modes
    # q   = np. multiply(np.multiply(Phi.T,M),Phi)    # orthogonality property
    # Phi = Phi/np.tile(np.sqrt(np.diag(q)).T,(2,1))

    return w