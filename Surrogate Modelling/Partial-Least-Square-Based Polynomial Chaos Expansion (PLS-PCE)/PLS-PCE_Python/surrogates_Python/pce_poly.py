"""
--------------------------------------------------------------------------
Compute coefficients of full d_x-dimensional PCE of order p based on
OLS of (X,Y), where X ~ Gauss(0,1)
--------------------------------------------------------------------------
Created by:
Fong-Lin Wu (fonglin.wu@tum.de)
Max Ehre (max.ehre@tum.de)
Technische Universitat Munchen
www.era.bgu.tum.de
--------------------------------------------------------------------------
Version: 
* 2019-08 Transition from Matlab
--------------------------------------------------------------------------
Input:
* X    : Design input generated by Latin-Hypercube
* Y    : Result from Finite-Element-Method
* pmax : Maximum polynomial order of PCE-PLS 
--------------------------------------------------------------------------
Output:
* f    : Lamda function of PCE-PLS model
* Yhat : Result output by PCE-PLS model
* a    : Coefficient of Hermite polynomials
* Psi  : Products of Hermite polynomials
* Q_sq : 1 - (Leave-One-Out error)
--------------------------------------------------------------------------
Based on:

--------------------------------------------------------------------------
"""

import numpy as np
from scipy.special import factorial 

# Make sure having these scripts under the same workspace
from multi_index import multi_index # Multi-index sequence for the computation of polynomials

def pce_poly(X,Y,pmax):

    # read DoE dimensions
    [_, d_x]   = np.shape(X)
    [n_s, d_u] = np.shape(Y)

    # Hermite polynomials by recursive formula
    He_p    = []
    He_p.append( [1] )      # H_1 = 1
    He_p.append( [1, 0] )   # H_2 = x

    for n in range(1,pmax+2):
        # recursive formula
        He_p.append( (np.array( He_p[n] + [0] ) - (n)*np.array( [0]+[0]+He_p[n-1] )).tolist() ) 

    He_p_mat = np.zeros((pmax+2,pmax+2))

    for row in range(pmax+2):
        He_p_mat[row, :] =  [0]*(pmax+1-row) + He_p[row] 

    # multi-index
    alpha  = multi_index(d_x, pmax+1)
    P      = np.size(alpha, 0)
    
    # initialize list
    Psi_v  = []
    Psi_v  = np.zeros( (np.size(X,0),P,d_x) )

    for i in range(d_x):
        for index in range(P):
            Psi_v[:,index,i] =  np.polyval( He_p_mat[ alpha[index][i], : ], X[:,i] )

    norm = np.sqrt( np.prod( factorial(alpha), 1) )
    Psi  = np.divide ( np.prod(Psi_v,2), norm.T )

    # information matrix
    M = Psi.T @ Psi

    # inverse information matrix
    MI = np.linalg.pinv(M)

    # projection matrix
    PP = Psi @ MI @ Psi.T

    # annihilator matrix
    AA = np.eye(n_s) - PP

    # LOO-error approximation
    h       = np.diag(PP)
    eps_LOO = np.mean( np.power( np.divide( (AA@Y),(1-h).reshape(n_s,d_u) ), 2) ,0 ) / np.var(Y) * (1+np.trace(MI)) * n_s / (n_s-P)
    Q_sq    = 1 - eps_LOO
        
    # estimate the coefficient a
    a       =  np.linalg.lstsq( (Psi.T @ Psi), ( Psi.T @ Y ), rcond=None )[0]

    # build predictive model using coefficient a and Psi
    def pce_constructor(input):
        output = 0
        for i in range(np.size(alpha,0)):
            Psi_i = 1
            for j in range(np.size(alpha,1)):
                Psi_i = np.multiply(Psi_i, np.polyval(He_p_mat[alpha[i,j],:], input[:,j]))
            output = output + a[i] * Psi_i / norm[i]
        return output

    f       = pce_constructor
    Yhat    = f(X).reshape(n_s,-1)

    return f,Yhat,a,Psi,Q_sq