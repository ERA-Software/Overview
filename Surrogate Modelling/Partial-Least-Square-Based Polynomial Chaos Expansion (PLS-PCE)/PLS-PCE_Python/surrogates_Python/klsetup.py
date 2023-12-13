import numpy as np
from numpy.matlib import repmat

# Make sure having these scripts under the same workspace
from myroots import myroots

def ef(omega, lam, x):
    # This function evaluates the eigenfunctions at the point x. Input omega 
    # are the eigenvalues for the given correlation length lam.

    norm = np.divide( 1 , (np.sqrt( np.multiply(np.sin(2*omega),(0.25*(lam**2 * omega - 1./omega))) - \
    0.5* lam * np.cos(2*omega) + (0.5*(1+lam+lam**2 * np.power(omega,2) )))) )
    # norm is a 1 x mkl vector
    # x is a n x 1 vector
    
    mkl = len(omega)
    n = len(x)
    XO = np.multiply( repmat(x,1,mkl) , repmat(omega,n,1) )
    out = np.multiply( repmat(norm,n,1) , (np.sin(XO)+lam* np.multiply(repmat(omega,n,1),np.cos(XO))) )
    
    return out

def klsetup(lam, mkl, xq):

    # calculate the eigenvalues.
    c=1/lam
    omega = myroots(lam, mkl)
    eigvals = np.divide( 2*c , np.power(omega,2) + c**2 )
    eigfcts = ef(omega, lam, xq)
    n = len(xq)
    A = np.multiply( repmat(np.sqrt(eigvals),n,1) , eigfcts )

    #return [A,eigvals]
    return A