import numpy as np
from numpy.matlib import repmat

def eval_quad(nele,L,xGL):

    ind = np.arange(1,nele+1)      
    l = L/nele       
    XQ = l/2 * repmat(xGL.T, 1, len(ind)) + l/2 * repmat( (2*ind-1), len(xGL),1)
    xq = np.reshape(XQ.T,(300,1))

    return xq