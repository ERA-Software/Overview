import numpy as np

def E_real2( efx, zeta, mu, sigma ):
    
    # sample lognormal random field
    Ex = mu * np.ones((np.size(efx,0),1))
    Ex = Ex + np.array ( np.dot ( sigma , np.dot(efx, zeta) ),ndmin=2 ).T
    Ex = np.exp(Ex)

    return Ex