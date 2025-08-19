from ERADist import ERADist
from ERANataf import ERANataf
import numpy as np
import matplotlib.pylab as plt

'''
---------------------------------------------------------------------------
Example file: Definition and use of ERADist empirical 
---------------------------------------------------------------------------
In this script the definition of a joint distribution object including an
empirical distribution based on a dataset with the ERANataf class and the
use of its methods are shown. 
For more information on ERANataf please have a look at the provided
documentation. 
---------------------------------------------------------------------------
Developed by:
Michael Engel

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2025-07
---------------------------------------------------------------------------
References:
1. Documentation of the ERA Distribution Classes
---------------------------------------------------------------------------
'''

def sample_bimodal_gaussian(n_samples=1000, mix_weights=(0.4, 0.6),
                            means=(-2, 3), stds=(0.7, 1.2)):
    comps = np.random.choice([0, 1], size=n_samples, p=mix_weights)
    data = np.where(
        comps == 0,
        np.random.normal(loc=means[0], scale=stds[0], size=n_samples),
        np.random.normal(loc=means[1], scale=stds[1], size=n_samples),
    )
    return data

if __name__ == "__main__":
    np.random.seed(2025) #initializing random number generator
    n = 2000 #number of data points
    n_samples = 1000 # number of samples for the plot
    
    # generate a bimodal Gaussian mixture dataset
    data = sample_bimodal_gaussian(n_samples=n,
                                   mix_weights=(0.2, 0.8),
                                   means=(-2, 3),
                                   stds=(0.5, 1.0))

    ''' Definition of the ERANataf object '''
    
    # definition of the marginal distributions
    M = list()
    M.append(ERADist('normal','PAR',[4,2]))
    M.append(ERADist('gumbel','MOM',[1,2]))
    M.append(ERADist('empirical','DATA',[data, None, "linear", None, {}]))
    
    # definition of the correlation matrix
    Rho  = np.array([[1.0, 0.5, 0.5],[0.5, 1.0, 0.5],[0.5, 0.5, 1.0]])
    
    # definition of the joint distribution    
    T_Nataf = ERANataf(M,Rho)
    
    ''' Methods '''
    
    # generation of n random samples to work with
    X = T_Nataf.random(5)
    print("X", X)
    
    # computation of joint PDF
    PDF_X = T_Nataf.pdf(X)
    print("PDF", PDF_X)
    
    # computation of joint CDF
    CDF_X = T_Nataf.cdf(X)
    print("CDF", CDF_X)
    
    # transformation from physical space X to the standard normal space U and
    # Jacobian of the transformation of the first sample
    U, Jac_X2U = T_Nataf.X2U(X,'Jac')
    print("U", U)
    print("Jac_X2U", Jac_X2U)
    
    # transformation from standard normal space U to physical space X and
    # Jacobian of the transformation of the first sample
    X_backtransform, Jac_U2X = T_Nataf.U2X(U,'Jac')
    print("X backtransformed", X_backtransform)
    print("Jac_U2X", Jac_U2X)
    
    
    ''' Creation of samples in physical space, transformation to standard normal
     space and plot of the samples to show the isoprobabilistic transformation '''
    
    # generation of n random samples
    X_plot = T_Nataf.random(n_samples)
    # transformation from physical space X to the standard normal space U 
    U_plot = T_Nataf.X2U(X_plot)
    
    # plot samples in physical and standard normal space
    fig_Samples = plt.figure(figsize=[16, 9])
    
    fig_SamplesAx1 = fig_Samples.add_subplot(121, projection='3d')
    fig_SamplesAx1.scatter(X_plot[:,0], X_plot[:,1], X_plot[:,2])
    fig_SamplesAx1.set_title('Physical Space')
    fig_SamplesAx1.set_xlabel(r'$X_{1}$')
    fig_SamplesAx1.set_ylabel(r'$X_{2}$')
    fig_SamplesAx1.set_zlabel(r'$X_{3}$')
    
    fig_SamplesAx2 = fig_Samples.add_subplot(122, projection='3d')
    fig_SamplesAx2.scatter(U_plot[:,0], U_plot[:,1], U_plot[:,2],color='r')
    fig_SamplesAx2.set_title('Standard normal space')
    fig_SamplesAx2.set_xlabel(r'$U_{1}$')
    fig_SamplesAx2.set_ylabel(r'$U_{2}$')
    fig_SamplesAx2.set_zlabel(r'$U_{3}$')
    
    plt.show()
