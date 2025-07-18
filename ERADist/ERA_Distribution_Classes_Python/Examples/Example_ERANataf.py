from ERADist import ERADist
from ERANataf import ERANataf
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

'''
---------------------------------------------------------------------------
Example file: Definition and use of ERADist objects
---------------------------------------------------------------------------

 In this script the definition of a joint distribution object with the
 ERANataf class and the use of its methods are shown.
 For more information on ERANataf please have a look at the provided
 documentation or execute the command 'help(ERANataf). 
---------------------------------------------------------------------------
Developed by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Luca Sardi
Alexander von Ramm
Matthias Willer
Peter Kaplan

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2021-03
---------------------------------------------------------------------------
References:
1. Documentation of the ERA Distribution Classes
---------------------------------------------------------------------------
'''

np.random.seed(2021) #initializing random number generator
n = 5; #number of data points
n_samples = 1000 # number of samples for the plot


''' Definition of the ERANataf object '''

# definition of the marginal distributions
M = list()
M.append(ERADist('normal','PAR',[4,2]))
M.append(ERADist('gumbel','MOM',[1,2]))
M.append(ERADist('exponential','PAR',4))

# definition of the correlation matrix
Rho  = np.array([[1.0, 0.5, 0.5],[0.5, 1.0, 0.5],[0.5, 0.5, 1.0]])

# definition of the joint distribution    
T_Nataf = ERANataf(M,Rho)


''' Methods '''

# generation of n random samples to work with
X = T_Nataf.random(5)

# computation of joint PDF
PDF_X = T_Nataf.pdf(X)

# computation of joint CDF
CDF_X = T_Nataf.cdf(X)

# transformation from physical space X to the standard normal space U and
# Jacobian of the transformation of the first sample
U, Jac_X2U = T_Nataf.X2U(X,'Jac')

# transformation from standard normal space U to physical space X and
# Jacobian of the transformation of the first sample
X_backtransform, Jac_U2X = T_Nataf.U2X(U,'Jac')


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
