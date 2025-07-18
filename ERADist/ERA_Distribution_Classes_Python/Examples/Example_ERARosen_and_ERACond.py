from ERADist import ERADist
from ERACond import ERACond
from ERARosen import ERARosen
import numpy as np

#from add_fun import *

'''
---------------------------------------------------------------------------
Example file: Definition and use of ERARosen and ERACond objects
---------------------------------------------------------------------------

 In this script the definition of a multivariate distribution with the
 ERARosen class using ERACond and ERADist distribution objects is shown.
 For more information on ERARosen and ERACond please have a look at the
 provided documentation or execute the commands 'help(ERARosen)' or 
 'help(ERACond)' respectively.
 
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
Version 2022-01
---------------------------------------------------------------------------
References:
1. Documentation of the ERA Distribution Classes
---------------------------------------------------------------------------
'''

np.random.seed(2021) #initializing random number generator
n = 5; #number of data points

''' Creation of the different distribution objects '''

# marginal distributions defined by ERADist classes
x1_dist = ERADist('normal','PAR',[3,2],'A');
x2_dist = ERADist('normal','PAR',[5,4],'B');

# conditional distributions defined by ERACond classes
a=3 # use of a constant in function handle for demonstration purposes
# conditional distributions defined by ERACond classes 
x3_dist = ERACond('normal','PAR',lambda X:[X[0]*X[1]+a,2],'C')
# use of a user-defined function for demonstration purposes in x4_dist
def subtraction(a,b):
                    return(a-b)
x4_dist = ERACond('normal','PAR',lambda X:[subtraction(X[0],X[1]),abs(X[0])],'D');
x5_dist = ERACond('exponential','PAR',lambda X:abs(X[0]**2-X[1]),'E');
x6_dist = ERACond('normal','PAR',lambda X:[3*X,4],'F');
x7_dist = ERACond('normal','PAR',lambda X:[X[0]+X[1]-X[2],1],'G');

# # alternative for conditional distributions with more complicated parameter functions  
# x3_dist = ERACond('normal','PAR',lambda X: [np.max(X,axis=0),2],'C');
# x4_dist = ERACond('normal','PAR',lambda X: [np.linalg.norm(X,axis=0),abs(X[0])],'D');
# x5_dist = ERACond('normal','PAR',lambda X: [np.mean(X**2,axis=0)*np.mean(X,axis=0), X[0]**2],'E');
# x6_dist = ERACond('normal','PAR',lambda X: [3*X, 4],'F');
# A=1; B=np.array([[1,2,3]]); C=np.array([[2,1,0],[1,2,1],[0,1,2]]); D=np.array([[3,4,5]]).T
# x7_dist = ERACond('normal','PAR',lambda Y:[A+B@np.linalg.inv(C)@(Y-D),1],'G');

# collecting all the distribution objects in a list
dist = [x1_dist,x2_dist,x3_dist,x4_dist,x5_dist,x6_dist,x7_dist]

# describing the dependency by parents using a list
depend = [[],[],[0,1],[0,2],[2,1],3,[2,3,4]]

# creation of the ERARosen class
X_dist = ERARosen(dist,depend)



''' Methods of the ERARosen class '''

# plot of the graph definining the dependency in the distribution

# ...with naming of the nodes according to their order in input dist
# (overwrites the ID of the distribution)
figure_numb = X_dist.plotGraph('numbering');

# ... with naming of the nodes according to their ID
figure = X_dist.plotGraph();

# creation of n samples of the joint distribution
X = X_dist.random(n)

# transformation from physical space X to the standard normal space U
U = X_dist.X2U(X)
 
# transformation from standard normal space U to physical space X
X_backtransform = X_dist.U2X(U)

# computation of joint PDF
pdf = X_dist.pdf(X)
