# import of modules
import numpy as np
from scipy import stats
from ERADist import ERADist
from ERACond import ERACond
import networkx as nx
import matplotlib.pyplot as plt

'''
---------------------------------------------------------------------------
Generation of joint distribution objects based on marginal and conditional 
distribution objects.
---------------------------------------------------------------------------
Developed by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Luca Sardi
Nicola Bronzetti
Alexander von Ramm
Matthias Willer
Peter Kaplan

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Version 2022-01:
* Adaptation to the modification in the ERACond class for the definition of
the parameter/moment functions.
First Release, 2021-03
--------------------------------------------------------------------------
This software generates joint distribution objects. The joint distribution
is defined by the connection of different marginal and conditional
distributions through the framework of a Bayesian network. 
While the marginal distributions are defined by ERADist classes, the
conditional distribution are defined by ERACond classes(see respective
classes).
The ERARosen class allows to carry out the transformation from physical
space X to standard normal space U (X2U) and vice versa (U2X) according
to the Rosenblatt transformation.
The other methods allow the computation of the joint PDF, the generation of
multivariate random samples and the plot of the Bayesian network which
defines the dependency between the different marginal and conditional
distributions.
---------------------------------------------------------------------------
References:

1. Hohenbichler, M. and R. Rackwitz (1981) - Non-Normal Dependent Variables 
   in Structural Reliability. 
   Journal of Eng. Mech., ASCE, 107, 1227-1238.

2. Rosenblatt, M. (1952) - Remarks on a multivariate transformation.
   Ann. Math. Stat., 23, 470-472.

3. Documentation of the ERA Distribution Classes
---------------------------------------------------------------------------
'''

# %%
class ERARosen(object):
    """
    Generation of joint distribution objects. 
    Construction of the joint distribution object with
    
    Obj = ERARosen(dist,depend,opt)
    
    'dist' must be a list or array which contains all the
    marginal distributions (ERADist objects) and conditional distributions
    (ERACond objects) that define the joint distribution.
    
    'depend' describes the dependency between the different marginal and
    conditional distributions. The dependency is defined by a list of arrays 
    which contain the indices of the parents of the respective distributions.
    The arrays within the list must be ordered according to the place of the 
    corresponding distribution in the input 'dist'. If a distribution is
    defined as a marginal, and therefore has no parents, an empty array([])
    must be given for that distribution in 'depend'. For conditional 
    distributions the order of the indices within one of the arrays
    corresponds to the order of the variables of the respective function
    handle of the respective ERACond object.
    """
    
    def __init__(self, dist, depend):
        """
        Constructor method, for more details have a look at the
        class description.
        """
        self.Dist = dist
        self.Parents = depend
        
        n_dist = len(dist)
        n_dist_dep = len(depend)
        if n_dist != n_dist_dep:
            raise RuntimeError("The number of distributions according to the inputs"
                               " dist and depend doesn't match.")
        
        n_parents = np.zeros(n_dist)
        for i in range(n_dist):
            if isinstance(dist[i],ERACond):
                n_parents[i] = dist[i].Param.__code__.co_argcount
            elif not isinstance(dist[i],ERADist):
                raise RuntimeError("The objects in dist must be either ERADist or ERACond objects.")
        
        # build adjacency matrix
        adj_mat = np.zeros([n_dist,n_dist])
        for i in range(n_dist):
            adj_mat[i,depend[i]] = 1
        # check if obtained network represents a directed acyclical graph 
        adj_prod = np.identity(n_dist)
        for i in range(n_dist+1):
            adj_prod = np.matmul(adj_prod, adj_mat)
            if sum(np.diag(adj_prod)) != 0:
                raise RuntimeError("The graph defining the dependence between the different "
                                   "distributions must be directed and acyclical.")
        
        self.Adjacency = np.matrix(adj_mat)
        
        # sort distributions according to dependencies
        layers = list()
        rem_dist = np.arange(0,n_dist)
        while len(rem_dist) > 0:
            n_dep_rem = np.sum(adj_mat,1)
            curr_d = n_dep_rem == 0 # distributions on current layer
            curr_dist = rem_dist[curr_d]
            layers.append(curr_dist) # save distributions on current layer
            adj_mat[:,curr_dist] = 0
            adj_mat = adj_mat[np.logical_not(curr_d),:]
            rem_dist = rem_dist[np.logical_not(curr_d)]
        
        if len(layers) > 1:
            self.Order = [layers[0], np.concatenate(layers[1:])]
            self.Layers = layers
        else:
            raise RuntimeError("The defined joint distribution consists only of independent distributions."
                               "This type of joint distribution is not supported by ERARosen.")
            
# %%  
    def X2U(self, X, error=True):
        """
        Carries out the transformation from physical space X to
        standard normal space U.    
        X must be a [n,d]-shaped array (n = number of data points,
        d = dimensions).
        If no error message should be given in case of the detection
        of an improper distribution, give error=False as second input.
        The output for the improper data points is then given as nan.
        """
        
        n_dim = len(self.Dist)
        X = np.array(X, ndmin=2, dtype=float)
        
        # check if all marginal and conditional distributions are continuous
        for i in range(n_dim):
            if self.Dist[i].Name in ['binomial','geometric','negativebinomial','poisson']:
                raise RuntimeError("At least one of the marginal distributions or conditional distributions " 
                                   "is a discrete distribution, the transformation X2U is therefore not possible.")
                
        # check of the dimensions of input X  
        if X.ndim > 2:
            raise RuntimeError("X must have not more than two dimensions. ")
        if np.shape(X)[1] == 1 and n_dim != 1:
            # in case that only one point X is given, he can be defined either as row or column vector
            X = X.T
        if np.shape(X)[1] != n_dim:
            raise RuntimeError("X must be an array of size [n,d], where d is the"
                               " number of dimensions of the joint distribution.")
            
        n_X = np.shape(X)[0]
        U = np.zeros([n_X,n_dim])
        
        for i in self.Order[0]:
            U[:,i] = stats.norm.ppf(self.Dist[i].cdf(X[:,i]))
    
        for i in self.Order[1]:
            U[:,i] = stats.norm.ppf(self.Dist[i].condCDF(X[:,i],X[:,self.Parents[i]]))
            
        # find rows with nan
        lin_ind = np.any(np.isnan(U),1)
        
        if error:
            if not all(np.logical_not(lin_ind)):
                raise RuntimeError("Invalid joint distribution was created.")
        else:
            U[lin_ind,:] = np.nan
        
        return np.squeeze(U)
            
# %%    
    def U2X(self, U, error=True):
        """
        Carries out the transformation from standard normal space U 
        to physical space X .    
        U must be a [n,d]-shaped array (n = number of data points,
        d = dimensions).
        If no error message should be given in case of the detection
        of an improper distribution, give error=False as second input.
        The output for the improper data points is then given as nan.
        """        
        
        n_dim = len(self.Dist)
        U = np.array(U, ndmin=2, dtype=float)
                
        # check of the dimensions of input U  
        if U.ndim > 2:
            raise RuntimeError("U must have not more than two dimensions. ")
        if np.shape(U)[1] == 1 and n_dim != 1:
            # in case that only one point X is given, he can be defined either as row or column vector
            U = U.T
        if np.shape(U)[1] != n_dim:
            raise RuntimeError("U must be an array of size [n,d], where d is the"
                               " number of dimensions of the joint distribution.")
            
        n_U = np.shape(U)[0]
        X = np.zeros([n_U,n_dim])
        CDF_values = stats.norm.cdf(U)
        
        for i in self.Order[0]:
            X[:,i] = self.Dist[i].icdf(CDF_values[:,i])
            
        for i in self.Order[1]:
            X[:,i] = self.Dist[i].condiCDF(CDF_values[:,i],X[:,self.Parents[i]])
            
        # find rows with nan
        lin_ind = np.any(np.isnan(X),1)
        
        if error:
            if not np.all(np.logical_not(lin_ind)):
                raise RuntimeError("Invalid joint distribution was created.")
        else:
            X[lin_ind,:] = np.nan
        
        return np.squeeze(X)            
            
# %%    
    def pdf(self, X, error=True):
        """
        Computes the joint PDF.
        X must be a [n,d]-shaped array (n = number of data points,
        d = dimensions).
        If no error message should be given in case of the detection
        of an improper distribution, give error=False as second input.
        The output for the improper data points is then given as nan.
        """
        
        n_dim = len(self.Dist)
        X = np.array(X, ndmin=2, dtype=float)
                
        # check of the dimensions of input X  
        if X.ndim > 2:
            raise RuntimeError("X must have not more than two dimensions. ")
        if np.shape(X)[1] == 1 and n_dim != 1:
            # in case that only one point X is given, he can be defined either as row or column vector
            X = X.T
        if np.shape(X)[1] != n_dim:
            raise RuntimeError("X must be an array of size [n,d], where d is the"
                               " number of dimensions of the joint distribution.")
            
        n_X = np.shape(X)[0]
        pdf_values = np.zeros([n_X,n_dim])
        
        for i in self.Order[0]:
            pdf_values[:,i] = self.Dist[i].pdf(X[:,i])
            
        for i in self.Order[1]:
            pdf_values[:,i] = self.Dist[i].condPDF(X[:,i],X[:,self.Parents[i]])
            
        jointpdf = np.prod(pdf_values, 1)
        nan_ind = np.isnan(jointpdf)
        
        if error:
            if not np.all(np.logical_not(nan_ind)):
                raise RuntimeError("Invalid joint distribution was created.")
        
        if np.size(jointpdf) == 1:
            return jointpdf[0]
        else:
            return jointpdf    
        
# %%    
    def random(self, n=1):
        """
        Creates n samples of the joint distribution.
        Every row in the output array corresponds to one sample.
        """
        
        n_dim = len(self.Dist)
        X = np.zeros([n,n_dim])
        
        for i in self.Order[0]:
            X[:,i] = self.Dist[i].random(n)
            
        for i in self.Order[1]:
            try:
                X[:,i] = self.Dist[i].condRandom(X[:,self.Parents[i]])
            except ValueError:
                raise RuntimeError("Invalid joint distribution was created.")            
        
        return np.squeeze(X)

# %%    
    def plotGraph(self,opt=False):
        """
        Plots the Bayesian network which defines the dependency  
        between the different distributions.
        If opt is given as 'numbering' the nodes are named according
        to their order of input in dist(e.g., the first distribution
        is named #0, etc.). If no ID was given to a certain 
        distribution, the distribution is also named according to its
        position in dist, otherwise the property ID is taken as the
        name of the distribution.
        """
        
        n_layer = len(self.Layers)
        vert = np.flip(np.linspace(0,1,n_layer))
        pos_n = dict()
        pos_l = dict()
        for i in range(n_layer):
            cur_l = self.Layers[i]
            n_cur = len(cur_l)
            horiz = np.linspace(0,1,n_cur+2)
            for j in range(n_cur):
                pos_n[str(cur_l[j])] = (horiz[j+1],vert[i])
                pos_l[cur_l[j]] = (horiz[j+1]+0.06,vert[i])                
                
        n_dim = len(self.Dist)
        labels = dict()
        if not opt:
            for i in range(n_dim):
                if self.Dist[i].ID:
                    labels[i] = self.Dist[i].ID
                else:
                    labels[i] = '#'+str(i)
        elif not opt.lower == 'numbering':
            for i in range(n_dim):
                labels[i] = '#'+str(i)
        else:
            raise RuntimeError("opt must be given as 'numbering'.")
                
        G_Adj = nx.from_numpy_matrix(self.Adjacency)
        G = nx.DiGraph()
        for i in range(1,n_layer):
            cur_l = self.Layers[i]
            n_cur = len(cur_l)
            for j in range(n_cur):
                s_n = np.array(self.Parents[cur_l[j]],ndmin=1)
                for k in range(np.size(s_n)):
                    G.add_edge(str(s_n[k]),str(cur_l[j]))
            
        nx.draw(G, pos_n,node_color='k',alpha = 0.3,node_size=100,arrowsize=20,arrows=True)
        nx.draw_networkx_labels(G_Adj,pos_l,labels,colors='r',font_size=15)
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.1,1.1])
        plt.show()
