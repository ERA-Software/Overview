import numpy as np
import scipy as sp

def  febar2( Ex, q, A, L, nele ):
    # returns the tip displacement of a bar subjected to variable axial loading
    # Ex: Young's modulus evaluated at quadrature nodes
    # q: uniformly distributed load
    # A: cross sectional area
    # L: length of beam
    # nele: number of elements

    # number of nodes
    nnode = nele+1

    # element length
    l = L/nele

    # stiffness matrix
    K = sp.sparse.coo_matrix((nnode,nnode))

    # consistent nodal forces
    F = np.zeros((nnode,1))

    # shape functions and their derivatives
    wGL = np.divide( [5, 8, 5] , 9 ).reshape(1,-1)

    # assemble stiffness matrix and force vector
    F[0:-1] = l*q
    F[0]     = F[0]/2
    F[-1]   = F[-1]/2

    Kei = A / l / 2 * Ex
    Ke = np.sum( np.reshape( np.multiply(Kei, np.matlib.repmat(wGL.T, nele, 1)), (3, nele) ), 0 )  

    # scipy.sparse.coo_matrix((data, (i, j)), [shape=(M, N)])
    K = K + sp.sparse.coo_matrix( ( Ke, (np.arange(nele), np.arange(nele)) ), shape=(nnode,nnode) )
    K = K + sp.sparse.coo_matrix( ( -Ke, (np.arange(nele), np.arange(1,nele+1)) ), shape=(nnode,nnode) )
    K = K + sp.sparse.coo_matrix( ( -Ke, (np.arange(1,nele+1), np.arange(nele)) ), shape=(nnode,nnode) )
    K = K + sp.sparse.coo_matrix( ( Ke, (np.arange(1,nele+1), np.arange(1,nele+1)) ), shape=(nnode,nnode) )
            
    Kred = K[1:nnode,1:nnode].toarray()
    Fred = F[1:nnode].flatten()

    # solution for nodal displacements
    u = np.linalg.solve(Kred, Fred)

    # tip displacement
    u_tip = u[ -1 ]

    return u_tip