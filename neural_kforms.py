import numpy as np
import matplotlib as mpl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors


# for building the boundary matrices
import gudhi as gd
from scipy import sparse
from scipy.sparse import coo_matrix,diags


# ------------------ #
# Functions for dealing with vector fields and path integration
# ------------------ #

## plot a vector field given a function f: R^2 -> R^2
def plot_component_vf(f, ax, comp = 0, x_range=5, y_range=5):
    """ 
    A function for plotting a component of a vector field given a function f: R^2 -> R^2

    Parameters
    ----------
    f : a Pytorch Sequential object
        A function f: R^2 -> R^2, represented as a Pytorch Sequential object
    
    ax : matplotlib axis object
        The axis on which to plot the vector field

    comp : int
        The component of the vector field to plot

    x_range : float
        The range of x values to plot
    
    y_range : float
        The range of y values to plot

    Returns
    -------
    None
    """
    x = np.linspace(-x_range,x_range,20)
    y = np.linspace(-y_range,y_range,20)
    X,Y = np.meshgrid(x,y)

    X = torch.tensor(X).double()
    Y = torch.tensor(Y).double()


    U = np.zeros((20,20))
    V = np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            inp = np.array([X[i,j],Y[i,j]])
            inp = torch.tensor(inp).float()
            tv = f.forward(inp).reshape(2,-1)

            U[i,j] = tv[:,comp][0]
            V[i,j] = tv[:,comp][1]
    ax.quiver(X,Y,U,V)



# a function for turning a path into a chain
def path_to_chain(p):
    """  
    A function for turning a path into a chain

    Parameters
    ----------
    p : numpy array
        A path in R^n, represented as a numpy array of shape (p,n), where p is the number of points in the path

    Returns
    -------
    chain : numpy array
        A chain in R^n, represented as a numpy array of shape (p-1,2,n), where p is the number of points in the path.
        The middle index corresponds to start and endpoints of the edges in the chain.
    """

    r = len(p) - 1

    n = p[0].shape[0]
    
    
    chain = torch.zeros((r,2,n))

    chain[:,1,:] = torch.tensor(p[1:,:])
    chain[:,0,:] = torch.tensor(p[0:-1,:])

    return chain

    
# a function for turning a chain into a discretized chain
def discretize_chain(chain,d):
    """ 
    A function for turning a chain into a discretized chain

    Parameters
    ----------
    chain : numpy array
        A chain in R^n, represented as a numpy array of shape (p-1,2,n), where p is the number of points in the path.

    d : int
        The number of points in the discretized chain

    Returns
    -------
    d_chain : numpy array
        A discretized chain in R^n, represented as a numpy array of shape (p-1,d,n), where p is the number of points in the path.

    """

    r = chain.shape[0]

    n = chain.shape[2]

    d_chain = torch.zeros((r,d,n))

    t = np.linspace(0,1,d)

    for i in range(d):

        d_chain[:,i,:] = (1-t[i]) * chain[:,0,:] + t[i] * chain[:,1,:] 

    return d_chain


# a function for turning a chain into an integration matrix
def integration_matrix(vf,chain, d = 5):
    """
    A function for generating a integration matrix from a chain and a vector field

    Parameters
    ----------
    vf : a Pytorch Sequential object
        The vector field to be applied to the chain
    
    chain : a torch tensor of shape (r,2,n)
        The chain to be turned into an integration matrix

    d : int
        The number of discrete steps in the discretization of the chain
    
    Returns
    -------
    out : a torch tensor of shape (r,c)
        The integration matrix
    """

    
    # discretize the chain
    chain = discretize_chain(chain, d)

    # number of simplicies
    r = chain.shape[0]

    # number of discrete steps
    d = chain.shape[1]

    # dimension of ambient space
    n = chain.shape[2]

    # number of feature-cochains in the integration matrix
    c = int(vf[-1].out_features / n)

    # apply the vector field to the discretized chain
    out = vf(chain).reshape((r,d,n,c))

    # calculate the simplex gradients
    simplex_grad = chain[:,1,:] - chain[:,0,:]

    # swap dimensions n and c in out
    out = out.permute(0,1,3,2)

    # calculate the inner product of the vector field and the simplex gradients at each discrete step on each simplex
    inner_prod = torch.matmul(out,simplex_grad.T/(d-1))

    # take diagonal of out3 along axis 0 and 3 (this corresponds to correcting the broadcasted multplication effect)
    inner_prod = torch.diagonal(inner_prod, dim1 = 0, dim2 = 3)

    # permute dimensions 0 and 2 of out4
    inner_prod = inner_prod.permute(2,0,1)

    # apply the trapzoidal rule to the inner product
    cdm = (inner_prod[:,1:,:] + inner_prod[:,0:-1,:])/2
    cdm = cdm.sum(axis = 1)

    return cdm

# a function for calculating the length of a path
def path_length(path):
    """
    Calculates the length of a path
    """
    length = 0
    for i in range(len(path)-1):
        length += np.linalg.norm(path[i+1]-path[i])
    return length

# -----------------------------------------------------------
# Functions for dealing with simplicial complexes
# -----------------------------------------------------------


# turn ripser type cocycle into a vectorized cochain
def vectorize(cocycle, simplices):

    """ 
    function to vectorize a cocycle

    Parameters
    ----------
        cocycle: np.array
            cocycle in the form of a list of [i,j,val] where i,j are the vertices
            and val is the value of the cocycle on the edge between i and j
    
    Returns
    -------
        cocycle_vec: np.array
            a cochain vectorized over the edges of the simplicial complex which agrees with the indexing in simplices
    """


    for k in range(len(cocycle)):
        [i,j,val] = cocycle[k,:]
        
        # correct orientation

        if i > j:
            i,j = j,i
            val = -val
            cocycle[k,:] = [i,j,val]

    # vectorize cocycle
    cocycle_vec = np.zeros(len(simplices[1]))
    for k in range(cocycle.shape[0]):

        [i,j,val] = cocycle[k,:]

        # check if edge is in simplices[1], if so, add value to vector
        # this is because we may need to restrict the cocycle to a subcomplex
        if frozenset([i,j]) in simplices[1].keys():
            cocycle_vec[simplices[1][frozenset([i,j])]] = val
    
    return cocycle_vec




# create ripser type cochain from a vectorized cochain
def devectorize(projection, simplices):
    
        """ 
        function to devectorize a cocycle
    
        Parameters
        ----------
            projection: np.array
                vectorized cocycle
        
        Returns
        -------
            cocycle: np.array
                cocycle in the form of a list of [i,j,val] where i,j are the vertices
                and val is the value of the cocycle on the edge between i and j
        """
    
        c = np.zeros((len(simplices[1]),3))
        for k in range(len(simplices[1])):
            [i,j] = np.sort(list(list(simplices[1])[k]))
            c[k,:] = [i,j,projection[k]]

        
        return c




# extract the simplices from the simplex tree
def extract_simplices(simplex_tree):
    """Extract simplices from a gudhi simplex tree.

    Parameters
    ----------
    simplex_tree: gudhi simplex tree

    Returns
    -------
    simplices: List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the 0-simplices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.
    """
    
    simplices = [dict() for _ in range(simplex_tree.dimension()+1)]
    for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
        k = len(simplex)
        simplices[k-1][frozenset(simplex)] = len(simplices[k-1])
    return simplices


# a function for turning a path into a chain
def one_skeleton_to_chain(one_simplices, points):
    """  
    A function for turning a 1-skeleton into a chain

    Parameters
    ----------
    p : numpy array
        A set of 1-simplices extracted from the Gudhi simplex tree

    points : numpy array
        A set of vertices corresponding to the embedding

    Returns
    -------
    chain : numpy array
        A chain in R^n, represented as a numpy array of shape (p-1,2,n), where p is the number of points in the path.
        The middle index corresponds to start and endpoints of the edges in the chain.
    """

    r = len(one_simplices)

    n = points[0].shape[0]
    
    
    chain = torch.zeros((r,2,n))

    for i in range(r):

        chain[i,0,:] = torch.tensor(points[one_simplices[i][0]])
        chain[i,1,:] = torch.tensor(points[one_simplices[i][1]])
    

    return chain

# Build the boundary operators
def build_boundaries(simplices):
    """Build unweighted boundary operators from a list of simplices.

    Parameters
    ----------
    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the 0-simplices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    Returns
    -------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension: i-th boundary is in i-th position
    """
    boundaries = list()
    for d in range(1, len(simplices)):
        idx_simplices, idx_faces, values = [], [], []
        for simplex, idx_simplex in simplices[d].items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1)**i)
                face = simplex.difference({left_out})
                idx_faces.append(simplices[d-1][face])
        assert len(values) == (d+1) * len(simplices[d])
        boundary = coo_matrix((values, (idx_faces, idx_simplices)),
                                     dtype=np.float32,
                                     shape=(len(simplices[d-1]), len(simplices[d])))
        
        
        boundaries.append(boundary)


    return boundaries



