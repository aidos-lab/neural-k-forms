import numpy as np
import matplotlib as mpl  
import torch  
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import math 

import scipy 

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



# -----------------------------------------------------------
# Functions for 2-forms 
# -----------------------------------------------------------


# ---------
# Data generation functions
# ---------

def unit_square_grid(n=3):
    """
    Make a grid of points in the unit square [0,1] x [0,1]

    Parameters:
    n : int
        The number of points in each direction
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    return np.array([X.flatten(), Y.flatten()]).T

def plot_unit_square_grid(n=3):
    """ 
    Plot the alpha complex of a grid in the unit square
    Return: the points and the alpha complex
    """
    pts = unit_square_grid(n)
    ac = gd.AlphaComplex(pts).create_simplex_tree()
    plt.scatter(pts[:,0], pts[:,1])

    for s in ac.get_skeleton(2):
        if len(s[0]) == 2: 
            plt.plot(pts[s[0], 0], pts[s[0], 1], 'r-')
    return pts, ac


def random_surface_sin_x(n, scale = 1, period = 1, eps = 0.1):
    """ 
    Generate a random surface in R^3 with a sine wave in the x direction

    Input: 
        n: number of points in each direction
        scale: scale of the surface
        period: period of the sine wave
        eps: magnitude of the noise
    Output:
        X, Y, Z: the coordinates of the surface
        ar: an array of points    
    """

    trans_x = np.random.uniform(-10, 10, 1) # translation in x direction
    trans_y = np.random.uniform(-10, 10, 1) # translation in y direction 

    x = np.arange(0, n, 1)/scale + trans_x # x coordinates
    y = np.arange(0, n, 1)/scale + trans_y # y coordinates

    Eps = np.random.uniform(-eps,eps, (n,n)) # noise  

    X, Y = np.meshgrid(x, y) # create a grid of points
    Z = np.zeros(X.shape) # initialize the z coordinates

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = math.sin(X[i,j]/period) 
    
    Z = Z + Eps # add noise to the surface 
    ar = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    ar = torch.from_numpy(ar)

    return X, Y, Z, ar

def random_surface_sin_y(n, scale =1, period =1, eps = 0.1):
    """
    Generate a random surface in R^3 with a sine wave in the y direction

    Input:
        n: number of points in each direction
        scale: scale of the surface
        period: period of the sine wave
        eps: magnitude of the noise
    Output:
        X, Y, Z: the coordinates of the surface
        ar: an array of points
    """

    trans_x = np.random.uniform(-10, 10, 1)
    trans_y = np.random.uniform(-10, 10, 1)
    x = np.arange(0, n, 1)/scale + trans_x
    y = np.arange(0, n, 1)/scale + trans_y
    Eps = np.random.uniform(-eps,eps, n)    
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = math.sin(Y[i,j]/period) 
            # return an array of points
    Z = Z + Eps      
    ar = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    ar = torch.from_numpy(ar)

    return X, Y, Z, ar


def generate_surfaces_sin_x(num_surf, n_pts, scale =1, period =1, eps = 0.1): 
    """
    Generate a list of random surfaces in R^3 with a sine wave in the x direction

    Input:
        num_surf: number of surfaces to generate
        n_pts: number of points in each direction
        scale: scale of the surface
        period: period of the sine wave
        eps: magnitude of the noise
    Output:
        surfaces: a list of num_surf surfaces 
    """
    surfaces = []
    for i in range(num_surf):
        surfaces.append(random_surface_sin_x(n_pts, scale = scale, period = period, eps = eps))
    return surfaces

def generate_surfaces_sin_y(num_surf, n_pts, scale =1, period =1, eps = 0.1): 
    """
    Generate a list of random surfaces in R^3 with a sine wave in the y direction
    
    Input:
        num_surf: number of surfaces to generate
        n_pts: number of points in each direction
        scale: scale of the surface
        period: period of the sine wave
        eps: magnitude of the noise
    Output:
        surfaces: a list of num_surf surfaces 
    """

    surfaces = []
    for i in range(num_surf):
        surfaces.append(random_surface_sin_y(n_pts, scale = scale, period = period, eps = eps))
    return surfaces


#-------- 
# Functions to help integration of 2-forms
#--------


def subdivide_simplex_torch(n):
    """
    Subdivide a 2-simplex into n^2 2-simplices
    ---- 
    Input: n the number of subdivision of the simplex
    Output: a list of vertices of the subdivision of the simplex
    """
    vertices = []
    for i in range(n+1):
        for j in range(n+1-i):
            vertices.append([i/n,j/n])
    vertices = torch.tensor(vertices)
    return vertices

def is_interior(point):
    """ 
    check if a point is strictly in the interior of a triangle with vertices (0,0), (1,0), (0,1)
    """
    if point[0] > 0 and point[1] > 0 and point[0] + point[1] < 1:
        return True
    else:
        return False

def is_edge(point):
    """
    check if a point is on the edge of a triangle with vertices (0,0), (1,0), (0,1)
    """

    if point[0] == 0 and point[1] > 0 and point[1] < 1:
        # vertical edges
        return True
    elif point[0] > 0 and point[0] < 1 and point[1] == 0:
        # horizontal edge
        return True
    elif point[1] >0 and point[1] < 1 and point[0] >0 and point[0] < 1 and point[0] + point[1] == 1:
        # diagonal edge
        return True
    else:
        return False

def is_vertex(point):
    """
    check if a point is a vertex of a triangle with vertices (0,0), (1,0), (0,1)
    """
    if point[0] == 0 and point[1] == 0:
        return True
    elif point[0] == 1 and point[1] == 0:
        return True
    elif point[0] == 0 and point[1] == 1:
        return True
    else:
        return False
    
def coef_vertex(v): 
    """
    compute the contrbution of a vertex of the subdivision in the integration formula

    If the vertex is in the interior, the vertex is part of 6 2-simplices 
    If the vertex is on an edge, the vertex is part of 3 2-simplices
    If the vertex is a vertex, the vertex is part of 1 2-simplex
    """

    if is_interior(v):
        return torch.tensor([6]).float()
    if is_edge(v):
        return torch.tensor([3]).float()
    if is_vertex(v):
        return torch.tensor([1]).float()

def subdivide_simplex_coef_torch(n):
    """
    Input: n the number of subdivision of the simplex
    Output: a list of vertices of the subdivision of the simplex and the coefficients for the contribution of each vertex
    """
    vertices = []
    for i in range(n+1):
        for j in range(n+1-i):
            vertices.append([i/n,j/n])
    vertices = torch.tensor(vertices)

    coefs = []
    for i in range(len(vertices)):
        coefs.append(coef_vertex(vertices[i]))

    return vertices, coefs


def phi_b(embedded_vertices):
    """build the affine transforamtion matrix that corresponds to the embedded vertices"""
    #dim = embedded_vertices.shape[1] 
    
    a = [embedded_vertices[i] - embedded_vertices[0] for i in range(1,3)]
    phi = torch.stack(a)
    b = embedded_vertices[0] 
    
    return phi.float(), b.float()

def build_determinant_tensor(dim, k =2):
    """Input: n the number of subdivision of the simplex
            k the dimension of the simplex, default is 2 for now 
        Output: Tensors of size (n choose k) x n x n that can be used to compute the determinant entering in the integration formuala
        of  k-form over an embedded simplex
    """

    N = int(scipy.special.binom(dim,k))
    deter_tensor = torch.zeros(N, dim, dim)
    ind = 0
    for i in range(dim):
        for j in range(i+1,dim):
            # each tensor is a nxn matrix with (i,j)-coordinate equal to 1 and (j,i)-coordinate equal to -1
            deter_tensor[ind,i,j] = torch.tensor(1)
            deter_tensor[ind,j,i] = torch.tensor(-1)
            ind += 1

    return deter_tensor.float()


# ----- 
# Functions for integrating 2-forms and creating cochains 
# -----

def integrate_kform(kform, phi, b, det, subdivision_vert, subdivision_coefs, num_sub, dim = 3, k = 2):
                    
    """ Integrate a single 2-form over 2-simplex
    Input: 
        kform: a 2-form
        phi: a matrix containing the embedding of the vertices of the simplex in R^dim
        b: a vector containing the embedding of the basepoint of the simplex in R^dim
        det: the tensor computed by build_determinant_tensor
        dim: the dimension of the embedding space
        k: the dimension of the form
        subdivision_vert: a list of vertices of the subdivision of the simplex
        subdivision_coefs: a list of coefficients for the contribution of each vertex to the integral
        num_sub: the number of subdivisions of the simplex 
    Output:
        integral: the value of integral of the 2-form over the 2-simplex 
    """
   
    num_simplices = int(num_sub**2)
    integral = torch.tensor([0]).float()

    N = int(scipy.special.binom(dim,k))

    for i in range(len(subdivision_vert)): ## This is the slow part  
        p = subdivision_vert[i]
        phi_p = phi.T @ p + b
        g_p = torch.tensor([0]).float() 
        for ind in range(N):
            g_p+= kform(phi_p)[ind] * torch.matmul(torch.matmul(phi[0], det[ind]), phi[1].T)
        cof = subdivision_coefs[i]
        integral += torch.mul(g_p,cof)

    vol = torch.tensor([1/(2*(num_sub**2))]).float() ## I'm not sure we need this but double check the theory 
    return ((integral* vol )/num_simplices).float()


def form2cochain(kform, surface_dict, deter_tensor,subivision_vert, subdivision_coeffs, num_sub, dim =3 ,k = 2): 

    """From a 2-form we get a cochain by integrating the form over every 2-simplex in the simplicial complex

    Input:
        kform: a 2-form
        surface_dict: a dictionary containing all of the information of the triangulated surface in R^dim
        deter_tensor: the tensor computed by build_determinant_tensor 
        dim: the dimension of the embedding space
        k: the dimension of the form
        subivision_vert: a list of vertices of the subdivision of the simplex
        subdivision_coefs: a list of coefficients for the contribution of each vertex to the integral
    
    Output:
        cochain: a cochain on the simplicial complex  
     """

    Emb_comp = surface_dict['points']
    assert Emb_comp.shape[1] == dim, "The dimension of the embedding space is not equal to the number of columns of the matrix Emb_comp"

    cochain = torch.zeros(len(surface_dict['simplices']))
    for i in range(len(surface_dict['simplices'])):
     
        phi_simplex = surface_dict['Phi'][i]
        b_simplex = surface_dict['b'][i]
        cochain[i] = integrate_kform(kform, phi_simplex, b_simplex, deter_tensor, subivision_vert, subdivision_coeffs, dim, k)

    return cochain



def integrate_kforms(kform, phi, b, det, subdivision_vert, subdivision_coefs, num_sub, l =1, dim = 3, k = 2):
    """ Integrate multiple 2-forms over a 2-simplex
    --------
    Input: 
        kform: l 2-forms
        phi: a matrix containing the embedding of the vertices of the simplex in R^dim
        b: a vector containing the embedding of the basepoint of the simplex in R^dim
        det: the tensor computed by build_determinant_tensor
        l: the number of 2-forms to integrate
        dim: the dimension of the embedding space
        k: the dimension of the form
        subdivision_vert: a list of vertices of the subdivision of the simplex
        subdivision_coefs: a list of coefficients for the contribution of each vertex to the integral
        num_sub: the number of subdivisions of the simplex 
    Output:
        integral: the value of integral of the l 2-forms over the 2-simplex 
    """

    ## number of vertices in the subdivision
    num_simplices = int(num_sub**2) ## number of simplices in the subdivision

    N = int(scipy.special.binom(dim,k))

    integrals = torch.zeros(l,1) 

    for i in range(l):
        ## integrate the i-th k-form
        for j in range(len(subdivision_vert)): 
            p = subdivision_vert[j]
            phi_p = phi.T @ p + b
            g_p = torch.tensor([0]).float() 
            for ind in range(i*N, (i+1)*N):
                m = ind - i*N
                #print('m:',m)
                #print('ind: ',ind)
                #print('kform: ',kform(phi_p)[ind])
                
                g_p+= kform(phi_p)[ind] * torch.matmul(torch.matmul(phi[0], det[m]), phi[1].T)
            
            cof = subdivision_coefs[j]


            integrals[i] += torch.mul(g_p,cof)

        vol = torch.tensor([1/(2*(num_sub**2))]).float()
        
    return   ((vol/num_simplices)*integrals).float()

def forms2cochains(kform,surface_dict, deter_tensor, subdivision_vert, subdivision_coef, num_sub, l = 1, dim = 3, k = 2):
    """ 
    From l 2-forms we get l cochains on the complex by integrating the forms over every 2-simplex in the simplicial complex

    Input:
        kform: l 2-forms
        surface_dict: a dictionary containing all of the information of the triangulated surface in R^dim
        deter_tensor: the tensor computed by build_determinant_tensor 
        dim: the dimension of the embedding space
        k: the dimension of the form
        subivision_vert: a list of vertices of the subdivision of the simplex
        subdivision_coefs: a list of coefficients for the contribution of each vertex to the integral
        num_sub: the number of subdivisions of the simplex for the approximation of the integral
    
    Output:
        cochains: l cochains on the simplicial complex
    """


    Emb_comp = surface_dict['points']

    assert Emb_comp.shape[1] == dim, "The dimension of the embedding space is not equal to the number of columns of the matrix Emb_comp"
    
    cochains = torch.zeros(len(surface_dict['simplices']),l)   
    
    #deter_tensor = build_determinant_tensor(dim, k)
    for i in range(len(surface_dict['simplices'])):
       
        phi_simplex = surface_dict['Phi'][i]
        b_simplex = surface_dict['b'][i]
        cochains[i] = integrate_kforms(kform, phi_simplex, b_simplex,deter_tensor, subdivision_vert, subdivision_coef, num_sub, l, dim, k)[:,0].float()
    return cochains.float()

