
import numpy as np
import matplotlib as mpl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import math
import gudhi as gd 
import scipy.special

# import the package for 3d plotting
from mpl_toolkits.mplot3d import Axes3D

## k-form integratiton functions: FILL INNN

def subdivide_simplex_torch(n):
    # create a list of vertices
    vertices = []
    for i in range(n+1):
        for j in range(n+1-i):
            vertices.append([i/n,j/n])
    vertices = torch.tensor(vertices)
    return vertices

# check if a point is strictly the interior of a triangle with vertices (0,0), (1,0), (0,1)
def is_interior(point):
    if point[0] > 0 and point[1] > 0 and point[0] + point[1] < 1:
        return True
    else:
        return False
    

# check if a point is striclty on the intetior one of the edges of a triangle with vertices (0,0), (1,0), (0,1)
def is_edge(point):
    if point[0] == 0 and point[1] > 0 and point[1] < 1:
        return True
    elif point[0] == 1 and point[1] > 0 and point[1] < 1:
        return True
    elif point[0] > 0 and point[0] < 1 and point[1] == 0:
        return True
    else:
        return False

#check if a point is a vertex of a triangle with vertices (0,0), (1,0), (0,1)
def is_vertex(point):
    if point[0] == 0 and point[1] == 0:
        return True
    elif point[0] == 1 and point[1] == 0:
        return True
    elif point[0] == 0 and point[1] == 1:
        return True
    else:
        return False
    
def sum_points(n, g):
    """ sums the values of g over all vertives in the subdivided simplex"""
    val = 0
    vertices = subdivide_simplex_torch(n)
    for pt in vertices:
        if is_interior(pt):
            val += 6 * g(pt)
        if is_edge(pt):
            val += 3 * g(pt)
        if is_vertex(pt):
            val += g(pt)

    vol = 1/(2*(n**2))
    num_triangles = n**2 
    return  (val* vol )/num_triangles


def build_PHI(phi): 
    """Input: phi an array containing the embedding values of the vertices of a simplex
    Output: PHI a linear map that can map all points inside a simplex into the embedding space """
    PHI = torch.zeros((phi.shape[0]-1, phi.shape[1]))
    for i in range(1,phi.shape[0]):
        PHI[i-1] = phi[i] - phi[0]
    #return transpose of PHI
    return PHI.transpose(0,1)

def build_big_deter_tensor(n, k =2):
    """Input: n the number of subdivision of the simplex
            k the dimension of the simplex, default is 2 for now 
        Output: Tensors of size (n choose k) x n x n that can be used to compute the determinant entering in the integration formuala
        of  k-form over an embedded simplex
    """
    N = int(scipy.special.binom(n,k))
    deter_tensor = torch.zeros(N, n, n)
    ind = 0
    for i in range(n):
        for j in range(i+1,n):
            # each tensor is a nxn matrix with (i,j)-coordinate equal to 1 and (j,i)-coordinate equal to -1
            deter_tensor[ind,i,j] = torch.tensor(1)
            deter_tensor[ind,j,i] = torch.tensor(-1)
            ind += 1

    return deter_tensor

def g(p, kform, phi, deter_tensor):
    """Input: p a point in the simplex, 
              kform: a NN of the k-form to integrate 
              phi: an array containing the embedding values of the vertices of a simplex
              deter_tensor: a tensor of size (n choose k) x n x n that can be used to compute the determinant entering in the integration formuala
              of  k-form over an embedded simplex
        Output: the coefficients of the k-form evaluated at p
              """
    PHI = build_PHI(phi)
    p_emb = PHI @ p
    k_form_val = kform(p_emb)

    # make a vector containing all values of det 
    det = torch.zeros(deter_tensor.shape[0])

    for i in range(deter_tensor.shape[0]):
        #det = PHI[0].transpose(0,1) @ deter_tensor[i] @ PHI[1]
        #print(PHI[:,0].unsqueeze(1).transpose(0,1).shape)
        #print(deter_tensor[i].shape)
        #print(PHI[:,1].unsqueeze(1).shape)
        det[i] = torch.matmul(torch.matmul(PHI[:,0].unsqueeze(1).transpose(0,1), deter_tensor[i]), PHI[:,1].unsqueeze(1))

    g_val = k_form_val * det
    g_p = torch.sum(g_val)
    return g_p

def integral_simplex(kform, phi, deter_tensor, n):
    """Input: kform: a NN of the k-form to integrate
                phi: an array containing the embedding values of the vertices of a simplex
                deter_tensor: a tensor of size (n choose k) x n x n that can be used to compute the determinant entering in the integration formuala
                of  k-form over an embedded simplex
                n: the number of subdivision of the simplex
                pt: a point in the simplex ## why??
        Output: the integral of the k-form over the simplex
    """
    vertices = subdivide_simplex_torch(n)
    val = 0
    # compute g at each vertex
    for pt in vertices:
        if is_interior(pt):
            cof = 6
        elif is_edge(pt):
            cof = 3
        else: 
            cof = 1

        val += cof * g(pt, kform, phi, deter_tensor)

    vol = 1/(2*(n**2))

    num_triangles = n**2 

    return  (val* vol )/num_triangles

def integral_SC(kform, phi, simplices, dim, num_sub):
    """Input: kform: a NN of the k-form to integrate
                phi: an array containing the embedding values of the vertices of the complex
                simplices: a list of k-simplices in the complex
                dim: the embedding dimension 
                num_sub: the number of subdivision of each simplex
        Output: the integral of the k-form over the simplicial complex """

    deter_tensor = build_big_deter_tensor(dim,2)
    #print(deter_tensor) 
    val = 0
    vertices = subdivide_simplex_torch(num_sub)
    for simplex in simplices:
        phi_simplex = torch.zeros((len(simplex), dim))
        phi_simplex = phi[torch.tensor(simplex)]
        #print(phi_simplex)
        for v in vertices:
            val += integral_simplex(kform, phi_simplex, deter_tensor, num_sub, v)
    return val

def form2cochain(kform, phi, simplices, dim, num_sub): 
    cochain = torch.zeros(len(simplices))
    deter_tensor = build_big_deter_tensor(dim,2) ## maybe this computation can be done outside the function 
    for i in range(len(simplices)):
        simplex = simplices[i]
        phi_simplex = torch.zeros((len(simplex), dim))
        phi_simplex = phi[torch.tensor(simplex)]
        cochain[i] = integral_simplex(kform, phi_simplex, deter_tensor, num_sub)

    return cochain


########### SURFACE GENERATION #########

## generating random surfaces in 3D  
def random_surface_yz(n, eps = 0.1):

    x = np.sort(np.random.uniform(-10, 10, n))
    y = np.sort(np.random.uniform(-10, 10, n))
    Eps = np.random.uniform(-eps,eps, n)    
    X, Y = np.meshgrid(x, y)
    Z = Y + Eps
    # return an array of points
    ar = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    ar = torch.from_numpy(ar)
    return X, Y, Z, ar

def random_surface_xz(n, eps = 0.1):

    x = np.sort(np.random.uniform(-10, 10, n))
    y = np.sort(np.random.uniform(-10, 10, n))
    Eps = np.random.uniform(-eps,eps, n)
    #EPS = np.meshgrid(Eps, Eps)
    
    X, Y = np.meshgrid(x, y)
    Z = X + Eps
    # return an array of points
    
    ar = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    ar = torch.from_numpy(ar)

    return X, Y, Z, ar

def random_curved_surface(n, eps = 0.1): 

    x = np.sort(np.random.uniform(-10, 10, n))
    y = np.sort(np.random.uniform(-10, 10, n))
    X, Y = np.meshgrid(x, y)

    Eps = np.random.uniform(-eps,eps, n)
    EPS = np.meshgrid(Eps, Eps)
    Z = X**2 + Y**2 + Eps 

    ar = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    ar = torch.from_numpy(ar)

    return X, Y, Z, ar
    

def generate_surfaces_yz(num_surf, n_pts , eps): 
    surfaces = []
    for i in range(num_surf):
        surfaces.append(random_surface_yz(n_pts, eps))

    return surfaces

def generate_surfaces_xz(num_surf, n_pts , eps):
    surfaces = []
    for i in range(num_surf):
        surfaces.append(random_surface_xz(n_pts, eps))

    return surfaces

def generate_curved_surfaces(num_surf, n_pts , eps):
    surfaces = []
    for i in range(num_surf):
        surfaces.append(random_curved_surface(n_pts, eps))

    return surfaces

## plotting function for a surface in 3D 
def plot_surface(X,Y,Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # change the view
    #ax.view_init(60, 35)
    ax.view_init(40, 60)
    plt.show()
    # plot the contour in a plot that I can rotate
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='viridis')
    ax.set_title('Contour plot')
    ax.view_init(60, 35)
    plt.show()
    
    return None
        


# subdivide unit square and return a single array of points
def unit_square_grid(n=3):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    return np.array([X.flatten(), Y.flatten()]).T

## return the alpha complex of a unit square grid
def unit_square_grid_complex(n=3):
    pts = unit_square_grid(n)
    ac = gd.AlphaComplex(pts).create_simplex_tree()
    return ac

# plot and return the alpha complex of a unit square grid
def plot_unit_square_grid(n=3):
    pts = unit_square_grid(n)
    ac = gd.AlphaComplex(pts).create_simplex_tree()
    plt.scatter(pts[:,0], pts[:,1])

    for s in ac.get_skeleton(2):
        if len(s[0]) == 2: 
            plt.plot(pts[s[0], 0], pts[s[0], 1], 'r-')
    return pts, ac

