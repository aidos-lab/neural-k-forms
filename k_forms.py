
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
    
def coef_vertex(v): 
    if is_interior(v):
        return torch.tensor([6]).float()
        #return 6
    if is_edge(v):
        return torch.tensor([3]).float()
        #return 3
    if is_vertex(v):
        return torch.tensor([1]).float()
        #return 1

def subdivide_simplex_coef_torch(n):
    # create a list of vertices
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
    dim = embedded_vertices.shape[1] ### check this
    #print(dim)
    a = [embedded_vertices[i] - embedded_vertices[0] for i in range(1,3)]
    phi = torch.stack(a)
    b = embedded_vertices[0] 
    ## should we return the transpose of phi and b?
    return phi.float(), b.float()

def build_determinant_tensor(dim, k =2):
    """Input: n the number of subdivision of the simplex
            k the dimension of the simplex, default is 2 for now 
        Output: Tensors of size (n choose k) x n x n that can be used to compute the determinant entering in the integration formuala
        of  k-form over an embedded simplex
    """
    ### Can this be writen in a more efficient way? i.e. without using a for loop
    
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


def integrate_kform(kform, phi, b, det, subdivision_vert, subdivision_coefs, num_sub, dim = 3, k = 2):
                    
    """ Integrate a $2$-form over $2$-simplex
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

    ## TODO: add assert to check inputs are good 

   
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

    """From a 2-form we get a 2-cochain by integrating the form over every 2-simplex in the simplicial complex
    Input:
        kform: a 2-form
        surface_dict: a dictionary containing all of the information of the triangulated surface in R^dim
        deter_tensor: the tensor computed by build_determinant_tensor
        dim: the dimension of the embedding space
        k: the dimension of the form
        subivision_vert: a list of vertices of the subdivision of the simplex
        subdivision_coefs: a list of coefficients for the contribution of each vertex to the integral
    
    Output:
        cochain: a 2-cochain, torch tensor of shape FILL IN!!!!
     """

    Emb_comp = surface_dict['points']
    assert Emb_comp.shape[1] == dim, "The dimension of the embedding space is not equal to the number of columns of the matrix Emb_comp"

    cochain = torch.zeros(len(surface_dict['simplices']))
    for i in range(len(surface_dict['simplices'])):
     
        phi_simplex = surface_dict['Phi'][i]
        b_simplex = surface_dict['b'][i]
        cochain[i] = integrate_kform(kform, phi_simplex, b_simplex, deter_tensor, subivision_vert, subdivision_coeffs, dim, k)

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

