
import numpy as np
import matplotlib as mpl   
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.tri as mtri

# -----------
# Functions to plot surfaces in 3D
# -----------


def plot_surface(X,Y,Z, title= None):
    """
    Plot a surface in 3D.

    Parameters
    ----------
    X : array
        Array containing the x-coordinates of the points.
    Y : array
        Array containing the y-coordinates of the points.
    Z : array
        Array containing the z-coordinates of the points.
    title : str, optional
        Title of the plot. The default is None.

    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.view_init(40, 60)
    
    if title != None:
        ax.set_title(title)

    plt.show()
    return None

    
def plot_surface_integral(surface, color, simplices, edges, num_pts, title = None, view = None): 
    """
    Plot the surface integral of a 2-form on a surface.

    Parameters
    ----------
    surface : dict
        Dictionary containing the surface data.
    color : array
        Array containing the values of the 2-form on the simplices.
    simplices : list
        List of 2-simplices.
    edges : list
        List of edges.
    num_pts : int
        The grid has num_pts x num_pts points.  
    title : str, optional
        Title of the plot. The default is None.
    view : tuple, optional
        View of the 3D plot. The default is None.
    """

    sx = surface['points'][:,0].reshape(num_pts,num_pts)
    sy = surface['points'][:,1].reshape(num_pts,num_pts)
    sz = surface['points'][:,2].reshape(num_pts,num_pts)

    # plot the surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(len(simplices)): 
        simplex = simplices[i]
        pts = surface['points'][simplex]
        x = pts[:,0]
        y = pts[:,1]
        z = pts[:,2]

        tri = mtri.Triangulation(x, y)
        ax.plot_trisurf(x, y, z, triangles=tri.triangles, color =color, alpha = 0.9)

    
    for e in edges: 
        pts = surface['points'][e]
        pt1 = np.array(pts[0])
        pt2 = np.array(pts[1])
        # plot the edge from the points
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k', alpha = 0.9)
    
    # set axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    if view is not None: 
        ax.view_init(view[0], view[1])
    else:
        ax.view_init(60,10)

    if title is not None: 
        ax.set_title(title)
    else: 
        ax.set_title('Surface plot')

    return None
        

# -----------
# Functions to help with plotting
# -----------


def value2color(val):
    values = val.copy()
    values -= values.min()
    values = values/(values.max()-values.min()) # get values between 0 and 1 !! maybe when we want to compare cochains this is not the best 
    return mpl.cm.bwr(values)

def value2color_all(val, min_val, max_val): 
    value = val.copy()
    value -= min_val
    value = value/(max_val-min_val) 
    return mpl.cm.bwr(value)
