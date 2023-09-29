
import numpy as np
import matplotlib as mpl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import math
import gudhi as gd 
import scipy.special
from sklearn.decomposition import PCA
import pandas as pd

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.tri as mtri

###Plotting Function
def get_positions(points,simplices):
    polygons = list()
    for i, simplex in enumerate(simplices):
        polygon = list()
        for vertex in simplex:
            polygon.append(points[vertex])
        polygons.append(polygon)
    return polygons



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



def plot_triangles(colors,points,simplices, min_val = None, max_val = None, ax=None, **kwargs):
    triangles = get_positions(points,simplices)
    if ax is None:
        fig, ax = plt.subfigs()

    if min_val is None:
        colors = value2color(colors)
    else:
        colors = value2color_all(colors, min_val, max_val)
    for triangle, color in zip(triangles, colors):
        triangle = plt.Polygon(triangle, color=color, **kwargs)
        ax.add_patch(triangle)
    ax.autoscale()


def plot_edges(ac, pts, ax = None):
    if ax is None:
        fig, ax = plt.subfigs()
        for s in ac.get_skeleton(2):
            if len(s[0]) == 2: 
                ax.plot(pts[s[0], 0], pts[s[0], 1], 'k', alpha = 0.5)
    else:
         for s in ac.get_skeleton(2):
            if len(s[0]) == 2: 
                ax.plot(pts[s[0], 0], pts[s[0], 1], 'k', alpha = 0.5) 


def plot_comparison_cochains(cochains, points, simplices, ac, num_rows): 
    cochain_np= np.array(cochains)
    cochain_np = cochain_np.flatten()
    min_val = cochain_np.min()
    max_val = cochain_np.max()

    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(20, 20))

    for i in range(num_rows):
        ## class 0
        plot_triangles(cochains[i],points,simplices,min_val, max_val, ax=axes[i,0], zorder=1)
        plot_edges(ac,points,ax=axes[i,0])
        ## class 1
        plot_triangles(cochains[-i-1],points,simplices, min_val, max_val, ax=axes[i,1], zorder=1)
        plot_edges(ac,points,ax=axes[i,1])
        
    plt.suptitle('Cochains of surfaces in the dataset', fontsize=16)

    axes[0,0].set_title('Class 0')
    axes[0,1].set_title('Class 1')


def plot_comparison_cochains_indices(cochains, points, simplices, ac, indices0, indices1): 

    assert len(indices0) == len(indices1), "The number of indices for each class must be the same"
    num_rows = len(indices0)

    cochains0 = [cochains[i] for i in indices0]
    cochains1 = [cochains[i] for i in indices1]

    
    cochain_np= np.array(cochains0+cochains1)
    cochain_np = cochain_np.flatten()
    min_val = cochain_np.min()
    max_val = cochain_np.max()

    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(20, 20))

    for i in range(num_rows):
        ## class 0
        plot_triangles(cochains0[i],points,simplices,min_val, max_val, ax=axes[i,0], zorder=1)
        plot_edges(ac,points,ax=axes[i,0])
        ## class 1
        plot_triangles(cochains1[i],points,simplices, min_val, max_val, ax=axes[i,1], zorder=1)
        plot_edges(ac,points,ax=axes[i,1])
        
    plt.suptitle('Cochains of surfaces in the dataset', fontsize=16)

    axes[0,0].set_title('Class 0')
    axes[0,1].set_title('Class 1')



def plot_pca_kform(cochains, labels, n_components, title = None, view = None, save = False, save_path = None): 

    if n_components == 2: 
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(cochains)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = ['principal component 1', 'principal component 2'])

        # add the labels to the principal components
        principalDf['labels'] = labels
        # plot the principal components
        fig, ax = plt.subplots()
        scatter = ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'], c=principalDf['labels'])
        if title != None: 
            ax.set_title(title)
        else:
            ax.set_title('PCA of the cochains')

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Classes")

        if save == True:
            if save_path == None:
                plt.savefig('pca_kform.png')
            else: 
                plt.savefig(save_path)



        plt.show()


    if n_components == 3:

        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(cochains)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

        # add the labels to the principal components
        principalDf['labels'] = labels

        # plot the principal components
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'], principalDf['principal component 3'], c=principalDf['labels'], alpha = 0.7)
        
        if title != None: 
            ax.set_title(title)
        else:
            ax.set_title('PCA of the cochains')
            
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')

        if view != None:
            ax.view_init(view[0], view[1])

        if save == True:
            if save_path == None:
                plt.savefig('pca_kform.png')
            else: 
                plt.savefig(save_path)
        plt.show()




def plot_surface_cochain(surface, color, simplices, edges, title = None, view = None): 

    sx = surface['points'][:,0].reshape(7,7)
    sy = surface['points'][:,1].reshape(7,7)
    sz = surface['points'][:,2].reshape(7,7)

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
        ax.plot_trisurf(x, y, z, triangles=tri.triangles, color =list(color[i])[:3], alpha = 0.9)

    
    for e in edges: 
        pts = surface['points'][e]
        pt1 = np.array(pts[0])
        pt2 = np.array(pts[1])
        # plot the edge from the points
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'k', alpha = 0.9)

    # add the colorbar
    norm = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
    sm = plt.cm.ScalarMappable(cmap=mpl.cm.bwr, norm = norm)
    sm.set_array([])
    fig.colorbar(sm)


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
   

    