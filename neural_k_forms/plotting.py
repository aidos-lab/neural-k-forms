"""Plotting routines for cochains."""

import numpy as np
import matplotlib.pyplot as plt

import torch


def plot_surface(X, Y, Z, title=None):
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
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    ax.view_init(40, 60)

    if title is not None:
        ax.set_title(title)

    plt.show()
    return None


def plot_component_vector_field(f, ax, comp=0, x_range=5, y_range=5):
    """Plot a component of a vector field given by a function f: R^2 -> R^2.

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
    x = np.linspace(-x_range, x_range, 20)
    y = np.linspace(-y_range, y_range, 20)
    X, Y = np.meshgrid(x, y)

    X = torch.tensor(X).double()
    Y = torch.tensor(Y).double()

    U = np.zeros((20, 20))
    V = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            inp = np.array([X[i, j], Y[i, j]])
            inp = torch.tensor(inp).float()

            tv = f.forward(inp).reshape(2, -1)

            U[i, j] = tv[:, comp][0]
            V[i, j] = tv[:, comp][1]
    ax.quiver(X, Y, U, V)
