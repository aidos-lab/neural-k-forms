"""Functions for operating with and on chains."""

import numpy as np

import torch


def path_to_chain(p):
    """Turn a path into a chain.

    Parameters
    ----------
    p : numpy array
        A path in R^n, represented as a numpy array of shape (p,n),
        where p is the number of points in the path

    Returns
    -------
    chain : numpy array
        A chain in R^n, represented as a numpy array of shape (p-1,2,n),
        where p is the number of points in the path. The middle index
        corresponds to start and endpoints of the edges in the chain.
    """
    r = len(p) - 1
    n = p[0].shape[0]

    chain = torch.zeros((r, 2, n))

    chain[:, 1, :] = torch.tensor(p[1:, :])
    chain[:, 0, :] = torch.tensor(p[0:-1, :])

    return chain


def discretize_chain(chain, d):
    """Discretize a chain.

    Parameters
    ----------
    chain : numpy array
        A chain in R^n, represented as a numpy array of shape (p-1,2,n),
        where p is the number of points in the path.

    d : int
        The number of points in the discretized chain

    Returns
    -------
    d_chain : numpy array
        A discretized chain in R^n, represented as a numpy array of
        shape (p-1,d,n), where p is the number of points in the path.
    """
    r = chain.shape[0]
    n = chain.shape[2]

    d_chain = torch.zeros((r, d, n))

    # TODO (BR): I think we can use `torch.linspace` here. Will revisit
    # once the skeleton is up and running again.
    t = np.linspace(0, 1, d)

    for i in range(d):
        d_chain[:, i, :] = (1 - t[i]) * chain[:, 0, :] + t[i] * chain[:, 1, :]

    return d_chain


# TODO (BR): Is this still required? There are some assumptions here
# about the norm that we could maybe turn into parameters.
def path_length(path):
    """Calculate the length of a path."""
    length = 0
    for i in range(len(path) - 1):
        length += np.linalg.norm(path[i + 1] - path[i])
    return length
