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


def generate_cochain_data_matrix(vf, chain, d=5):
    """Generate a cochain data matrix from a chain and a vector field.

    Parameters
    ----------
    vf : a Pytorch Sequential object
        The vector field to be applied to the chain

    chain : a torch tensor of shape (r,2,n)
        The chain to be turned into a cochain data matrix

    d : int
        The number of steps for the discretization of the chain

    Returns
    -------
    out : a torch tensor of shape (r,c)
        The cochain data matrix
    """

    # discretize the chain
    chain = discretize_chain(chain, d)

    # number of simplicies
    r = chain.shape[0]

    # number of discrete steps
    d = chain.shape[1]

    # dimension of ambient space
    n = chain.shape[2]

    # number of feature-cochains in the cochain data matrix
    c = int(vf[-1].out_features / n)

    # apply the vector field to the discretized chain
    out = vf(chain).reshape((r, d, n, c))

    # calculate the simplex gradients
    simplex_grad = chain[:, 1, :] - chain[:, 0, :]

    # swap dimensions n and c in out
    out = out.permute(0, 1, 3, 2)

    # calculate the inner product of the vector field and the simplex
    # gradients at each discrete step on each simplex
    inner_prod = torch.matmul(out, simplex_grad.T / (d - 1))

    # take diagonal of out3 along axis 0 and 3 (this corresponds to
    # correcting the broadcasted multplication effect)
    inner_prod = torch.diagonal(inner_prod, dim1=0, dim2=3)

    # permute dimensions 0 and 2 of out4
    inner_prod = inner_prod.permute(2, 0, 1)

    # apply the trapezoidal rule to the inner product
    cdm = (inner_prod[:, 1:, :] + inner_prod[:, 0:-1, :]) / 2
    cdm = cdm.sum(axis=1)

    return cdm
