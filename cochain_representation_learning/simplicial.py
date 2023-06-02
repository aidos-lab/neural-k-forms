"""Functions for dealing with simplicial complexes."""

import numpy as np


def vectorize(cocycle, simplices):
    """Turn Ripser-type cocycle into a vectorized cochain.

    Parameters
    ----------
    cocycle: np.array
        cocycle in the form of a list of [i,j,val] where i,j are the
        vertices and val is the value of the cocycle on the edge between
        i and j

    Returns
    -------
    cocycle_vec: np.array
        A cochain vectorized over the edges of the simplicial complex
        which agrees with the indexing in simplices
    """

    for k in range(len(cocycle)):
        [i, j, val] = cocycle[k, :]

        # correct orientation

        if i > j:
            i, j = j, i
            val = -val
            cocycle[k, :] = [i, j, val]

    # vectorize cocycle
    cocycle_vec = np.zeros(len(simplices[1]))
    for k in range(cocycle.shape[0]):
        [i, j, val] = cocycle[k, :]

        # check if edge is in simplices[1], if so, add value to vector
        # this is because we may need to restrict the cocycle to a subcomplex
        if frozenset([i, j]) in simplices[1].keys():
            cocycle_vec[simplices[1][frozenset([i, j])]] = val

    return cocycle_vec


def devectorize(projection, simplices):
    """Create Ripser-type cochain from a vectorized cochain.

    Parameters
    ----------
        projection: np.array
            vectorized cocycle

    Returns
    -------
    cocycle: np.array
        Cocycle in the form of a list of [i,j,val] where i,j are the
        vertices and val is the value of the cocycle on the edge between
        i and j
    """

    c = np.zeros((len(simplices[1]), 3))
    for k in range(len(simplices[1])):
        [i, j] = np.sort(list(list(simplices[1])[k]))
        c[k, :] = [i, j, projection[k]]

    return c
