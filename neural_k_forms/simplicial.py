"""Functions for dealing with simplicial complexes."""

import numpy as np

from scipy.sparse import coo_matrix


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


# TODO (BR): This is relatively loosely-coupled and assumes the
# existence of a simplex tree. I think we can leave it like this for the
# time being.
def extract_simplices(simplex_tree):
    """Extract simplices from a gudhi simplex tree.

    Parameters
    ----------
    simplex_tree: gudhi simplex tree

    Returns
    -------
    simplices: List of dictionaries, one per dimension d.

    The size of the dictionary is the number of d-simplices. The
    dictionary's keys are sets (of size d + 1) of the 0-simplices that
    constitute the d-simplices. The dictionary's values are the indexes
    of the simplices in the boundary and Laplacian matrices.
    """
    simplices = [dict() for _ in range(simplex_tree.dimension() + 1)]

    for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
        k = len(simplex)
        simplices[k - 1][frozenset(simplex)] = len(simplices[k - 1])

    return simplices


def build_boundaries(simplices):
    """Build unweighted boundary operators from a list of simplices.

    Parameters
    ----------
    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the
        dictionary is the number of d-simplices. The dictionary's keys
        are sets (of size d + 1) of the 0-simplices that constitute the
        d-simplices. The dictionary's values are the indexes of the
        simplices in the boundary and Laplacian matrices.

    Returns
    -------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension: i-th boundary is
       in i-th position
    """
    boundaries = list()
    for d in range(1, len(simplices)):
        idx_simplices, idx_faces, values = [], [], []
        for simplex, idx_simplex in simplices[d].items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1) ** i)
                face = simplex.difference({left_out})
                idx_faces.append(simplices[d - 1][face])
        assert len(values) == (d + 1) * len(simplices[d])
        boundary = coo_matrix(
            (values, (idx_faces, idx_simplices)),
            dtype=np.float32,
            shape=(len(simplices[d - 1]), len(simplices[d])),
        )

        boundaries.append(boundary)

    return boundaries
