
import numpy    as np
import pandas   as pd
import xarray   as xr

from scipy.linalg  import norm
from hypergraph_centralities.tensor_operations import edge_list_to_tensor, apply_parallel, is_irreducible


def H_centrality( T :       xr.DataArray,
                  tol:      float=1e-4,
                  maxiter:  int=100,
                )        -> pd.Series:
    """
    Implementation of an iteration algorithm that calculates an m-uniform hypergraph's leading H-eigenvector. That
    eigenvalue is interpreted as the H-centrality of the hypergraph, see details below.

    input:
    -----
    T:          An m-order n-dimensional tensor
    tol:        Threshold of relative change across vector components that indicates convergence.
    maxiter:    If algorithm hasn't converged in less than maxiter iterations, an error is raised.

    return:
    ------
    c:          leading eigenvector, interpreted as H-centralities

    details:
    -------
    Let T be the adjacency tensor of a hypergraph. Its H-eigenvector vector is the positive real vector c satisfying
        T c^{m-1} = lam * c^m with ||c||_1=1
    for some eigenvalue lam > 0. See [1,2] for more details. Here T c^{m-1} is short-hand notation for the apply-
    transformation (see tensor_opterations.py for an implementation thereof).

    To obtain c, we apply an iteration procedure akin to the power-method to obtain leading eigenvalues of matrices.
    See [1] for a details and proof of convergence and [2] for examples.

    example:
    -------
    As an example, we consider a k-uniform sunflower hypergraph with r petals (see Figure 1 in [2] for a visualization).
    Conveniently, the sunflower hypergraph allows for an analytical calculation of the ratio between centrality of the
    most central node and all other ones (see [2] for a derivation).  Here, we compare numerical and analytical
    implementation.

    >>> m, r   = 4, 5
    >>> T      = generate_sunflower_HG( m=m, r=r )
    >>> c      = H_centrality( T )
    >>> r_n    = c.iloc[-1] / c.iloc[0]
    >>> r_a    = r**(1/m)
    >>> print('analytical equal numerical?', np.isclose(r_n, r_a, atol=1e-4)) # indeed, the ratios are the same

    references:
    ----------
    [1] 2010 - Ng et al. - Finding the largest eigenvalue of a nonnegative tensor
    [2] 2019 - Benson - Three hypergraph eigenvector centralities
    """

    # check that input is provided in correct format
    ####################################################################################################################
    assert isinstance( T, xr.DataArray ),                 'tensor T must be an xarray'
    for dim in T.shape: assert dim==T.shape[0],           'T must be m-uniform, n-dimensional tensor'
    assert len(T.shape) >= 2,                             'T must be at least 2-uniform'
    indices = list(T.indexes.values())                    # indices along each dimension
    for ind in indices: assert ind.equals(indices[0]),    'indices along each dimension must be the same'
    assert is_irreducible(T),                             'T must be irreducible'

    # intialize some basic variables
    ####################################################################################################################
    m     = len( T.shape )                              # tensor dimensionality
    n     = T.shape[0]                                  # number of values along each dimension
    c     = pd.Series( np.ones(n)/n, index=indices[0])  # random initial vector with all positive values and ||c||_1 = 1
    y     = apply_parallel( T, c )                      # first iteration

    # iterate the apply-transform until convergene is reached (see Theorem 2.4 in [1]).
    ####################################################################################################################
    for i in range(maxiter):

        y_scaled  = y**(1/(m-1))                       # rescale the apply-transform
        c         = y_scaled / norm(y_scaled, 1)       # renormalize to get next interation of eigenvector approximation
        y         = apply_parallel( T, c )             # apply next transform
        s         = y / c**(m-1)                       # difference between current candiate and rescaled next iteration
        converged = (max(s) - min(s)) / min(s) < tol   # are all entries in y and c^{m-1} roughly the same?

        if converged:  return c                        # stop iterating once converged

    raise ValueError(f'No convergence after {maxiter}-iterations. Note that T must be irreducible.')


def generate_sunflower_HG(  m : int=4,
                            r : int=5,
                            ) -> xr.DataArray:
    """
    This function returns the adjacency tensor of an m-uniform sunflower hypergraph with r petals. See Figure 1 in [1]
    for a visualization of this graph. This graph is useful to test the implementation of the centrality measures.

    [1] 2019 - Benson - Three hypergraph eigenvector centralities
    """

    nr_nodes  = (m-1)*r + 1                            # number of nodes of sunfolder (+1 accounts for core)
    last_node = nr_nodes - 1                           # name of core node, its the last one, we start counting from 0
    edges     = np.arange(nr_nodes-1)                  # all nodes except core
    edges     = np.array_split(edges, r)               # split into r petals (no core attached yet)
    edges     = [ list(e)+[last_node] for e in edges ] # append core to every petal
    edges     = np.array(edges)                        # requred format for edge_list_to_tensor
    T         = edge_list_to_tensor( edges )           # reshape into tensor

    return T
