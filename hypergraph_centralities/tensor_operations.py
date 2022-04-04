
import numpy  as np
import pandas as pd
import xarray as xr

from itertools import product


def edge_list_to_tensor( edges:np.ndarray  ) -> xr.DataArray:
    """
    Given a list of edges representing a k-uniform hypergraph, this function returns the associated k-dimensional
    adjacency tensor.

    input:
    -----
    edges:  Numpy array of size (n,k), representing n hyper-edges of size k. Each element represents the name of a node.

    return:
    ------
    T:      k-dimensional tensor representing the hypergraph's adjacency matrix.


    details:
    -------
    -   We use xarray to represent tensors. See [1] for tutorials on how to work with xarray.

    example:
    -------
    The following code defines a 3-uniform hypergraph with a total of 5 nodes

    >>> edges = np.array([  ['a','b','c'],
    >>>                     ['b','c','d'],
    >>>                     ['c','d','e'],
    >>>                     ['d','e','b'],
    >>>                                 ])
    >>> T = edge_list_to_tensor( edges  )

    Now T is a 5x5x5 Tensor with entry 1 where there is an edge and 0 else. The total number of 1 is equal to 4 since
    there are 4 edges. Consistency checks:

    >>> T.sel(dim_1='a', dim_2='b', dim_3='c').values  # = 1 since there is an (a,b,c)-edge
    >>> T.sel(dim_1='a', dim_2='b', dim_3='e').values  # = 0 since there is no (a,b,e)-edge

    references:
    ----------
    [1] https://github.com/xarray-contrib/xarray-tutorial
    """

    # check that input is provided in correct format
    ####################################################################################################################
    assert isinstance(edges, np.ndarray), 'edges must be numpy array'
    assert len(edges.shape)==2, 'edges must be two-dimensional array'

    # transform list of edges into an m-uniform n-dimensional tensor.
    ####################################################################################################################
    n, k    = edges.shape                                               # n = nr of edges, degree of edges
    nodes   = list(np.unique(edges.flatten()))                          # list of all nodes
    mi      = pd.Series( 1, index=pd.MultiIndex.from_arrays(edges.T))   # multi-index, each edge is an index
    ar      = xr.DataArray.from_series(mi)                              # create k-dimensional array
    names   = dict([(f'level_{k}',f'dim_{k+1}') for k in range(k)])     # rename from level_0 to dim_1 etc.
    ar      = ar.rename(names)                                          # rename from level_0 to dim_1 etc.
    indices = dict([(f'dim_{k+1}',nodes) for k in range(k)])            # make each dimension of same shape
    ar      = ar.reindex(indices)                                       # make each dimesnion of same shape
    ar      = ar.fillna(0)                                              # repalce NaN by 0

    return ar


def apply( T :  xr.DataArray,
           x :  pd.Series,
           ) -> pd.Series:
    """
    Implementation of the 'apply'-operation which is important to calculate different hypergraph centrality measures.
    See details below for a definition.

    input:
    -----
    T:          An m-order n-dimensional tensor
    x:          An n-dimensional vector (the index of must match the index of :param T:)

    return:
    ------
    y:          an n-dimensional vector y := T x^{m-1}, see details below.

    details:
    -------
    Consider an m-order n-dimensional tensor T with components
        T_{j_1, j_2, ..., j_m} where j_i \in {1, 2, .., n}                                                 (1)
    and an n-dimensional vector x with components x_i.
    The apply function, abbreviated
        y = T x^{m-1}                                                                                      (2)
    results is an n-dimensional vector y with the i-th component given by
        y_i = sum_{j_2, ..., j_m=1}^n T_{i, j_2, ..., j_m} * x_{j_2} * ...* x_{j_m},                       (3)
    see for instance equation (1.2) in paper [1] for details.
    It is straight forward to verify that the case m=2 corresponds to the standard matrix-vector
    multiplication. For m=3, the transformation reads
        y_i = sum_{j, k = 1}^n T_{i,j,k} x_i x_k.                                                          (4)


    example 1:
    ---------
    As a first example, we consider a 2-uniform tensor, i.e. a matrix. In that case, the apply function is the same
    as a matrix-vector multiplication. The following test confirms this:

    >>> edges = np.array([  ['a','b'],
    >>>                     ['b','c'],
    >>>                     ['c','d'],
    >>>                     ['a','d'],
    >>>                             ])
    >>> T  = edge_list_to_tensor( edges  )
    >>> x  = pd.Series([ 1, 2, 3, 4 ], index=['a','b','c','d'])
    >>> y1 = apply(  T, x ).values # special case of the apply-function for 2-uniform graph
    >>> y2 = np.dot( T, x )        # simple matrix vector multiplication
    >>>
    >>> print('y1=y2?', np.equal(y1,y2).all()) # indeed, y1 is the same as y2

    example 2:
    ---------
    As a second example, we consider a 3-uniofrm tensor. In that case, we can also implement equation (3) more
    explicitly via equation (4). Here, we compare that the results are the same.

    >>> edges = np.array([  ['a','b','c'],
    >>>                     ['b','c','d'],
    >>>                     ['c','d','e'],
    >>>                     ['d','e','b'],
    >>>                     ['a','c','d'],
    >>>                                 ])
    >>> T     = edge_list_to_tensor( edges  )
    >>> x     = pd.Series([ 1, 2, 3, 4, 5 ], index=['a','b','c','d','e'])
    >>> y1    = apply( T, x ).values # apply-transform (3)
    >>> combs = list(product( range(5), range(5) )) # all parameter combinations (j,k) to sum over
    >>> y2    = np.array([ sum([ T[i,j,k] * x[j] * x[k] for (j,k) in combs ]) for i in range(5) ]) # equation (4)
    >>>
    >>> print('y1=y2?', np.equal(y1,y2).all()) # indeed, y1 is the same as y2

    references:
    ----------
    [1] 2010 - Ng et al. - Finding the largest eigenvalue of a nonnegative tensor
    """

    # check that input is provided in correct format
    ####################################################################################################################
    assert isinstance( T, xr.DataArray ),                 'tensor T must be an xarray'
    for dim in T.shape: assert dim==T.shape[0],           'T must be m-uniform, n-dimensional tensor'
    assert len(T.shape) >= 2,                             'T must be at least 2-uniform'
    indices = list(T.indexes.values())                    # indices along each dimension
    for ind in indices: assert ind.equals(indices[0]),    'indices along each dimension must be the same'

    assert isinstance( x, pd.Series ),                    'input vector x must be a pandas Series'
    assert x.index.equals( indices[0] ),                  'x must have same index as T'

    # intialize some basic variables
    ####################################################################################################################
    m     = len( T.shape )                                      # tensor dimensionality
    n     = T.shape[0]                                          # number of values along each dimension
    y     = np.nan * np.zeros(n)                                # initialize result of apply transform
    combs = list(product(*[ range(n) for _ in range(m-1) ]))    # all index-combinations to sum across

    # iterate each component and calculate the apply-function, cf. equation (3)
    ####################################################################################################################
    for i in range(n):                                               # calculate each component of T x^{m-1}

        sT   = T[i].values                                           # sub-tensor T_{i, :}
        y[i] = sum([                                                 # sum across all index constellations
                        sT[comb]                                     # prefactor T_{i, j_2, ..., j_m}
                        *
                        np.product([ x.iloc[l] for l in comb ])      # x_{j_2} * ...* x_{j_m}
                        for comb in combs                            # interate all index-combinations j_2 ... j_m
                        ])

    y = pd.Series(y, index=x.index)                                  # add names as index

    return y


def generate_sunflower_HG(  m : int=4,
                            r : int=5,
                            ) -> xr.DataArray:
    """
    This function returns the adjacency tensor of an m-uniform sunflower hypergraph with r petals. See Figure 1 in [1]
    for a visualization of this graph.

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