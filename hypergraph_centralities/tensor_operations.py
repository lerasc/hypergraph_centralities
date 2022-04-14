
import numpy           as np
import pandas          as pd
import xarray          as xr
import networkx        as nx
import multiprocessing as mp 

from itertools import product, permutations
from functools import partial 


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
    edges   = list(edges)                                               # list of all edges
    edges   = [ np.vstack(list(permutations(e))) for e in edges ]       # form all combinations, i.e. 'undirected edge'
    edges   = np.vstack( edges )                                        # stack back together
    edges   = pd.DataFrame(edges).drop_duplicates(keep='first').values  # drop duplicates
    _, k    = edges.shape                                               # n = nr of edges, degree of edges
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
           i =  None,
           ) -> pd.Series:
    """
    Implementation of the 'apply'-operation y = T x^{m-1} which is important to calculate hypergraph centralities.
    See details below for a definition.

    input:
    -----
    T:          An m-order n-dimensional tensor (e.g. output from edge_list_to_tensor)
    x:          An n-dimensional vector (the index of must match the index of :param T:)
    i:          If None, calculate entry vector y. Else, calculate only the i-th component. This is useful to 
                call the apply-function in parallel for different components (cf. apply_parallel below).

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

    if i is not None: 
        assert i >= 0 and i < T.shape[0],                 'if i is not None, i must not be larger than dimension of y'

    # intialize some basic variables
    #########################,###########################################################################################
    m     = len( T.shape )                                      # tensor dimensionality
    n     = T.shape[0]                                          # number of values along each dimension
    y     = np.nan * np.zeros(n)                                # initialize result of apply transform
    combs = list(product(*[ range(n) for _ in range(m-1) ]))    # all index-combinations to sum across

    # iterate each component and calculate the apply-function, cf. equation (3)
    ####################################################################################################################
    ivals = np.arange(n) if i is None else [i]                       # which components to calculate

    for i in range(n):                                               # calculate each component of T x^{m-1}

        sT   = T[i].values                                           # sub-tensor T_{i, :}
        y[i] = sum([                                                 # sum across all index constellations
                        sT[comb]                                     # prefactor T_{i, j_2, ..., j_m}
                        *
                        np.product([ x.iloc[l] for l in comb ])      # x_{j_2} * ...* x_{j_m}
                        for comb in combs                            # interate all index-combinations j_2 ... j_m
                        ])

    y = pd.Series(y[ivals], index=x.index[ivals])                    # add names as index if all components a

    return y


def apply_parallel(T :  xr.DataArray,
                   x :  pd.Series,
                   ) -> pd.Series:
    """
    Same as apply-function above, but with parallel execution along the components of y. This is useful since for large
    shapes of T, the apply function is very slow.
    """

    ivals    = np.arange(T.shape[0])                         # all components of interest
    func     = partial( apply, *[T,x] )                      # fix all arguments but the component
    nc       = mp.cpu_count() - 1                            # number of cores + 1 reserve
    pool     = mp.Pool(processes=nc)                         # initialize mp instance
    y        = pool.map( func, ivals )                       # run in parallel
    y        = pd.concat(y, axis='index')                    # merge all components into one axis
    _        = pool.close()                                  # close mp instance
    _        = pool.join()                                   # close mp instances

    return y 


def clique_exansion( T : xr.DataArray ) -> pd.DataFrame:
    """
    The clique expansion algorithm construcs a graph from the original hypergraph by replacing each hyperedge with an
    edge for each pair of vertices in the hyperedge [1,2].

    inputs:
    ------
    T:          An m-order n-dimensional tensor (e.g. output from edge_list_to_tensor)

    return:
    ------
    A:          Matrix that can be interpreted as weighted adjacency matrix. The entry at position (i,j) represents
                the number of hyper-edges that encompass both node i and node j.


    references:
    ----------
    [1] Zien et. al. Multi-level spectral hypergraph partitioning with arbitrary vertex sizes. IEEE Transactions on
        Computer-Aided Design of Integrated Circuits and Systems, 18, 1389–1399. (1999)

    [2] Agarwal et. al. Higher order learning with graphs. In Proceedings of the 23rd International Conference on
        achine Learning, 17–24 (2006).
    """

    # check that input is provided in correct format
    ####################################################################################################################
    assert isinstance( T, xr.DataArray ),                 'tensor T must be an xarray'
    for dim in T.shape: assert dim==T.shape[0],           'T must be m-uniform, n-dimensional tensor'
    assert len(T.shape) >= 3,                             'T must be at least 3-uniform'
    indices = list(T.indexes.values())                    # indices along each dimension
    for ind in indices: assert ind.equals(indices[0]),    'indices along each dimension must be the same'

    # get clique-expanded adjacency matrix by summing across all but the first two dimensions
    ####################################################################################################################
    combs   = list(product(indices[0],indices[0]))                            # all node combinations (i,j)
    weights = [ float( T.sel(dim_1=i, dim_2=j).sum() ) for (i,j) in combs ]   # weight = sum across all other dims
    weights = pd.DataFrame( weights, index=pd.MultiIndex.from_tuples(combs) ) # make into DataFrame
    weights = weights.reset_index()                                           # undo multi-index
    weights = weights.pivot(index='level_0', columns='level_1', values=0)     # reshape into adjaccency matrix

    return weights


def get_irreducible_subcomponents( T ):
    """
    The centrality measures are only well-defined if the adjacency tensor is irreducible [1]. Here, we check if the
    tensor is irreducible as follows: First, we use a clique-expansion to turn the hypergraph into a network.
    Subsequently, we check if the associated adjacency matrix is reducible.

    input:
    -----
    T:          An m-order n-dimensional tensor (e.g. output from edge_list_to_tensor)

    output:
    ------
    sub_Ts:     list of fully connected sub-hypergraphs of T.

    references:
    ----------
    [1] 2010 - Ng et al. - Finding the largest eigenvalue of a nonnegative tensor
    """

    A        = clique_exansion( T )                                           # adjacency matrix of clique exp. of T
    G        = nx.from_pandas_adjacency(A)                                    # turn into networkx object
    sub_Gs   = list(nx.connected_components(G))                               # list of all connected components
    dims     = list(T.indexes.keys())                                         # name of all dimensions
    sub_ds   = [ dict([ (dim,list(SG)) for dim in dims ]) for SG in sub_Gs ]  # list of indices of connected components
    sub_Ts   = [ T.sel(sub_dim) for sub_dim in sub_ds ]                       # select connected sub-hyper-graphs

    return sub_Ts
