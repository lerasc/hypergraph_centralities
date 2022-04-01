
import numpy  as np
import pandas as pd
import xarray as xr


def edge_list_to_tensor( edges:np.ndarray  ):
    """
    Given a list of edges representing a k-uniform hypergraph, this function returns the associated k-dimensional
    adjacency tensor. We use xarray to represent tensors. See [1] for good tutorials on xarray.

    input:
    -----
    edges:  Numpy array of size (n,k), representing n hyper-edges of size k. Each element represents the name of a node.

    return:
    ------
    T:      k-dimensional tensor representing the hypergraph's adjacency matrix.

    references:
    ----------
    [1] https://github.com/xarray-contrib/xarray-tutorial

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
    """

    # check that input is provided in correct format
    ####################################################################################################################
    assert isinstance(edges, np.ndarray), 'edges must be numpy array'
    assert len(edges.shape)==2, 'edges must be two-dimensional array'

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


def generate_sunflower_HG( k:int=4, r:int=5 ):
    """
    This function returns the adjacency tensor of a k-uniform sunflower hypergraph with r petals. See Figure 1 in [1]
    for a visualization of this graph.

    [1] 2019 - Benson - Three hypergraph eigenvector centralities
    """

    nr_nodes  = (k-1)*r + 1                            # number of nodes of sunfolder (+1 accounts for core)
    last_node = nr_nodes - 1                           # name of core node, its the last one, we start counting from 0
    edges     = np.arange(nr_nodes-1)                  # all nodes except core
    edges     = np.array_split(edges, r)               # split into r petals (no core attached yet)
    edges     = [ list(e)+[last_node] for e in edges ] # append core to every petal
    edges     = np.array(edges)                        # requred format for edge_list_to_tensor
    T         = edge_list_to_tensor( edges )           # reshape into tensor

    return T