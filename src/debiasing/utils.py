import numpy as np
import scipy


def get_design_matrix(index):
    """Calculate the design matrix for group-mean-centering
    
    Example:
    Given ``index = [1, 1, 2, 2, np.nan]``, the design matrix becomes
                          [[1/2, 1/2,    0,   0, 0],
                           [1/2, 1/2,    0,   0, 0],
    np.eye(5) - np.array(  [  0,   0,  1/2, 1/2, 0],  )
                           [  0,   0,  1/2, 1/2, 0],
                           [  0,   0,    0,   0, 0]]
    """
    N = index.shape[0]
    mask_nan = np.isnan(index)
    N_nan = sum(mask_nan)
    _, group_counts = np.unique(index[~mask_nan], return_counts=True)
    assert np.all(group_counts > 1)  # There must be more than 2 elements in each group.
    blocks = [np.full([c, c], 1./c) for c in group_counts] + [0.0] * N_nan
    design_matrix = scipy.sparse.eye(N) - scipy.sparse.block_diag(blocks)
    return design_matrix

def is_sorted(index):
    index_ = np.sort(index)
    return np.all((index == index_) | (np.isnan(index) & np.isnan(index_)))

