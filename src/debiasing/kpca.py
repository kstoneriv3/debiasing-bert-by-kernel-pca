import numpy as np
import scipy
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_kernels
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import _check_psd_eigenvalues

from src.debiasing.utils import get_design_matrix, is_sorted


class DebiasingKernelPCA:
    """Kernel PCA with mean-centering in feature space within each group.
    
    The design of this class is mostly based on sklearn.decomposition.KernelPCA.  
    
    Parameters
    ----------
    n_components : int, default=None
        Number of components. If None, all non-zero components are kept.    
    kernel: string
        The kernel to use when calculating kernel between instances in a
        feature array. Valid string values for kernel are:
        ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'],
        which must be one of the kernels in sklearn.pairwise.PAIRWISE_KERNEL_FUNCTIONS.
    inv_kernel: string
        See kernel. Used for the kernel ridge regression approximation of the pre-imaging.
    alpha : int, default=1.0
        Hyperparameter of the ridge regression that learns the
        inverse transform.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        
    Attributes
    ----------
    lambdas_ : array, (n_components,)
        Eigenvalues of the centered kernel matrix in decreasing order.
        If `n_components` and `remove_zero_eig` are not set,
        then all values are stored.
    alphas_ : array, (n_samples, n_components)
        Eigenvectors of the centered kernel matrix. If `n_components` and
        `remove_zero_eig` are not set, then all components are stored.
    dual_coef_ : array, (n_samples, n_features)
        Inverse transform matrix. Only available when
        ``fit_inverse_transform`` is True.
    X_transformed_fit_ : array, (n_samples, n_components)
        Projection of the fitted data on the kernel principal components.
        Only available when ``fit_inverse_transform`` is True.
    X_fit_ : (n_samples, n_features)
        The data used to fit the model. If `copy_X=False`, then `X_fit_` is
        a reference. This attribute is used for the calls to transform.
    X_fit_index_: (n_samples,)

    Remark
    ------
    The design of this class is based on the sklearn.decomposition.KernelPCA.
    """
    def __init__(self, n_components, *, kernel="linear", inv_kernel=None, 
                 gamma=None, degree=3, coef0=1, kernel_params=None,
                 alpha=1.0, n_jobs=None):
        self.n_components = n_components
        self.kernel = kernel
        self.inv_kernel = kernel if inv_kernel is None else inv_kernel
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.X_fit_ = None

    def fit(self, X, X_index=None):
        """Fit grouped samples in X to extract bias subspace.

        Parameters
        ----------
        X: array [n_samples, n_features]
        X_index: array: [n_samples]
            `X_index` indicates the group to which specific sample belongs to.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        K = _centered_kernel(X=X, X_index=X_index, kernel=self.kernel)

        if self.n_components is None:
            n_components = K.shape[0]
        else:
            n_components = min(K.shape[0], self.n_components)

        # compute eigenvectors
        if K.shape[0] > 200 and n_components < 10:  # use ARPACK
            # initialize with [-1,1] as in ARPACK
            v0 = np.random.RandomState(0).uniform(-1, 1, K.shape[0])
            self.lambdas_, self.alphas_ = scipy.sparse.linalg.eigsh(K, n_components,
                                                which="LA",
                                                tol=0,
                                                maxiter=None,
                                                v0=v0)
        else:
            self.lambdas_, self.alphas_ = scipy.linalg.eigh(
                K, eigvals=(K.shape[0] - n_components, K.shape[0] - 1))
        
        # make sure that the eigenvalues are ok and fix numerical issues
        self.lambdas_ = _check_psd_eigenvalues(self.lambdas_,
                                               enable_warnings=False)

        # flip eigenvectors' sign to enforce deterministic output
        self.alphas_, _ = svd_flip(self.alphas_,
                                   np.zeros_like(self.alphas_).T)

        # sort eigenvectors in descending order
        indices = self.lambdas_.argsort()[::-1]
        self.lambdas_ = self.lambdas_[indices]
        self.alphas_ = self.alphas_[:, indices]

        # remove eigenvectors with a zero eigenvalue (null space) if required
        mask_non_zero = self.lambdas_ > 1e-8
        if sum(mask_non_zero) < self.n_components:
            print("The effective number of the components was smaller than the given `n_components`."
                  "Please be careful about the dimension of the tranformed feature, as it is smaller than `n_components`.")
        self.alphas_ = self.alphas_[:, self.lambdas_ > 1e-8]
        self.lambdas_ = self.lambdas_[self.lambdas_ > 1e-8]
        
        self.X_fit_ = X
        self.X_fit_index_ = X_index
        
        self._fit_inverse_transform(X, None)

        return self

    def _fit_inverse_transform(self, X, X_index=None):
        
        X_transformed = self.transform(X)
        self.X_fit_inverse_ = X
        self.X_transformed_fit_ = X_transformed
        
        # fit inversion
        K = _centered_kernel(X=X_transformed, kernel=self.inv_kernel)
        n_samples = K.shape[0]
        K.flat[::n_samples + 1] += self.alpha
        self.nonlinear_dual_coef_ = scipy.linalg.solve(K, X, sym_pos=True, overwrite_a=True)
        
        # This cannot fit grouped values
        # self.kernel_ridge = KernelRidge(alpha=self.alpha, kernel=self.inv_kernel).fit(X_transformed, X)
        
    def transform(self, X, X_index=None):
        """Transform X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        
        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
        """
        assert self.X_fit_ is not None  # Fit the model first before using this method!
        K = _centered_kernel(X, X_index, self.X_fit_, self.X_fit_index_, self.kernel)
        X_transformed =  (K @ self.alphas_) @ np.diag(1. / np.sqrt(self.lambdas_))
        return X_transformed

    def inverse_transform(self, X_transformed):
        """Transform X back to original space using kernel ridge regression.
        
        Parameters
        ----------
        X_transformed : array-like, shape (n_samples, n_components)
        
        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
        
        References
        ----------
        "Learning to Find Pre-Images", G BakIr et al, 2004.
        """
        assert self.X_fit_ is not None  # Fit the model first before using this method!
        K = _centered_kernel(X=X_transformed, Y=self.X_transformed_fit_, kernel=self.inv_kernel)
        return K @ self.nonlinear_dual_coef_
        # return self.kernel_ridge.predict(X_transformed)
        
    def debias(self, X, X_index=None):
        """Debias the embeddings by reprojection of kernel PCA.
        
        Parameters
        ----------
        X: array [n_samples, n_features]
        method: "ridge" or "optimization"
            When the "ridge" is used, kernel ridge regression is used to approximate the pre-imaging. 
            For details, see the reference. When the "optimization" is used, optimization is used to 
            obtain the pre-image. In this case, the kernel must support the option `torch=True` so 
            that gradient-based optimization can be applied.
            
        References
        ----------
        "Learning to Find Pre-Images", G BakIr et al, 2004.
        """
        assert self.X_fit_ is not None  # Fit the model first before using this method!
        X_tr = self.transform(X, X_index)
        X_orth = X - self.inverse_transform(X_tr)
        return X_orth

def _centered_kernel(X, X_index=None, Y=None, Y_index=None, kernel="linear", filter_params=True, n_jobs=None, **kwds):
    """Compute the group-mean-centered kernel between arrays X and Y.

    This method takes either a vector array and returns a kernel matrix. 
    For the mean centering of the kernel within each group, the index to which 
    each sample belongs to must be privided as ``X_index`` and ``Y_index``. 
    When there is a sample that does not belong to any group, its index 
    must be set ``nan``, and centering is not applied to the sample.

    Parameters
    ----------
    X : array [n_samples_a, n_features]
        A feature array which is sorted accoding to the group index. 
        Note that if the index of an sample is 'np.nan', the sample must come at the last of the array.  
    X_index: integer array [n_samples]
        A sorted array of indices to which samples in X belongs to. When a sample does 
        not belong to any group, its index must be ``np.nan``. 
    Y: array [n_samples_b, 1 + n_features]
        A second feature array which is sorted accoding to the group index. 
        Note that if the index of an sample is 'np.nan', the sample must come at the last of the array.  
    Y_index: integer array [n_samples]
        A sorted array of indices to which samples in Y belongs to. When a sample does 
        not belong to any group, its index must be ``np.nan``. 
    kernel: string or callable
        The kernel to use when calculating kernel between instances in a
        feature array. Valid string values for kernel are:
        ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'],
        which must be one of the kernels in sklearn.pairwise.PAIRWISE_KERNEL_FUNCTIONS.
    filter_params : boolean
        Whether to filter invalid parameters or not.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the kernel function.

    Returns
    -------
    K : array or torch.Tensor of shape [n_samples_a, n_samples_b]
        A mean-centered kernel matrix K such that K_{i, j} is the kernel between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then K_{i, j} is the kernel between the ith array
        from X and the jth array from Y.

    """
    KXY = pairwise_kernels(X, Y, metric=kernel, filter_params=filter_params, n_jobs=n_jobs, **kwds)
    
    if Y is None:
        Y = X
        Y_index = X_index
        
    X_index = _fix_index_if_none(X, X_index)
    Y_index = _fix_index_if_none(Y, Y_index)
    
    assert is_sorted(X_index)
    assert is_sorted(Y_index)
    
    DX = get_design_matrix(X_index)
    DY = get_design_matrix(Y_index)

    KXY_centered = DX @ KXY @ DY
    return KXY_centered

def _fix_index_if_none(X, X_index):
    if X_index is None:
        return np.full(shape=X.squeeze().shape[0], fill_value=np.nan)
    else:
        return X_index
    
def _test_centered_kernel():
    
    D = 500
    N = 1000
    N_grouped = 900
    N_nan = N - N_grouped
    num_groups = 100
    
    # X contains groups of various size and non-centered samples
    X = np.random.randn(N * D).reshape(N, D)
    X_index = np.concatenate([
        np.repeat(np.arange(num_groups), 2),  
        np.random.randint(0, num_groups, N_grouped - 2*num_groups),
        np.full([N_nan], np.nan)
    ])
    X_index.sort()
    
    
    # Y only contains non-centered samples
    Y = np.random.randn(N * D).reshape(N, D)
    Y_index = np.array([np.nan] * N)
    
    Kxx = _centered_kernel(X, X_index, X, X_index, kernel="rbf")
    Kxx_ = _centered_kernel(X, X_index, kernel="rbf")
    assert np.allclose(Kxx, Kxx_)
    assert np.allclose(Kxx, Kxx.T)
    eigvals = np.linalg.eigvalsh(Kxx)
    assert np.all(-1e-8 < eigvals)  # Kernel matrix must be positive definite.
    assert np.sum(1e-8 < eigvals) == (N - num_groups)  # The effective rank of the matrix should be (N - num_groups).
    
    Kxy = _centered_kernel(X, X_index, Y, Y_index, kernel="rbf")
    Kyx = _centered_kernel(Y, Y_index, X, X_index, kernel="rbf")
    assert np.allclose(Kxy, Kyx.T)  # Kernel matrix must be symmetric. 
    
if __name__ == "__main__":
    _test_centered_kernel()
