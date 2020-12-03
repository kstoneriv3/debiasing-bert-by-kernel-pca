import numpy as np
import torch
import scipy
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_kernels
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import _check_psd_eigenvalues

from src.debiasing.utils import get_design_matrix, is_sorted
from src.debiasing.numpy_kpca import _fix_index_if_none

DTYPE = torch.float64
DEVICE = torch.device('cuda')  # 6.7x speed up by gpu
#DEVICE = torch.device('cpu')

# gamma=0.14 seems to work well for the BERT embeddings.
def TorchRBF(X, Y, gamma=0.14):
    gamma = 1. / X.shape[-1] if (gamma is None) else gamma
    K = torch.exp( - gamma * torch.sum((X - Y)**2, axis=-1))
    return K

class TorchDebiasingKernelPCA:
    """Kernel PCA with mean-centering in feature space within each group.
    
    The design of this class is mostly based on sklearn.decomposition.KernelPCA.  
    
    Parameters
    ----------
    n_components : int, default=None
        Number of components. If None, all non-zero components are kept.    
    kernel: string or callable
        The kernel to use when calculating kernel between instances in a
        feature array. Valid string values for kernel are:
        ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'],
        which must be one of the kernels in sklearn.pairwise.PAIRWISE_KERNEL_FUNCTIONS.
        If callable is passed, kernel must have following arguments:
        - X: array [n_sample_a, n_features]
        - Y: array [n_sample_b, n_features]
        - torch: bool, whose Default is set `False`.
        When `torch` is set `True`, the kernel must return torch.Tensor which is differentiable w.r.t. X and Y.   
    coef0 : float, default=1
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.
    kernel_params : mapping of string to any, default=None
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.
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

    Remark
    ------
    The design of this class is based on the sklearn.decomposition.KernelPCA.
    """
    def __init__(self, n_components=2, *, kernel=TorchRBF):
        self.n_components = n_components
        self.kernel = self.add_autoreshape(kernel)
        self.X_fit_ = None
        self.DX_fit_ = None
        self.alphas_ = None
        self.lambdas_ = None

    @staticmethod 
    def add_autoreshape(kernel):
        def new_kernel(X, Y=None, autoreshape=True):
            Y = X if Y is None else Y
            if autoreshape:
                return kernel(X[:, None, :], Y[None, :, :])
            else:
                return kernel(X, Y)
        return new_kernel
        
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
        X_index = _fix_index_if_none(X, X_index)
        assert is_sorted(X_index)

        X = torch.tensor(X, dtype=DTYPE, device=DEVICE)
        DX = get_design_matrix(X_index)
        DX = _to_torch_sparse(DX)
        K = self.kernel(X)
        
        K = torch.sparse.mm(DX, torch.sparse.mm(DX, K).transpose(0, 1))
        K = K.detach().cpu().numpy()
        
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
        
        self.alphas_ = torch.tensor(self.alphas_[:, mask_non_zero], dtype=DTYPE, device=DEVICE)
        self.lambdas_ = torch.tensor(self.lambdas_[mask_non_zero], dtype=DTYPE, device=DEVICE)
        
        self.X_fit_ = X
        self.DX_fit_ = DX
        
        return self

       
    def transform(self, X):
        """Transform X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        
        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
        """
        assert self.X_fit_ is not None  # Fit the model first before using this method!
        X = torch.tensor(X, dtype=DTYPE, device=DEVICE)
        K = torch.sparse.mm(self.DX_fit_, self.kernel(self.X_fit_, X)).transpose(0,1)
        X_transformed =  K.mm(self.alphas_).mm(torch.diag(1. / torch.sqrt(self.lambdas_)))
        ret = X_transformed.detach().cpu().numpy()
        del X_transformed
        return ret
        
        
    def orth_losses(self, X, X_orth):
        
        n_samples, n_features = X.shape
        n_train = self.X_fit_.shape[0]
        
        X_fit_ = self.X_fit_
        DX_fit_ = self.DX_fit_
        alphas = self.alphas_
        lambdas_inv = torch.diag(1. / self.lambdas_)
        
        Koo = self.kernel(X_orth, autoreshape=False)
        Kox = self.kernel(X_orth, X, autoreshape=False)
        KoP = torch.sparse.mm(self.DX_fit_, self.kernel(X_fit_, X_orth)).transpose(0,1)
        A = KoP.mm(alphas)#; del KoP
        KPx = torch.sparse.mm(self.DX_fit_, self.kernel(X_fit_, X))
        B = self.alphas_.transpose(0,1).mm(KPx)#; del KPx
        
        losses = Koo - 2. * Kox + 2. * A.mm(lambdas_inv).mm(B)
        return losses
        
        
    def debias(self, X, n_iter=30, lr=0.03):
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
        
        X = torch.tensor(X, dtype=DTYPE, device=DEVICE)
        X_orth = torch.tensor(X, dtype=DTYPE, device=DEVICE, requires_grad=True)
        
        optimizer, = torch.optim.SGD(lr=lr, momentum=0, params=[X_orth]), 
        for i in range(n_iter):
            optimizer.zero_grad()
            losses = self.orth_losses(X, X_orth)
            loss = torch.sum(losses)
            loss.backward()
            optimizer.step()
        
        return X_orth.detach().cpu().numpy()
    
    def __del__(self):
        del self.kernel
        del self.X_fit_
        del self.DX_fit_
        del self.alphas_
        del self.lambdas_
        del self


def _to_torch_sparse(M):
    """
    input: M is Scipy sparse matrix
    output: pytorch sparse tensor in GPU 
    """
    M = M.tocoo().astype(np.float64)
    indices = torch.tensor(np.vstack((M.row, M.col)), dtype=torch.long, device=DEVICE)
    values = torch.tensor(M.data, dtype=DTYPE, device=DEVICE)
    shape = torch.Size(M.shape)
    M_torch = torch.sparse_coo_tensor(indices, values, shape, device=DEVICE)
    return M_torch

