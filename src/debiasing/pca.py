import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
import torch
from src.debiasing.utils import get_design_matrix, is_sorted


class DebiasingPCA:

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, X_index):
        """Fit grouped samples in X to extract bias subspace.

        Parameters
        ----------
        X: array [n_samples, n_features]
        X_index: array: [n_samples]
            `X_index` indicates the group to which specific sample belongs to.

        """
        assert is_sorted(X_index)
        self.pca = PCA(n_components=self.n_components)
        X_centered = self.center_linearly(X, X_index)
        self.pca.fit(X_centered)

    def debias(self, X):
        if not self.pca:
            raise NotFittedError("`DebiasingPCA` is not fitted with debiasing set."
                                 " Use `DebiasingPCA.fit` to fit the PCA before using `DebiasingPCA.bebias`.")
        X_tr = self.pca.transform(X)
        X_orth = X - self.pca.inverse_transform(X_tr)
        return X_orth

    def transform(self, X):
        return self.pca.transform(X)

    def transfer_to_torch(self, device):
        self.pca_components_torch = torch.from_numpy(self.pca.components_).float().to(device)
        self.mean_torch = torch.from_numpy(self.pca.mean_).float().to(device)

    def torch_transform(self, X):
        X = X - self.mean_torch
        X_transformed = torch.matmul(X, torch.transpose(self.pca_components_torch, 0, 1))
        return X_transformed

    def torch_inverse_transform(self, X):
        return torch.matmul(X, self.pca_components_torch) + self.mean_torch

    def torch_debias(self, X):
        X_tr = self.torch_transform(X)
        X_orth = X - self.torch_inverse_transform(X_tr)
        return X_orth

    @staticmethod
    def center_linearly(X, X_index):
        """Center each samples in X using group mean.

        Parameters
        ----------
        X: array [n_samples, n_features]
        X_index: array: [n_samples]
            `X_index` indicates the group to which specific sample belongs to.

        """
        DX = get_design_matrix(X_index)
        centeredX = DX @ X
        return centeredX
