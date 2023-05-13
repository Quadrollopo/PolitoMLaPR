import numpy as np
from tiblib import TransformerBase, covariance


class PCA(TransformerBase):
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.princ_comps = None

    def fit(self, X, y=None):
        X = X.T
        cov = covariance(X)
        U, _, _ = np.linalg.svd(cov)
        self.princ_comps = U[:, :self.n_dims]

    def transform(self, X):
        if self.princ_comps is None:
            raise ValueError('PCA was not fitted on any data!')
        X = X.T
        return (self.princ_comps.T @ X).T

    def __str__(self):
        return f'PCA (d={self.n_dims})'