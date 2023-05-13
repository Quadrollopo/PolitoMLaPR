import numpy as np
import scipy
from tiblib import TransformerBase, covariance


class LDA(TransformerBase):
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.U = None

    def fit(self, X, y):
        assert self.n_dims <= len(np.unique(y)) - 1, 'LDA has n_dims > n_classes - 1'
        X = X.T
        Sw = within_class_covariance(X, y)
        Sb = between_class_covariance(X, y)
        _, U = scipy.linalg.eigh(Sb, Sw)
        self.U = U[:, ::-1][:, :self.n_dims]

    def transform(self, X):
        if self.U is None:
            raise ValueError('LDA was not fitted on any data!')
        X = X.T
        return (self.U.T @ X).T


def within_class_covariance(X, y):
    # X should have samples as columns
    n_feats, n_samples = X.shape
    Sw = np.zeros((n_feats, n_feats))
    classes = np.unique(y)
    for c in classes:
        X_c = X[:, y == c]
        Sw += covariance(X_c) * X_c.shape[1]
    return Sw / n_samples


def between_class_covariance(X, y):
    # X should have samples as columns
    n_feats, n_samples = X.shape
    mu = X.mean(axis=1)
    Sb = np.zeros((n_feats, n_feats))
    classes = np.unique(y)
    for c in classes:
        X_c = X[:, y == c]
        mu_c = X_c.mean(axis=1)
        mu_diff = (mu_c - mu).reshape(-1,1)
        Sb += (mu_diff @ mu_diff.T) * X_c.shape[1]
    return Sb / n_samples
