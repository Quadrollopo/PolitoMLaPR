import numpy as np

from tiblib import TransformerBase
from scipy.stats import norm


class Gaussianizer(TransformerBase):
    def __init__(self):
        self.X_train = None

    def fit(self, X, y=None):
        self.X_train = X.T
        return self

    def transform(self, X):
        X = X.T
        if self.X_train is None:
            raise ValueError('Gaussianizer was not fitted on any data!')
        n_feats, n_samples = X.shape
        transformed = np.empty([n_feats, n_samples])
        _, n_samples_train = self.X_train.shape
        for i in range(n_samples):
            rank = (1 + np.sum(X[:, i].reshape([n_feats, 1]) < self.X_train, axis=1)).astype(float)
            rank /= (n_samples_train + 2)
            transformed[:, i] = norm.ppf(rank)
        return transformed.T
