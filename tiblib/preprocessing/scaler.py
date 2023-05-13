import numpy as np
from tiblib import TransformerBase


class StandardScaler(TransformerBase):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X, y=None):
        if self.mean is None or self.std is None:
            raise ValueError('StandardScaler was not fitted on any data!')
        return (X - self.mean) / self.std
