import json
import warnings
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


def covariance(X):
    assert len(X.shape) == 2, 'X is not a 2D matrix'
    # X should have samples as columns
    X_centered = X - np.mean(X, axis=1).reshape(-1, 1)
    N = X_centered.shape[1]
    cov = 1 / N * (X_centered @ X_centered.T)
    return cov


class TransformerBase(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError('fit method was not implemented!')

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError('transform method was not implemented!')

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def __len__(self): # Needed for pipeline
        return 1

    def __str__(self): # Needed for gridCV
        return self.__class__.__name__

class ClassifierBase(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError('fit method was not implemented!')

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError('predict method was not implemented!')

    @abstractmethod
    def predict_scores(self, X, get_ratio=False):
        raise NotImplementedError('predict_scores method was not implemented!')

    def score(self, X, y, metric=accuracy_score):
        return metric(y, self.predict(X))

    def __str__(self): # Needed for gridCV
        return self.__class__.__name__


def train_test_split(X, y, test_size, seed=0):
    assert test_size <= 1, 'test_size is more than 100%'
    train_size = 1 - test_size
    n_samples, n_feats = X.shape
    if n_feats > n_samples:
        warnings.warn("This method expects samples as rows. Are you sure X is not transposed?")
    n_train = int(n_samples * train_size)
    np.random.seed(seed)
    idx = np.random.permutation(n_samples)
    idx_train = idx[0:n_train]
    idx_test = idx[n_train:]
    X_train = X[idx_train]
    X_test = X[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]
    return X_train, X_test, y_train, y_test


def load_iris_binary():
    iris = load_iris()
    X, y = iris['data'], iris['target']
    X = X[y != 0]  # We remove setosa from D
    y = y[y != 0]  # We remove setosa from L
    y[y == 2] = 0  # We assign label 0 to virginica (was label 2)
    return X, y


def load_iris_multiclass():
    iris = load_iris()
    X, y = iris['data'], iris['target']
    return X, y

def load_fingerprint():
    fingerprint_train = pd.read_csv('../datasets/Train.txt', header=None)
    fingerprint_test = pd.read_csv('../datasets/Test.txt', header=None)
    X_train = fingerprint_train.iloc[:, :-1].to_numpy()
    y_train = fingerprint_train.iloc[:, -1].to_numpy()
    X_test = fingerprint_test.iloc[:, :-1].to_numpy()
    y_test = fingerprint_test.iloc[:, -1].to_numpy()
    return X_train, X_test, y_train, y_test

def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]


def GAU_logpdf(x, mu, var):
    return -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var) - np.power(x - mu, 2) / (2 * var)


def logpdf_GAU_ND(x: np.ndarray, mu: np.ndarray, C: np.ndarray):
    diff = x - mu
    _, slog = np.linalg.slogdet(C)
    return - (x.shape[0] * np.log(2 * np.pi) + slog + np.diagonal(diff.T @ np.linalg.inv(C) @ diff)) / 2

def logpdf_GAU_ND_multi_comp(x: np.ndarray, mu: np.ndarray, C: np.ndarray):
    # mu: n_components x n_feats
    n_components = mu.shape[0]
    x = np.array([x] * n_components)
    mu = mu[:, :, np.newaxis]
    diff = x - mu
    _, slog = np.linalg.slogdet(C)
    slog = slog.reshape(-1, 1)
    const = x.shape[1] * np.log(2 * np.pi)
    diag_term = np.diagonal(diff.transpose((0, 2, 1)) @ np.linalg.inv(C) @ diff, axis1=1, axis2=2)
    return - (const + slog + diag_term) / 2

def logpdf_GMM(X, gmm):
    '''
    Computes log density of a gaussian mixture given its parameters
    Note: gmm is different from what suggested in the lab pdf

    :param X: samples organized by column
    :param gmm: tuple (W, M, S) containing weights, means, covariances respectively as ndarrays
    :return:
    '''
    W, M, S = gmm
    assert W.ndim == 1, 'W has wrong number of dimensions'
    assert M.ndim == 2, 'M has wrong number of dimensions'
    assert S.ndim == 3, 'S has wrong number of dimensions'
    assert W.shape[0] == S.shape[0] and W.shape[0] == M.shape[0], 'n_components across W, M, S don\'t match'
    assert S.shape[1] == M.shape[1], 'Size of covariance matrix and means don\'t match'
    logscores = logpdf_GAU_ND_multi_comp(X, M, S) + np.log(W).reshape(-1, 1)
    loglikelihood = scipy.special.logsumexp(logscores, axis=0)
    # responsibilities = np.exp(logscores) / np.sum(np.exp(logscores), axis=0)
    responsibilities = np.exp(logscores - loglikelihood)
    return loglikelihood, responsibilities

def empirical_bayes_risk(cm, pi=.5, cfn=1, cfp=10):
    with np.errstate(invalid='ignore'):
        fnr = cm[0, 1] / (cm[0, 1] + cm[1, 1])
        fpr = cm[1, 0] / (cm[1, 0] + cm[0, 0])
    return pi * cfn * fnr + (1 - pi) * cfp * fpr


def normalized_det_cost_func(cm, pi=.5, cfn=1, cfp=10):
    dcf = empirical_bayes_risk(cm, pi, cfn, cfp)
    return dcf / min(pi * cfn, (1 - pi) * cfp)

def optimal_bayes_decision(score, pi=.5, cfn=1, cfp=10):
    threshold = - np.log(pi / (1 - pi)) + np.log(cfn / cfp)
    return (score > threshold).astype(int)


def detection_cost_func(score, y_true, pi=.5, cfn=1, cfp=10):
    opt_decision = optimal_bayes_decision(score, pi, cfn, cfp)
    cm = confusion_matrix(y_true, opt_decision)

    return normalized_det_cost_func(cm, pi, cfn, cfp)


def confusion_matrix(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Inputs must have the same length"
    n_labels = len(np.unique(y_true))
    matrix = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            matrix[i, j] = np.sum((y_pred == i) & (y_true == j))
    return matrix

def min_detection_cost_func(score, y_true, pi=.5, cfn=1, cfp=10):
    min_dcf = np.inf
    opt_t = 0

    dcfs = []
    for t in np.sort(score):
        y_pred = (score > t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        norm_dcf = normalized_det_cost_func(cm, pi, cfn, cfp)
        dcfs.append(norm_dcf)
        if norm_dcf < min_dcf:
            min_dcf = norm_dcf
            opt_t = t

    return min_dcf, opt_t