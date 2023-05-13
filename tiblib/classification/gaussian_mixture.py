import json
import matplotlib.pyplot as plt
import numpy as np

from tiblib import ClassifierBase, logpdf_GMM, load_iris_multiclass, train_test_split


class GaussianMixtureModel:
    def __init__(self, algorithm='em', tied=False, diag=False, n_components=3,
                 filename=None, alpha=1.0, psi=1e-3, stop_delta=1e-6):
        self.n_components = n_components
        self.alpha = alpha
        self.psi = psi
        self.tied = tied
        self.diag = diag
        self.stop_delta = stop_delta

        if filename is None:
            self.weights = None
            self.means = None
            self.covariances = None
            self.n_features = None
            self.file_init = False # GMM not initalized via file
            self.curr_components = 1
        else:
            with open(filename, 'r') as f:
                gmm = json.load(f)
            W = np.asarray([c[0] for c in gmm])[:]
            M = np.asarray([np.asarray(c[1]) for c in gmm]).squeeze(axis=2)
            S = np.asarray([np.asarray(c[2]) for c in gmm])
            assert len(W) <= n_components, 'Loaded model contains too many components'
            self.weights = W
            self.means = M
            self.covariances = S
            self.n_features = M.shape[1]
            self.curr_components = len(W)
            self.file_init = True  # GMM initalized via file

        self.algorithm = algorithm
        if algorithm == 'em':
            self.curr_components = self.n_components

    def fit(self, X, y=None):
        X = X.T
        n_features, n_samples = X.shape
        if not self.file_init: # GMM not initalized via file
            self.n_features = n_features
            self.covariances = np.tile(np.cov(X), (self.curr_components, 1, 1))
            self.means = np.tile(np.mean(X, axis=1), (self.curr_components, 1))
            self.weights = np.ones(self.curr_components)
        assert n_features == self.n_features, 'Model has been initialized with wrong number of features'

        if self.algorithm == 'em':
            self._fit_em(X)
        elif self.algorithm == 'lbg':
            self._fit_lbg(X)
        return self

    def _fit_em(self, X):
        n_feats, n_samples = X.shape
        last_ll = -np.inf
        delta_ll = 1
        loglikelihood, responsibilities = logpdf_GMM(X, (self.weights, self.means, self.covariances))

        while delta_ll > self.stop_delta:
            # Maximization step - update parameters
            Z = np.sum(responsibilities, axis=1)
            sum_Z = np.sum(Z)
            F = np.sum(np.einsum('cn,rn->crn',responsibilities,X), axis=2)
            XXT = np.stack([Xi.reshape(-1, 1) @ Xi.reshape(1, -1) for Xi in X.T])

            S = np.einsum('cn,nij->cij', responsibilities, XXT)

            self.weights = Z / sum_Z
            self.means = F / Z[:, np.newaxis]
            self.covariances = S / Z[:, np.newaxis, np.newaxis] - np.einsum('ci,cj->cij', self.means, self.means)

            # Enforce structure on covariance
            if self.tied:
                self.covariances = 1 / n_samples * np.tile(np.sum(self.covariances * Z[:,np.newaxis, np.newaxis], axis=0), (self.curr_components,1,1))
            if self.diag:
                self.covariances = self.covariances * np.eye(self.covariances.shape[1])[np.newaxis, :, :]

            # Bound covariance
            U, s, _ = np.linalg.svd(self.covariances)
            s[s < self.psi] = self.psi
            self.covariances = U @ (s[:,:,np.newaxis] * U.transpose(0,2,1)) * self.alpha

            loglikelihood, responsibilities = logpdf_GMM(X, (self.weights, self.means, self.covariances))
            sum_ll = np.sum(loglikelihood)
            delta_ll = np.abs(sum_ll - last_ll)
            if delta_ll < 10e-6:
                break
            last_ll = sum_ll

    def _fit_lbg(self,X):
        self._fit_em(X)
        while self.curr_components < self.n_components:
            U, s, _ = np.linalg.svd(self.covariances)
            d = U[:, :, 0:1] * s[:, None, 0:1] ** 0.5

            weights_p = self.weights/2
            means_p = self.means + d.squeeze(axis=2)
            covariances_p = self.covariances

            weights_n = self.weights/2
            means_n = self.means - d.squeeze(axis=2)
            covariances_n = self.covariances

            self.weights = np.concatenate([weights_p, weights_n])
            self.means = np.concatenate([means_p, means_n])
            self.covariances = np.concatenate([covariances_p, covariances_n])

            self.curr_components *= 2
            assert len(self.weights) == self.curr_components, 'Something went wrong when doubling components'
            self._fit_em(X)


    def estimate(self, X):
        X = X.reshape(-1, X.shape[-1]).T
        if self.n_features is None:
            raise ValueError('GMM was not fitted on any data!')
        log_prob, _ = logpdf_GMM(X, (self.weights, self.means, self.covariances))
        return log_prob


class GaussianMixtureClassifier(ClassifierBase):
    def __init__(self, algorithm='lbg', tied=False, diag=False, n_components=3, max_iter=10, alpha=0.1, psi=1e-3, stop_delta=1e-6):
        self.tied = tied
        self.diag = diag
        self.psi = psi
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_components = n_components
        self.algorithm = algorithm
        self.stop_delta = stop_delta
        self.n_classes = None

    def __str__(self):
        if self.diag and self.tied:
            return f'GMM (Diag, Tied, {self.n_components} components, $\\alpha = {self.alpha}$)'
        elif self.diag:
            return f'GMM (Diag, {self.n_components} components, $\\alpha = {self.alpha}$)'
        elif self.tied:
            return f'GMM (Tied, {self.n_components} components, $\\alpha = {self.alpha}$)'
        else:
            return f'GMM ({self.n_components} components, $\\alpha = {self.alpha}$)'


    def fit(self, X, y):
        self.models = {}  # Dict containing GMM per class
        labels = np.unique(y)
        self.n_classes = len(labels)
        for c in labels:
            X_c = X[y == c]
            gmm = GaussianMixtureModel(n_components=self.n_components, algorithm=self.algorithm,
                                       alpha=self.alpha, psi=self.psi, tied=self.tied, diag=self.diag,
                                       stop_delta=self.stop_delta)
            gmm.fit(X_c)
            self.models[c] = gmm

    def predict_scores(self, X, get_ratio=False):
        logprobs = np.array([gmm.estimate(X) for gmm in self.models.values()])
        if get_ratio:
            assert self.n_classes == 2, 'This is not a binary model'
            return logprobs[1, :] - logprobs[0, :] #llr
        else:
            return logprobs  # logscores

    def predict(self, X):
        scores = self.predict_scores(X)
        return np.array([list(self.models.keys())[i] for i in np.argmax(scores, axis=0)])
