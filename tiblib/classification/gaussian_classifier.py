from tiblib import ClassifierBase, covariance, logpdf_GAU_ND
import numpy as np
import scipy as sc


class GaussianClassifier(ClassifierBase):
	def __init__(self, naive=False, tied=False):
		self.tied = tied
		self.naive = naive
		self.mu = None
		self.cov = None
		self.logSJoint = None
		self.num_class = None
		if tied:
			self.tied_cov = None

	def __str__(self):
		if self.naive and self.tied:
			return 'Naive, Tied'
		elif self.naive:
			return 'Naive'
		elif self.tied:
			return 'Tied'
		else:
			return 'Full'

	def fit(self, X, y):
		X = X.T
		num_class = len(np.unique(y))
		self.mu = np.empty(num_class, dtype = np.ndarray)
		self.cov = np.empty(num_class, dtype = np.ndarray)
		self.logSJoint = np.empty(num_class, dtype = np.ndarray)
		self.num_class = num_class

		cov_list = np.empty(self.num_class, dtype = np.ndarray)
		for i in range(self.num_class):
			data: np.ndarray = X[:, y == i]
			self.mu[i] = data.mean(1).reshape((-1, 1))
			self.cov[i] = covariance(data)
			if self.tied:
				cov_list[i] = self.cov[i] * data.shape[1]
		if self.tied:
			self.tied_cov = cov_list.sum(0) / X.shape[1]

	def predict(self, X):
		logscores = self.predict_scores(X)
		SPost = np.exp(logscores)
		return np.argmax(SPost, 0).astype(int)

	def predict_scores(self, X, get_ratio=False):
		X = X.T
		li = []
		for i in range(self.num_class):
			C = self.cov[i]
			if self.tied:
				C = self.tied_cov
			if self.naive:
				C = np.diag(np.diag(C))
			li.append(logpdf_GAU_ND(X, self.mu[i], C) - np.log(self.num_class))
		self.logSJoint = np.vstack(li)
		logSMarginal = sc.special.logsumexp(self.logSJoint, axis=0)
		logSPost = self.logSJoint - logSMarginal
		if get_ratio:
			assert self.num_class == 2, 'This is not a binary model'
			return logSPost[1, :] - logSPost[0, :]
		else:
			return logSPost #logscores
