import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from tiblib import ClassifierBase


class SVC(ClassifierBase):
	def __init__(self, C=1.0, K=1.0, kernel='linear', c=2, d=1, gamma=0.1, pi=0.5):
		assert kernel == 'linear' or kernel == 'radial' or kernel == 'poly', f"{kernel} is not a valid kernel type, valid types are: 'linear', 'poly', 'radial'"
		self.x = None
		self.k = None
		self.z = None
		self.alpha = None
		self.gamma = gamma
		self.d = d
		self.c = c
		self.kernel = kernel
		self.W = None
		self.b = None
		self.K = K
		self.C = C
		self.pi = pi

	def __str__(self):
		if self.kernel == 'linear':
			return f'SVC (Linear, $C = {self.C}$)'
		elif self.kernel == 'poly':
			return f'SVC (Poly, $C = {self.C}$)'
		elif self.kernel == 'radial':
			return f'SVC (RBF, $C = {self.C}$, $\gamma = {self.gamma}$)'


	def fit(self, X, y):
		n_samples, n_features = X.shape
		# Initialize the primal variables
		alpha = np.zeros(n_samples)
		X = X.T
		self.x = X
		x_r = np.vstack((X, np.full(n_samples, self.K)))

		# Initialize the kernel matrix
		if self.kernel == 'linear':
			self.k = x_r.T @ x_r
		else:
			self.k = self._kern(X, X)
		self.z = np.where(y == 1, 1, -1)
		H = self.k * self.z.reshape((-1, 1)) * self.z

		# Define the optimization function
		def objective(alpha):
			return 0.5 * alpha.T @ H @ alpha - np.sum(alpha), (H @ alpha - 1).reshape(n_samples)

		# Define the bounds
		bounds = self._define_bounds(X, y)

		# Optimize the primal variables using scipy's fmin_l_bfgs_b function
		self.alpha = fmin_l_bfgs_b(objective, alpha, bounds=bounds, approx_grad=False, factr=1e6)[0]

		res = np.sum(self.alpha * self.z * x_r, 1)
		self.b = res[-1]
		self.W = res[:-1]

	def predict(self, X):
		predictions = self.predict_scores(X)
		return np.heaviside(predictions, 0)

	def _define_bounds(self, X, y):
		n_feats, n_samples = X.shape
		n_labels = len(np.unique(y))
		if n_labels != 2:
			bounds = [(0, self.C) for i in range(n_samples)]
		else: # Class balancing
			pi_emp = sum(y == 1) / y.shape[0]
			bounds = np.zeros([n_samples, 2])
			Ct = self.C * self.pi / pi_emp
			Cf = self.C * (1 - self.pi) / (1 - pi_emp)
			bounds[y == 1, 1] = Ct
			bounds[y == 0, 1] = Cf
		return bounds

	def _kern(self, x1, x2):
		if self.kernel == 'poly':
			return np.power(x1.T @ x2 + self.c, self.d) + self.K ** 2
		elif self.kernel == 'radial':
			# return np.exp(-self.gamma * np.square(np.linalg.norm(x1.T-x2.T))) + self.K ** 2
			a = np.repeat(x1, x2.shape[1], axis = 1)
			b = np.tile(x2, x1.shape[1])
			m = (np.linalg.norm(a - b, axis = 0) ** 2).reshape((x1.shape[1], x2.shape[1]))
			return np.exp(-self.gamma * m) + self.K
		else:
			raise ValueError(f"{self.kernel} is not a valid kernel type, valid types are: 'linear', 'poly', 'radial'")

	def predict_scores(self, X, get_ratio=False):
		if self.kernel == 'linear':
			score = self.W.T @ X.T + self.b * self.K
		else:
			score = (self.alpha * self.z) @ self._kern(self.x, X.T)
		return score
