
from tiblib import load_fingerprint
from tiblib.model_selection import grid_cv_multiprior
from tiblib.preprocessing import Gaussianizer, StandardScaler, PCA


print('GMM')


from tiblib.classification import GaussianMixtureClassifier

X_train, X_test, y_train, y_test = load_fingerprint()

model = GaussianMixtureClassifier
hyperparams = {'tied':[False, True],
               'diag':[False, True],
               'n_components':[4, 8, 16],
               'alpha':[1]}
prefix = 'gmm'
pis = [0.1, 0.5, 0.9]
gaussianizer = Gaussianizer()
scaler = StandardScaler()
# pca1 = PCA(n_dims=9)
preprocessings = [
    [],
    [gaussianizer],
    [scaler],
]
for pr in preprocessings:
    if len(pr) > 0:
        filename = '_'.join([str(p) for p in pr])
    else:
        filename = 'no_preproc'
    print(filename) # Prints current preprocessings in string form
    grid_cv_multiprior(X_train, y_train, pis=pis,
            preprocessing=pr,
            classifier=model, hyperparams=hyperparams, filename=f'results/results_{prefix}_{filename}.csv')
