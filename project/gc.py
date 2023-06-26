from tiblib import load_fingerprint
from tiblib.model_selection import grid_cv_multiprior
from tiblib.preprocessing import Gaussianizer, StandardScaler, PCA
from tiblib.classification import GaussianClassifier

print('Gaussian Classifier')

X_train, X_test, y_train, y_test = load_fingerprint()

model = GaussianClassifier
hyperparams = {'tied':[False, True],
               'naive':[False, True]}

gaussianizer = Gaussianizer()
scaler = StandardScaler()
pca1 = PCA(n_dims=7)
pca2 = PCA(n_dims=9)
preprocessings = [
    [],
    [gaussianizer],
    [scaler],
    [scaler, pca1],
    [scaler, pca2],
    [gaussianizer, scaler, pca1],
    [gaussianizer, scaler, pca2],
]
prefix = 'gc'
pis = [0.5]
for pr in preprocessings:
    if len(pr) > 0:
        filename = '_'.join([str(p) for p in pr])
    else:
        filename = 'no_preproc'
    print(filename) # Prints current preprocessings in string form
    grid_cv_multiprior(X_train, y_train, pis=pis,
            preprocessing=pr,
            classifier=model, hyperparams=hyperparams, filename=f'results/results_{prefix}_{filename}.csv')
