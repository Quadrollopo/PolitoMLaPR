from tiblib import load_fingerprint
from tiblib.model_selection import grid_cv_multiprior
from tiblib.preprocessing import Gaussianizer, StandardScaler, PCA
from tiblib.classification import SVC


print('SVM')


X_train, X_test, y_train, y_test = load_fingerprint()

model = SVC
hyperparams = {'C':[1e-1, 1e-2, 1e-3, 1e-4],
               'kernel': ['linear', 'poly', 'radial']}
prefix = 'svm'
pis = [0.1, 0.5, 0.9]
gaussianizer = Gaussianizer()
scaler = StandardScaler()
pca1 = PCA(n_dims=9)
pca2 = PCA(n_dims=5)
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
