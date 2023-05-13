
from tiblib import load_wine
from tiblib.model_selection import grid_cv_multiprior
from tiblib.preprocessing import Gaussianizer, StandardScaler, PCA
from tiblib.classification import LogisticRegression
from tiblib.classification import QuadraticLogisticRegression


print('Logistic Regression')


X_train, X_test, y_train, y_test = load_wine()

model = LogisticRegression
hyperparams = {'l':[1e-1, 1e-2, 1e-3, 1e-4]}

gaussianizer = Gaussianizer()
scaler = StandardScaler()
pca1 = PCA(n_dims=9)
pca2 = PCA(n_dims=5)
preprocessings = [
    [],
    [scaler],
    [scaler, pca1],
    [scaler, pca2]
]
prefix = 'lr'
pis = [0.1, 0.5, 0.9]
for pr in preprocessings:
    if len(pr) > 0:
        filename = '_'.join([str(p) for p in pr])
    else:
        filename = 'no_preproc'
    print(filename) # Prints current preprocessings in string form
    grid_cv_multiprior(X_train, y_train, pis=pis,
            preprocessing=pr,
            classifier=model, hyperparams=hyperparams, filename=f'results/results_{prefix}_{filename}.csv')

print('Quadratic Logistic Regression')

X_train, X_test, y_train, y_test = load_wine()

model = QuadraticLogisticRegression
hyperparams = {'l':[1e-1, 1e-2, 1e-3, 1e-4]}

gaussianizer = Gaussianizer()
scaler = StandardScaler()
pca1 = PCA(n_dims=9)
pca2 = PCA(n_dims=5)
preprocessings = [
    [],
    [scaler],
    [scaler, pca1],
    [scaler, pca2]
]
prefix = 'lr'
pis = [0.1, 0.5, 0.9]
for pr in preprocessings:
    if len(pr) > 0:
        filename = '_'.join([str(p) for p in pr])
    else:
        filename = 'no_preproc'
    print(filename) # Prints current preprocessings in string form
    grid_cv_multiprior(X_train, y_train, pis=pis,
            preprocessing=pr,
            classifier=model, hyperparams=hyperparams, filename=f'results/results_{prefix}_{filename}.csv')

