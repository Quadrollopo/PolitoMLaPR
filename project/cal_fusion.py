import numpy as np
from tiblib import load_wine
from tiblib.model_selection import Calibrate
from tiblib.classification import QuadraticLogisticRegression, GaussianClassifier
from tiblib.classification import GaussianMixtureClassifier, SVC, Pipeline
from tiblib.preprocessing import Gaussianizer, StandardScaler, PCA

print('Calibration')

X_train, _, y_train, _ = load_wine()

g = Gaussianizer()
ss = StandardScaler()
pca = PCA(n_dims=9)
gc = GaussianClassifier()
qrl = QuadraticLogisticRegression(l=1e-3)
svm1 = SVC(kernel='radial', gamma=1/np.e**2, C=10)
svm2 = SVC(kernel='radial', gamma=1/np.e, C=1)
gmm1 = GaussianMixtureClassifier(n_components=8)
gmm2 = GaussianMixtureClassifier(n_components=16, tied=True)

model1 = Pipeline(g, gc)
model2 = Pipeline([ss, pca], gc)
model3 = Pipeline([ss], qrl)
model4 = Pipeline([ss, pca], qrl)
model5 = Pipeline(ss, svm1)
model6 = Pipeline(ss, svm2)
model7 = Pipeline(ss, gmm1)
model8 = Pipeline(g, gmm2)

models = [model1, model2, model3, model4, model5, model6, model7, model8]
names = ['gc1', 'gc2', 'qlr1', 'qlr2', 'svm1', 'svm2', 'gmm1', 'gmm2']

for n, m in zip(names, models):
    min_dcf, act_dcf, cal_dcf, scores, cal_scores = Calibrate(m, X_train, y_train)
    print(f'{n} & {min_dcf:.3f} & {act_dcf:.3f} & {cal_dcf:.3f}')
    np.save(f'results/scores_{n}', scores)
    np.save(f'results/cal_scores_{n}', cal_scores)

print('Fusion')

from tiblib.model_selection import Fusion

X_train, _, y_train, _ = load_wine()

model_names = ['svm2', 'qlr1', 'gmm1']
scores = []
for n in model_names:
    scores.append(np.load(f'results/cal_scores_{n}.npy').reshape(-1,1))
scores = np.concatenate(scores, axis=1)

min_dcf, act_dcf, fused_score = Fusion(scores,y_train)
print(f'QLR + SVM + GMM & {min_dcf:.3} & {act_dcf:.3} \\\\')
np.save(f'results/fusion_scores_{"_".join(model_names)}', fused_score)

from tiblib.model_selection import Fusion

X_train, _, y_train, _ = load_wine()

model_names = ['svm2', 'qlr1']
scores = []
for n in model_names:
    scores.append(np.load(f'results/cal_scores_{n}.npy').reshape(-1,1))
scores = np.concatenate(scores, axis=1)

min_dcf, act_dcf, fused_score = Fusion(scores,y_train)
print(f'QLR + SVM & {min_dcf:.3} & {act_dcf:.3} \\\\')
np.save(f'results/fusion_scores_{"_".join(model_names)}', fused_score)


from tiblib.model_selection import Fusion

X_train, _, y_train, _ = load_wine()

model_names = ['svm2', 'gmm1']
scores = []
for n in model_names:
    scores.append(np.load(f'results/cal_scores_{n}.npy').reshape(-1,1))
scores = np.concatenate(scores, axis=1)

min_dcf, act_dcf, fused_score = Fusion(scores,y_train)
print(f'SVM + GMM & {min_dcf:.3} & {act_dcf:.3} \\\\')
np.save(f'results/fusion_scores_{"_".join(model_names)}', fused_score)


from tiblib.model_selection import Fusion

X_train, _, y_train, _ = load_wine()

model_names = ['qlr1', 'gmm1']
scores = []
for n in model_names:
    scores.append(np.load(f'results/cal_scores_{n}.npy').reshape(-1,1))
scores = np.concatenate(scores, axis=1)

min_dcf, act_dcf, fused_score = Fusion(scores,y_train)
print(f'QLR + GMM & {min_dcf:.3} & {act_dcf:.3} \\\\')
np.save(f'results/fusion_scores_{"_".join(model_names)}', fused_score)
