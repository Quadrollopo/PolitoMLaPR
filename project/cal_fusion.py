import numpy as np
from tiblib import load_fingerprint
from tiblib.model_selection import Calibrate
from tiblib.classification import QuadraticLogisticRegression
from tiblib.classification import GaussianMixtureClassifier, SVC, Pipeline
from tiblib.preprocessing import Gaussianizer, StandardScaler, PCA

print('Calibration')

X_train, X_test, y_train, y_test = load_fingerprint()
pi = 0.9
ss = StandardScaler()
qlr = QuadraticLogisticRegression(l=1e-2)
svm = SVC(kernel='radial', C=.1, gamma=np.exp(-2))
pca = PCA(9)
g = Gaussianizer()
gmm = GaussianMixtureClassifier(n_components=8, tied = True)

model1 = Pipeline([ss, pca], qlr)
model2 = Pipeline(g, svm)
model3 = Pipeline([], gmm)

models = [model1, model2, model3]
names = ['qlr1', 'svm1', 'gmm1']

for n, m in zip(names, models):
    min_dcf, act_dcf, cal_dcf, scores, cal_scores = Calibrate(m, X_train, y_train, pi = 0.9)
    print(f'{n} & {min_dcf:.3f} & {act_dcf:.3f} & {cal_dcf:.3f}')
    np.save(f'results/scores_{n}', scores)
    np.save(f'results/cal_scores_{n}', cal_scores)

print('Fusion')

from tiblib.model_selection import Fusion

X_train, _, y_train, _ = load_fingerprint()

model_names = ['svm1', 'qlr1', 'gmm1']
scores = []
for n in model_names:
    scores.append(np.load(f'results/cal_scores_{n}.npy').reshape(-1,1))
scores = np.concatenate(scores, axis=1)

min_dcf, act_dcf, fused_score = Fusion(scores,y_train)
print(f'QLR + SVM + GMM & {min_dcf:.3} & {act_dcf:.3} \\\\')
np.save(f'results/fusion_scores_{"_".join(model_names)}', fused_score)

from tiblib.model_selection import Fusion

X_train, _, y_train, _ = load_fingerprint()

model_names = ['svm1', 'qlr1']
scores = []
for n in model_names:
    scores.append(np.load(f'results/cal_scores_{n}.npy').reshape(-1,1))
scores = np.concatenate(scores, axis=1)

min_dcf, act_dcf, fused_score = Fusion(scores,y_train)
print(f'QLR + SVM & {min_dcf:.3} & {act_dcf:.3} \\\\')
np.save(f'results/fusion_scores_{"_".join(model_names)}', fused_score)


from tiblib.model_selection import Fusion

X_train, _, y_train, _ = load_fingerprint()

model_names = ['svm1', 'gmm1']
scores = []
for n in model_names:
    scores.append(np.load(f'results/cal_scores_{n}.npy').reshape(-1,1))
scores = np.concatenate(scores, axis=1)

min_dcf, act_dcf, fused_score = Fusion(scores,y_train)
print(f'SVM + GMM & {min_dcf:.3} & {act_dcf:.3} \\\\')
np.save(f'results/fusion_scores_{"_".join(model_names)}', fused_score)


from tiblib.model_selection import Fusion

X_train, _, y_train, _ = load_fingerprint()

model_names = ['qlr1', 'gmm1']
scores = []
for n in model_names:
    scores.append(np.load(f'results/cal_scores_{n}.npy').reshape(-1,1))
scores = np.concatenate(scores, axis=1)

min_dcf, act_dcf, fused_score = Fusion(scores,y_train)
print(f'QLR + GMM & {min_dcf:.3} & {act_dcf:.3} \\\\')
np.save(f'results/fusion_scores_{"_".join(model_names)}', fused_score)
