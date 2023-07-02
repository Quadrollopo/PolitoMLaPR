import numpy as np
from tiblib import load_fingerprint
from tiblib.classification import QuadraticLogisticRegression, SVC, GaussianMixtureClassifier, Pipeline
from tiblib.preprocessing import StandardScaler, PCA, Gaussianizer
from tiblib import min_detection_cost_func, detection_cost_func
from tiblib.model_selection import CVMinDCF
from tiblib.model_selection.cv import calibrate

# region Uncalibrated models
print('Uncalibrated models')

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
names = ['QLR', 'SVM', 'GMM']

for m, n in zip(models, names):
    m.fit(X_train, y_train)
    scores = m.predict_scores(X_test, get_ratio=True)
    min_dcf, _ = min_detection_cost_func(scores, y_test, pi=pi)
    act_dcf = detection_cost_func(scores, y_test, pi=pi)

    np.save(f'results/eval_{n}', scores)
    print(f'{n}\t & {min_dcf:.3} \t & {act_dcf:.3}')

#endregion

print('Calibrated models')


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
names = ['QLR', 'SVM', 'GMM']
for m, n in zip(models, names):

    _, _, scores_train = CVMinDCF(m, X_train, y_train, K=5, pi=.9, calibration=False, _lambda=1e-3)

    m.fit(X_train, y_train)
    scores_test = m.predict_scores(X_test, get_ratio=True)

    scores_cal = calibrate(scores_train, y_train, _lambda=1e-3, pi=pi)

    min_dcf, _ = min_detection_cost_func(scores_cal, y_test, pi=pi)
    act_dcf = detection_cost_func(scores_cal, y_test, pi=pi)

    np.save(f'results/eval_cal_{n}', scores_cal)
    print(f'{n}\t & {min_dcf:.3} \t & {act_dcf:.3}')


print('Fusion uncalibrated')


from tiblib.classification import LogisticRegression

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
names = ['QLR', 'SVM', 'GMM']
scores_train = []
scores_test = []
for m, n in zip(models, names):

    _, _, sc_tr = CVMinDCF(m, X_train, y_train, K=5, pi=pi, calibration=False, _lambda=1e-3)

    m.fit(X_train, y_train)
    sc_ts = m.predict_scores(X_test, get_ratio=True)

    scores_train.append(sc_tr.reshape(-1,1))
    scores_test.append(sc_ts.reshape(-1,1))

scores_train = np.concatenate(scores_train, axis=1)
scores_test = np.concatenate(scores_test, axis=1)

lr = LogisticRegression(l=1e-3)
lr.fit(scores_train, y_train)

fusion_scores = lr.predict_scores(scores_test, get_ratio=True)

min_dcf, _ = min_detection_cost_func(fusion_scores, y_test, pi=pi)
act_dcf = detection_cost_func(fusion_scores, y_test, pi=pi)

np.save(f'results/eval_fusion_QLR_SVM_GMM', fusion_scores)
print(f'QLR + SVM + GMM\t & {min_dcf:.3} \t & {act_dcf:.3}')


print('Fusion calibrated')


from tiblib.classification import LogisticRegression

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
names = ['QLR', 'SVM', 'GMM']
scores_train = []
scores_test = []
for m, n in zip(models, names):

    _, _, sc_tr = CVMinDCF(m, X_train, y_train, K=5, pi=pi, calibration=False, _lambda=1e-3)

    m.fit(X_train, y_train)
    sc_ts = m.predict_scores(X_test, get_ratio=True)

    sc_tr = calibrate(sc_tr, y_train, _lambda=1e-3)

    scores_train.append(sc_tr.reshape(-1,1))
    scores_test.append(sc_ts.reshape(-1,1))

scores_train = np.concatenate(scores_train, axis=1)
scores_test = np.concatenate(scores_test, axis=1)

lr = LogisticRegression(l=1e-3)
lr.fit(scores_train, y_train)

fusion_scores = lr.predict_scores(scores_test, get_ratio=True)

min_dcf, _ = min_detection_cost_func(fusion_scores, y_test, pi=pi)
act_dcf = detection_cost_func(fusion_scores, y_test, pi=pi)

np.save(f'results/eval_fusion_cal_QLR_SVM_GMM', fusion_scores)
print(f'QLR + SVM + GMM\t & {min_dcf:.3} \t & {act_dcf:.3}')
