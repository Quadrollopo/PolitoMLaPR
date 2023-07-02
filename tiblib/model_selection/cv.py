import warnings
from itertools import product

import numpy as np
import pandas as pd

from tiblib import min_detection_cost_func, detection_cost_func
from tiblib.classification import LogisticRegression, Pipeline


def calibrate(score_train, y_train, _lambda, pi=0.9):
    lr = LogisticRegression(l=_lambda)
    lr.fit(score_train.reshape(-1,1), y_train)
    # score = lr.predict_scores(score_test.reshape(-1,1), get_ratio=True)
    alpha = lr.w
    beta_p = lr.b
    cal_score = (alpha @ score_train.reshape(1,-1)) + beta_p - np.log(pi / (1 - pi))
    return cal_score


class Kfold:
    def __init__(self, num_sample=5):
        self.num_sample = num_sample

    def split(self, X):
        part = len(X) // self.num_sample
        index = np.arange(len(X))
        folds = []
        splits = np.empty(self.num_sample, dtype=np.ndarray)
        for j, i in enumerate(range(0, len(X), part)):
            splits[j] = (index[i:i + part])
        index = np.arange(self.num_sample)
        for i in range(self.num_sample):
            folds.append((np.concatenate(splits[index != i]), splits[i]))
        return folds


def CVMinDCF(model, X, y, K=5, pi=.5, calibration=False, _lambda=1e-3):
    if X.shape[0] < X.shape[1]:
        warnings.warn(f'Samples in X should be rows. Are you sure the dataset is not transposed? Size: {X.shape}')
    scores = np.empty(X.shape[0]) # Will store scores for the KFold

    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = int(n / K)
    for i in range(K): # Score computing
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        model.fit(X_train, y_train)
        val_scores = model.predict_scores(X_val, get_ratio=True)
        if calibration:
            val_scores = calibrate(val_scores, y_val, _lambda, pi)
        scores[val_indices] = val_scores

    min_dcf, _ = min_detection_cost_func(scores, y, pi=pi, cfp = 10)
    act_dcf = detection_cost_func(scores, y, pi=pi,  cfp = 10)
    return min_dcf, act_dcf, scores


def CVFusion(models, X, y, K=5, pi=.5, _lambda=1e-3):
    assert len(models) > 1, 'Fusion needs at least 2 models to work'
    if X.shape[0] < X.shape[1]:
        warnings.warn(f'Samples in X should be rows. Are you sure the dataset is not transposed? Size: {X.shape}')
    scores = np.empty((X.shape[0], len(models))) # Will store scores for the KFold
    fused_score = np.empty(X.shape[0])

    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = int(n / K)
    for i in range(K): # Score computing
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        for i, m in enumerate(models):
            m.fit(X_train, y_train)
            val_scores = m.predict_scores(X_val, get_ratio=True)
            scores[val_indices, i] = val_scores

    for i in range(K):
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)
        scores_train, y_train = scores[train_indices], y[train_indices]
        scores_val, y_val = scores[val_indices], y[val_indices]
        lr = LogisticRegression(l=_lambda)
        lr.fit(scores_train, y_train)

        val_scores = lr.predict_scores(scores_val, get_ratio=True)
        fused_score[val_indices] = val_scores

    min_dcf, _ = min_detection_cost_func(fused_score, y, pi=pi)
    act_dcf = detection_cost_func(fused_score, y, pi=pi)
    return min_dcf, act_dcf, fused_score


def Calibrate(model, X, y, K=5, pi=.5, _lambda=1e-3):
    if X.shape[0] < X.shape[1]:
        warnings.warn(f'Samples in X should be rows. Are you sure the dataset is not transposed? Size: {X.shape}')
    scores = np.empty(X.shape[0]) # Will store scores for the KFold
    cal_scores = np.empty(X.shape[0]) # Will store scores for the KFold

    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = int(n / K)
    for i in range(K): # Score computing
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        model.fit(X_train, y_train)
        val_scores = model.predict_scores(X_val, get_ratio=True)
        scores[val_indices] = val_scores
        val_scores = calibrate(val_scores, y_val, _lambda, pi)
        cal_scores[val_indices] = val_scores
    min_dcf, _ = min_detection_cost_func(scores, y, pi=pi)
    act_dcf = detection_cost_func(scores, y, pi=pi)
    cal_dcf = detection_cost_func(cal_scores, y, pi=pi)
    return min_dcf, act_dcf, cal_dcf, scores, cal_scores


def Fusion(scores, y, K=5, pi=.9, _lambda=1e-3):
    assert scores.shape[1] > 1, 'Fusion needs at least 2 models to work'
    if scores.shape[0] < scores.shape[1]:
        warnings.warn(f'Samples in X should be rows. Are you sure the dataset is not transposed? Size: {scores.shape}')
    fused_score = np.empty(scores.shape[0])

    n = scores.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = int(n / K)
    for i in range(K):
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, val_indices)
        scores_train, y_train = scores[train_indices], y[train_indices]
        scores_val, y_val = scores[val_indices], y[val_indices]
        lr = LogisticRegression(l=_lambda)
        lr.fit(scores_train, y_train)

        val_scores = lr.predict_scores(scores_val, get_ratio=True)
        fused_score[val_indices] = val_scores

    min_dcf, _ = min_detection_cost_func(fused_score, y, pi=pi)
    act_dcf = detection_cost_func(fused_score, y, pi=pi)
    return min_dcf, act_dcf, fused_score

def grid_cv(X, y, pi, preprocessing, classifier, hyperparams, filename):
    keys, values = zip(*hyperparams.items())
    results = []
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    for params in combinations:
        c = classifier(**params)
        model = Pipeline(transformers=preprocessing, classifier=c)
        best_score, best_act_score, _ = CVMinDCF(X=X, y=y, model=model, pi=pi)
        print(f'{c}\t\t & {best_act_score:.3f}\t & {best_score:.3f}')
        results.append({'Model' : str(c),
                        'Min DCF' : best_score,
                        'Act DCF' : best_act_score,
                        **params})
    pd_results = pd.DataFrame(results)
    pd_results.to_csv(filename, index=False)


def grid_cv_multiprior(X, y, pis, preprocessing, classifier, hyperparams, filename):
    # Version printing the priors
    keys, values = zip(*hyperparams.items())
    results = []
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    print(f'Showing results for pi = {pis}')
    for params in combinations:
        pi_min_dcf = []
        c = classifier(**params)
        model = Pipeline(transformers=preprocessing, classifier=c)
        for pi in pis:
            best_score, best_act_score, _ = CVMinDCF(X=X, y=y, model=model, pi=pi)
            pi_min_dcf.append(best_score)
            results.append({'Prior' : pi,
                            'Model' : str(c),
                            'Min DCF' : best_score,
                            'Act DCF' : best_act_score,
                            **params})
        print(f'{c}\t\t', end='')
        for r in pi_min_dcf:
            print(f'& {r:.3f}', end='\t')
        print('\\\\')
    pd_results = pd.DataFrame(results)
    pd_results.to_csv(filename, index=False)