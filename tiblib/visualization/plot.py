import numpy as np
from warnings import warn
from matplotlib import pyplot as plt
from tiblib import detection_cost_func, min_detection_cost_func, confusion_matrix


def pair_plot(X,y,columns=None):
    nrows = X.shape[0]
    ncols = X.shape[1]
    labels = np.unique(y)
    if ncols > nrows:
        warn(f'Samples in X should be rows. Are you sure the dataset is not transposed? Size: {X.shape}')
    if columns is not None:
        assert len(columns) == ncols, 'Size of X does not match number of column names provided'
    fig, axs = plt.subplots(ncols, ncols)
    fig.set_size_inches(18.5, 10.5)
    for feat1 in range(ncols):
        for feat2 in range(ncols):
            for l in labels:
                if feat1 == feat2:
                    axs[feat1, feat2].hist(X[y == l][:,feat1], alpha=0.3)
                    if columns is not None:
                        axs[feat1, feat2].set_xlabel(columns[feat1])
                else:
                    axs[feat1, feat2].scatter(X[y == l][:,feat1],X[y == l][:,feat2],marker='.')
                    if columns is not None:
                        axs[feat1, feat2].set_xlabel(columns[feat1])
                        axs[feat1, feat2].set_ylabel(columns[feat2])
    plt.tight_layout()
    plt.show()


def plot_2d(X, y):
    assert X.shape[1] == 2, 'Dataset is not 2D!'
    labels = np.unique(y)
    for l in labels:
        plt.scatter(X[y == l, 0], X[y == l, 1])
    plt.legend(labels)
    plt.show()


def plot_dcf(scores, y_true, name, save=False):
    prior = np.linspace(-4, 4, 100)

    act_dcf = np.zeros(prior.shape[0])
    min_dcf = np.zeros(prior.shape[0])

    for i, p in enumerate(prior):
        t = 1 / (1 + np.exp(-p))
        act_dcf[i] = detection_cost_func(scores, y_true, t)
        min_dcf[i], _ = min_detection_cost_func(scores, y_true, t)

    plt.plot(prior, act_dcf, label=f'{name} - act DCF')
    plt.plot(prior, min_dcf, label=f'{name} - min DCF', linestyle='dashed')

    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.legend()
    plt.xlabel(r'$\log \frac{\tilde{\pi}}{1 - \tilde{\pi}}$')
    plt.ylabel("DCF")
    if save:
        plt.savefig(f'./images/{name}.png')
    else:
        plt.show()


def multiplot_dcf(scores, y_true, plot_names, filename, save=False):
    prior = np.linspace(-4, 4, 100)

    act_dcf = np.zeros(prior.shape[0])
    min_dcf = np.zeros(prior.shape[0])

    fig, axes = plt.subplots(2, int(np.ceil(scores.shape[1]/2)))
    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.tight_layout(pad=5)
    if scores.shape[1] % 2 != 0:
        fig.delaxes(axes[1][-1])
    for ax, s, pn in zip(axes.reshape(-1), scores.T, plot_names):
        for i, p in enumerate(prior):
            t = 1 / (1 + np.exp(-p))
            act_dcf[i] = detection_cost_func(s, y_true, t)
            min_dcf[i], _ = min_detection_cost_func(s, y_true, t)

        ax.title.set_text(pn)
        ax.plot(prior, act_dcf, label=f'act DCF', linewidth=2.5)
        ax.plot(prior, min_dcf, label=f'min DCF', linestyle='dashed', linewidth=2.5)

        ax.set_ylim([0, 1.1])
        ax.set_xlim([-4, 4])
        ax.legend()
        ax.set_xlabel(r'$\log \frac{\tilde{\pi}}{1 - \tilde{\pi}}$')
        ax.set_ylabel("DCF")

    if save:
        plt.savefig(f'./images/{filename}.png')
    else:
        plt.show()


def plot_roc(scores_pred, y_true, names, filename=None):
    if scores_pred.ndim > 1:
        assert scores_pred.shape[0] > scores_pred.shape[1], 'Scores expected br rows, not columns'
        assert scores_pred.shape[1] == len(names), 'Size of scores doesn\'t match number of names'
        scores_pred = scores_pred.T
    else:
        scores_pred = scores_pred.reshape(1,-1)

    plt.xlabel("False Positive Ratio")
    plt.ylabel("True Positive Ratio")
    plt.grid(b=True)
    for s, n in zip(scores_pred, names):
        sorted_scores = np.sort(s)
        fpr = np.zeros(sorted_scores.shape[0])
        tpr = np.zeros(sorted_scores.shape[0])

        for i, t in enumerate(sorted_scores):
            matrix = confusion_matrix(y_true, s > t)
            fnr = matrix[0, 1] / (matrix[0, 1] + matrix[1, 1])
            fpr[i] = matrix[1, 0] / (matrix[0, 0] + matrix[1, 0])
            tpr[i] = 1 - fnr
        plt.plot(fpr, tpr, label=f'{n} AUC: {tpr.sum() / len(s):.3}')
    plt.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(f'./images/{filename}')
