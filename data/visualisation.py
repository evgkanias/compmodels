import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns

eps = np.finfo(float).eps


def plot_matrix(M, title="", labels1=None, labels2=None, vmin=-1., vmax=1., verbose=False):
    if verbose:
        print "M_max: %.2f, M_min: %.2f" % (M.max(), M.min())
    plt.figure(title, figsize=(10.7, 10))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.imshow(M, vmin=vmin, vmax=vmax, cmap="coolwarm", origin='lower', aspect="equal")
    plt.xlim([0, M.shape[1]])
    plt.ylim([0, M.shape[1]])

    ax2.imshow(M, vmin=vmin, vmax=vmax, cmap="coolwarm", origin='lower', aspect="equal")
    plt.xlim([0, M.shape[1]])
    plt.ylim([0, M.shape[1]])

    types = np.unique(labels2)
    names = np.unique(labels1)
    tp_ticks, nm_ticks = [], []
    for tp in types:
        for nm in names:
            q = np.argwhere(np.all([labels1 == nm, labels2 == tp], axis=0))
            if len(q) == 0:
                continue
            x0 = np.max(q) + 1
            ax2.plot([0, M.shape[1]], [x0, x0], 'b--', lw=.5)
            ax2.plot([x0, x0], [0, M.shape[1]], 'b--', lw=.5)
            nm_ticks.append([q.mean(), nm])
        q = np.argwhere(labels2 == tp)
        x0 = np.max(q) + 1
        ax2.plot([0, M.shape[1]], [x0, x0], 'k-', lw=1)
        ax2.plot([x0, x0], [0, M.shape[1]], 'k-', lw=1)
        tp_ticks.append([q.mean(), tp])

    tp_ticks = np.array(tp_ticks)
    nm_ticks = np.array(nm_ticks)
    ax1.xaxis.set_ticks(np.float32(nm_ticks[:, 0]))
    ax1.xaxis.set_ticklabels(nm_ticks[:, 1], rotation='vertical')
    ax1.xaxis.set_tick_params(which='major', labelsize=10)
    ax1.yaxis.set_ticks(np.float32(nm_ticks[:, 0]))
    ax1.yaxis.set_ticklabels(nm_ticks[:, 1])
    ax1.yaxis.set_tick_params(which='major', labelsize=10)

    ax2.xaxis.set_ticks(np.float32(nm_ticks[:, 0]))
    ax2.xaxis.set_ticklabels(nm_ticks[:, 1], rotation='vertical')
    ax2.xaxis.set_tick_params(which='major', labelsize=10)
    ax2.yaxis.set_ticks(np.float32(tp_ticks[:, 0]))
    ax2.yaxis.set_ticklabels(tp_ticks[:, 1])
    ax2.yaxis.set_tick_params(which='major', labelsize=10)

    plt.tight_layout()


def corr_matrix(df, mode="all", show=True):
    mask = np.zeros(df.T[5:].shape[0], dtype=bool)
    if mode is not list:
        mode = [mode]
    if None in mode or "all" in mode:
        mask[:] = 1
    if "pretrain" in mode or "trial-1" in mode:
        mask[:200] = 1
    if "training" in mode:
        mask[200:-300] = 1
    if "reversal" in mode:
        mask[-200:] = 1
    if "trial-2" in mode:
        mask[200:400] = 1
    if "trial-3" in mode:
        mask[400:600] = 1
    if "trial-4" in mode:
        mask[600:800] = 1
    if "trial-5" in mode:
        mask[800:1000] = 1
    if "trial-6" in mode:
        mask[1000:1200] = 1
    if "trial-7" in mode:
        mask[1300:1500] = 1
    if "trial-8" in mode:
        mask[1500:1700] = 1
    if "iter-1" in mode:
        mask[0:100] = 1
    if "iter-2" in mode:
        mask[100:200] = 1
    if "iter-3" in mode:
        mask[200:300] = 1
    if "iter-4" in mode:
        mask[300:400] = 1
    if "iter-5" in mode:
        mask[400:500] = 1
    if "iter-6" in mode:
        mask[500:600] = 1
    if "iter-7" in mode:
        mask[600:700] = 1
    if "iter-8" in mode:
        mask[700:800] = 1
    if "iter-9" in mode:
        mask[800:900] = 1
    if "iter-10" in mode:
        mask[900:1000] = 1
    if "iter-11" in mode:
        mask[1000:1100] = 1
    if "iter-12" in mode:
        mask[1100:1200] = 1
    if "iter-13" in mode:
        mask[1200:1300] = 1
    if "iter-14" in mode:
        mask[1300:1400] = 1
    if "iter-15" in mode:
        mask[1400:1500] = 1
    if "iter-16" in mode:
        mask[1500:1600] = 1
    if "iter-17" in mode:
        mask[1600:1700] = 1

    corr = df.T[5:].astype(float).loc[mask].corr()
    names = df['name']
    types = df['type']
    corr.columns = names
    corr.index = names

    if show:
        plot_matrix(corr, title="cc-matrix-%s" % mode[0], vmin=-1., vmax=1.,
                    labels1=names.values.astype('unicode'),
                    labels2=types.values.astype('unicode'))
        plt.show()

    return corr


def plot_covariance(plot_pca_2d=False):

    gens = FruitflyData()

    data_dict = gens.dataset(gens)
    data_dict = sort(data_dict, ['type', 'name'])

    x = data_dict['traces'].reshape((-1, data_dict['traces'].shape[-1]))
    x_max = x.max(axis=1)
    x = (x.T / (x_max + eps)).T

    C = x.T.dot(x) / x.shape[0]
    v = .02
    plot_matrix(C, title="fly-covariance-matrix",
                labels1=data_dict['name'], labels2=data_dict['type'], vmin=-v, vmax=v, verbose=True)

    if plot_pca_2d:
        from sklearn.decomposition import PCA

        pca = PCA(x.shape[1], whiten=False)
        pca.fit(x)
        x_proj = pca.transform(x)

        types = np.unique(data_dict['type'])

        plt.figure("pca-types", figsize=(9, 9))
        colours = {
            "KC": "black",
            "MBON-ND": "grey",
            "MBON-glu": "green",
            "MBON-gaba": "blue",
            "MBON-ach": "red",
            "PAM": "cyan",
            "PPL1": "magenta"
        }
        for t in types:
            x0 = x_proj.T[data_dict['type'] == t]
            plt.scatter(x0[:, 0], x0[:, 1], color=colours[t], marker=".", label=t)
        plt.xlim([-.25, .25])
        plt.ylim([-.25, .25])
        plt.legend()

        plt.figure("pca-location", figsize=(9, 9))
        colours = {
            u"\u03b1": "red",
            u"\u03b2": "green",
            u"\u03b1'": "pink",
            u"\u03b2'": "greenyellow",
            u"\u03b3": "blue"
        }

        for loc in np.sort(colours.keys()):
            q = np.where([loc in name for name in data_dict['name']])
            x0 = x_proj.T[q]
            plt.scatter(x0[:, 0], x0[:, 1], color=colours[loc], marker=".", label=loc)
        plt.xlim([-.25, .25])
        plt.ylim([-.25, .25])
        plt.legend()

    plt.show()


def plot_mutual_information(normalise=False):
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import mutual_info_regression

    dataset = FruitflyData()
    # gens = dataset.slice(types=["DAN", "MBON"], loc=u"\u03b1'")

    data_dict = FruitflyData.dataset(dataset)
    data_dict = sort(data_dict, ['type', 'name'])

    xs = data_dict['traces'].reshape((-1, data_dict['traces'].shape[-1]))
    if normalise:
        x_max = xs.max(axis=1)
        xs = (xs.T / (x_max + eps)).T

    MI = np.zeros((xs.shape[1], xs.shape[1]), dtype=float)
    for i, x in enumerate(xs.T):
        print i,
        for j, y in enumerate(xs.T):
            m = mutual_info_regression(x[..., np.newaxis], y)[0]
            print "%.2f" % m,
            MI[i, j] = m
        print ''

    np.savez_compressed("mi%s.npz" % ("-norm" if normalise else ""), MI=MI)

    v = .5
    plot_matrix(MI, title="fly-mutual-information" + ("-norm" if normalise else ""),
                labels1=data_dict['name'], labels2=data_dict['type'], vmin=-v, vmax=v)
    plt.show()


def plot_f_score(normalise=False):
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import f_regression

    data_dict = sort(FruitflyData.dataset(FruitflyData()), ['type', 'name'])

    xs = data_dict['traces'].reshape((-1, data_dict['traces'].shape[-1]))
    if normalise:
        x_max = xs.max(axis=1)
        xs = (xs.T / (x_max + eps)).T

    F = np.zeros((xs.shape[1], xs.shape[1]), dtype=float)
    for i, x in enumerate(xs.T):
        print i,
        for j, y in enumerate(xs.T):
            f = f_regression(x[..., np.newaxis], y)[0]
            print "%.2f" % f,
            F[i, j] = f
        print ''

    np.savez_compressed("f-score%s.npz" % ("-norm" if normalise else ""), F=F)

    v = .5
    plot_matrix(F, title="fly-f-score" + ("-norm" if normalise else ""),
                labels1=data_dict['name'], labels2=data_dict['type'], vmin=-v, vmax=v)
    plt.show()


def pairplot(df, cols=None):
    if cols is None or not cols:
        cols = []
        types = df['type'].unique().astype('unicode')
        for tp in types:
            cols.append(df[df['type'].values.astype('unicode') == tp].index[0])

    types = df['type'].unique().astype('unicode')

    x = df.T[cols][5:].astype(float)  # type: pd.DataFrame
    x.columns = types

    pp = sns.pairplot(x, size=1.8, aspect=1.8, plot_kws=dict(edgecolor='k', linewidth=0.5),
                      diag_kind='kde', diag_kws=dict(shade=True))
    fig = pp.fig
    fig.subplots_adjust(top=0.93, wspace=0.3)
    fig.show()

    plt.show()


def plot_corr_over_time(df):

    cors = []
    for i in xrange(17):
        c = corr_matrix(df.sort_values(by=['type', 'name']), mode="iter-%d" % (i + 1), show=False)
        cors.append(np.sqrt(np.sum(np.sum(np.square(c)))))

    plt.figure("corr-over-time", figsize=(10, 10))

    cors = np.array(cors)

    plt.plot(np.arange(9), cors[0::2], lw=2, label="CS-")
    plt.plot(np.arange(8) + .5, cors[1::2], lw=2, label="CS+")
    plt.plot([1.5, 2.5, 3.5, 4.5, 5.5, 7, 8], cors[[3, 5, 7, 9, 11, 14, 16]], 'ro', lw=1)

    plt.xticks(np.arange(0, 8.5, .5), [
        "1-", "1+", "2-", "2+", "3-", "3+", "4-", "4+", "5-", "5+", "6-", "6+", "7-", "7+", "8-", "8+", "9-"])
    plt.ylim([170, 255])
    plt.xlabel("Trial")
    plt.ylabel(r'$\sqrt{\sum{C^2}}$')
    plt.legend()


def plot_traces(df, title="traces", vmin=-20, vmax=20, normalise=False, verbose=False):
    if verbose:
        print "M_max: %.2f, M_min: %.2f" % (df.max(), df.min())

    labels1 = df['name'].values.astype('unicode')
    labels2 = df['type'].values.astype('unicode')

    plt.figure(title, figsize=(20, 10))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    img = df.T[5:].T.astype(float)
    if normalise:
        img_max = img.max(axis=1)
        img = (img.T / (img_max + eps)).T
        vmin = -1
        vmax = 1
    ax1.imshow(img, vmin=vmin, vmax=vmax, cmap="coolwarm",
               interpolation='nearest', origin="lower", aspect="auto")
    plt.xlim([0, img.shape[1]])
    plt.ylim([0, img.shape[0]])

    ax2.imshow(img, vmin=vmin, vmax=vmax, cmap="coolwarm",
               interpolation='nearest', origin="lower", aspect="auto")
    plt.xlim([0, img.shape[1]])
    plt.ylim([0, img.shape[0]])

    types = np.unique(labels2)
    names = np.unique(labels1)
    tp_ticks, nm_ticks = [], []
    for tp in types:
        for nm in names:
            q = np.argwhere(np.all([labels1 == nm, labels2 == tp], axis=0))
            if len(q) == 0:
                continue
            x0 = np.max(q) + 1
            ax2.plot([0, img.shape[1]], [x0, x0], 'b--', lw=.5)
            nm_ticks.append([q.mean(), nm])
        q = np.argwhere(labels2 == tp)
        x0 = np.max(q) + 1
        ax2.plot([0, img.shape[1]], [x0, x0], 'k-', lw=1)
        tp_ticks.append([q.mean(), tp])

    x_ticks = []
    shock = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    for trial, s in zip(xrange(img.shape[1] / 100), shock):
        plt.plot([trial * 100, trial * 100], [0, img.shape[0]], 'k-', lw=.5)
        if s:
            plt.plot([trial * 100 + 45, trial * 100 + 45], [0, img.shape[0]], 'r-', lw=.5)

        x_ticks.append([trial * 100 + 50, "Trial %d\nCS%s" % (trial / 2 + 1, '-' if trial % 2 == 0 else '+')])

    tp_ticks = np.array(tp_ticks)
    nm_ticks = np.array(nm_ticks)
    x_ticks = np.array(x_ticks)
    # plt.yticks(np.float32(nm_ticks[:, 0]), nm_ticks[:, 1])
    # plt.xticks(np.float32(x_ticks[:, 0]), x_ticks[:, 1])

    ax1.xaxis.set_ticks(np.float32(x_ticks[:, 0]))
    ax1.xaxis.set_ticklabels(x_ticks[:, 1])
    ax1.xaxis.set_tick_params(which='major', labelsize=10)
    ax1.yaxis.set_ticks(np.float32(nm_ticks[:, 0]))
    ax1.yaxis.set_ticklabels(nm_ticks[:, 1])
    ax1.yaxis.set_tick_params(which='major', labelsize=10)

    ax2.xaxis.set_ticks(np.float32(x_ticks[:, 0]))
    ax2.xaxis.set_ticklabels(x_ticks[:, 1])
    ax2.xaxis.set_tick_params(which='major', labelsize=10)
    ax2.yaxis.set_ticks(np.float32(tp_ticks[:, 0]))
    ax2.yaxis.set_ticklabels(tp_ticks[:, 1])
    ax2.yaxis.set_tick_params(which='major', labelsize=10)

    plt.tight_layout()
