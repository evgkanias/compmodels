import os
import re
import csv
import pandas as pd
import numpy as np
import yaml
import string


__dir__ = os.path.dirname(os.path.abspath(__file__))
__data_dir__ = os.path.join(__dir__, "FruitflyMB")

with open(os.path.join(__data_dir__, 'meta.yaml'), 'rb') as f:
    meta = yaml.load(f, Loader=yaml.BaseLoader)
eps = np.finfo(float).eps

#
# class FruitflyData(dict):
#
#     comps = {
#         u"\u03b1": [['p', 'c', 's']] * 3,
#         u"\u03b2": [['p', 'c', 's']] * 2,
#         u"\u03b1'": [['m', 'p', 'a']] * 3,
#         u"\u03b2'": [['m', 'p', 'a']] * 2,
#         u"\u03b3": [['m', 'd']] * 5
#     }
#
#     def __init__(self, **kwargs):
#         super(FruitflyData, self).__init__(**kwargs)
#         if kwargs:
#             pass
#
#         recs = load_data()
#         names = recs.keys()
#         self._odours = []
#         self._compartments = []
#         for name in names:
#             genotype, odour = re.findall(r'([\d\w\W]+)-([\w\W]+)', name)[0]
#             if genotype not in meta.keys():
#                 continue
#             if genotype not in self.keys():
#                 self[genotype] = meta[genotype]
#             self[genotype][odour] = recs[name]
#             if odour not in self._odours:
#                 self._odours.append(odour)
#             self[genotype]['loc'] = self.__name2location(meta[genotype]['name'])
#
#             # print meta[genotype]['type'], meta[genotype]['name'], self[genotype]['loc']
#
#     def __name2location(self, name):
#         """
#         '/' = or
#         '<' = from
#         '>' = to
#         :param name:
#         :return:
#         """
#
#         pedc = False
#         calyx = False
#         if 'pedc' in name:
#             pedc = True
#             name = string.replace(name, 'pedc', '')
#         if 'calyx' in name:
#             calyx = True
#             name = string.replace(name, 'calyx', '')
#
#         comps = []
#         for comp in re.findall(r'(\W\'{0,1}\d{0,1}\w*)', name):
#
#             cs = []
#             for c in comp:
#                 if c == "'":
#                     cs[-1] = cs[-1] + c
#                 elif c in self.comps.keys():
#                     cs.append(c)
#                 else:
#                     cs.append(c)
#
#             if len(cs) > 3:
#                 for c in cs[2:]:
#                     if c.isdigit():
#                         cs2 = [cs[0]]
#                     if 'cs2' in locals():
#                         cs2.append(c)
#                 if 'cs2' in locals():
#                     for c in cs2[1:]:
#                         cs.remove(c)
#                     if len(cs2) > 1 and cs2[1].isdigit():
#                         cs2[1] = int(cs2[1])
#             if len(cs) > 1 and cs[1].isdigit():
#                 cs[1] = int(cs[1])
#             comps.append(cs)
#             if 'cs2' in locals():
#                 comps.append(cs2)
#         if pedc:
#             comps.append(['pedc'])
#         if calyx:
#             comps.append(['calyx'])
#
#         return comps
#
#     def slice(self, types=None, loc=None, odours=None):
#
#         gens_ret = self.copy()
#
#         if types is not None:
#             if not isinstance(types, list):
#                 types = [types]
#             types_ = types[:]
#
#             for type_i in types_:
#                 if type_i.lower() in ["dan"]:
#                     types.remove(type_i)
#                     types.append("PAM")
#                     types.append("PPL1")
#                 if type_i.lower() in ["mbon"]:
#                     types.remove(type_i)
#                     types.append("MBON-glu")
#                     types.append("MBON-ach")
#                     types.append("MBON-gaba")
#                 elif type_i.lower() in ["glutamine", "glu"]:
#                     types.remove(type_i)
#                     types.append("MBON-glu")
#                 elif type_i.lower() in ["cholimergic", "ach"]:
#                     types.remove(type_i)
#                     types.append("MBON-ach")
#                 elif type_i.lower() in ["gaba"]:
#                     types.remove(type_i)
#                     types.append("MBON-gaba")
#
#             gens = gens_ret.copy()
#             gens_ret.clear()
#             for type_i in types:
#                 for genotype in gens.keys():
#                     if gens[genotype]['type'].lower() == type_i.lower():
#                         gens_ret[genotype] = gens[genotype]
#
#         if odours is not None:
#             if not isinstance(odours, list):
#                 odours = [odours]
#             odours_ = odours[:]
#
#             for odour in odours_:
#                 if "ROI" not in odour:
#                     odours.append(odour + "_bigROI")
#                     odours.append(odour + "_bROI")
#                     odours.append(odour + "_sROI")
#
#             gens = gens_ret.copy()
#             gens_ret.clear()
#             for odour in odours:
#                 for genotype in gens.keys():
#                     if odour in gens[genotype].keys():
#                         gens_ret[genotype] = gens[genotype]
#
#         if loc is not None:
#             locs = self.__name2location(loc)
#
#             for loc in locs:
#                 for l, _ in enumerate(loc):
#                     gens = gens_ret.copy()
#                     gens_ret.clear()
#                     for genotype in gens.keys():
#                         for comp in gens[genotype]['loc']:
#                             if np.all([loc[ll] in comp for ll in range(l+1)]):
#                                 gens_ret[genotype] = gens[genotype]
#                                 continue
#
#         return FruitflyData(**gens_ret)
#
#     @staticmethod
#     def dataset(data):
#
#         csp, csm = [], []
#         genotypes, names, types = [], [], []
#         for name in data.keys():
#             keys = data[name].keys()
#             for key in keys:
#                 if "O" in key or "A" in key or "B" in key or "M+S" in key:
#                     res = data[name][key]
#                     if "M+S" in key:
#                         csp.append(res)
#                     else:
#                         csm.append(res)
#             genotypes.append([name] * res.shape[-1])
#             names.append([data[name]['name']] * res.shape[-1])
#             types.append([data[name]['type']] * res.shape[-1])
#
#         csp = np.concatenate(csp, axis=2)
#         csm = np.concatenate(csm, axis=2)
#
#         return {
#             'genotype': np.concatenate(genotypes),
#             'name': np.concatenate(names),
#             'type': np.concatenate(types),
#             'traces': np.concatenate([
#                     csm[:1, ...], csp[:1, ...],
#                     csm[1:2, ...], csp[1:2, ...],
#                     csm[2:3, ...], csp[2:3, ...],
#                     csm[3:4, ...], csp[3:4, ...],
#                     csm[4:5, ...], csp[4:5, ...],
#                     csm[5:6, ...], csp[5:6, ...],
#                     csm[6:7, ...], csp[6:7, ...],
#                     csm[7:8, ...], csp[7:8, ...],
#                     csm[8:9, ...]
#                 ], axis=0)
#         }


class DataFrame(pd.DataFrame):

    comps = {
        u"\u03b1": [['p', 'c', 's']] * 3,
        u"\u03b2": [['p', 'c', 's']] * 2,
        u"\u03b1'": [['m', 'p', 'a']] * 3,
        u"\u03b2'": [['m', 'p', 'a']] * 2,
        u"\u03b3": [['m', 'd']] * 5
    }

    def __init__(self, **kwargs):
        if kwargs:
            super(DataFrame, self).__init__(**kwargs)
            pass

        recs = self.load_data()
        raw_data = {}
        for name in recs.keys():
            genotype, odour = re.findall(r'([\d\w\W]+)-([\w\W]+)', name)[0]
            if genotype not in meta.keys():
                continue
            if genotype not in raw_data.keys():
                raw_data[genotype] = meta[genotype]
            if 'M+S' in odour:
                raw_data[genotype]['CS+'] = odour
                raw_data[genotype]['cs+traces'] = recs[name]
            else:
                raw_data[genotype]['CS-'] = odour
                raw_data[genotype]['cs-traces'] = recs[name]

        ids, genotypes, names, types, trials, odours, conditions, traces = [], [], [], [], [], [], [], []
        for j, name in enumerate(raw_data.keys()):
            for i in xrange(9):
                nb_flies = raw_data[name]['cs-traces'].shape[-1]
                trials.append([i+1] * nb_flies)
                odours.append([raw_data[name]["CS-"]] * nb_flies)
                conditions.append(['CS-'] * nb_flies)
                types.append([raw_data[name]['type']] * nb_flies)
                genotypes.append([name] * nb_flies)
                names.append([raw_data[name]['name']] * nb_flies)
                traces.append(raw_data[name]['cs-traces'][i])
                ids.append(j * nb_flies + np.arange(nb_flies) + 1)

            for i in xrange(8):
                nb_flies = raw_data[name]['cs+traces'].shape[-1]
                trials.append([i+1] * nb_flies)
                odours.append([raw_data[name]["CS+"]] * nb_flies)
                conditions.append(['CS+'] * nb_flies)
                types.append([raw_data[name]['type']] * nb_flies)
                genotypes.append([name] * nb_flies)
                names.append([raw_data[name]['name']] * nb_flies)
                traces.append(raw_data[name]['cs+traces'][i])
                ids.append(j * nb_flies + np.arange(nb_flies) + 1)

        genotypes = np.concatenate(genotypes)[np.newaxis]
        names = np.concatenate(names)[np.newaxis]
        types = np.concatenate(types)[np.newaxis]
        trials = np.concatenate(trials)[np.newaxis]
        odours = np.concatenate(odours)[np.newaxis]
        conditions = np.concatenate(conditions)[np.newaxis]
        ids = np.concatenate(ids)[np.newaxis]
        traces = np.concatenate(traces, axis=-1)

        keys = ["type", "condition", "name", "genotype", "odour", "trial", "id"] + list(np.linspace(0.2, 20, 100))
        dat = np.concatenate([types, conditions, names, genotypes, odours, trials, ids, traces], axis=0)
        types = ['unicode'] * 5 + [int] * 2 + [float] * 100

        dat_dict = {}
        for key, d, t in zip(keys, dat, types):
            dat_dict[key] = d.astype(t)
        raw_data = pd.DataFrame(dat_dict)

        raw_data.set_index(["type", "condition", "name", "genotype", "odour", "trial", "id"], inplace=True)

        raw_data.columns = pd.MultiIndex(levels=[raw_data.columns.astype(float).values, [False, True]],
                                         codes=[range(100), [0] * 45 + [1] + [0] * 54],
                                         names=['time', 'shock'])

        super(DataFrame, self).__init__(raw_data.astype(float))

    def __name2location(self, name):
        """
        '/' = or
        '<' = from
        '>' = to
        :param name:
        :return:
        """

        pedc = False
        calyx = False
        if 'pedc' in name:
            pedc = True
            name = string.replace(name, 'pedc', '')
        if 'calyx' in name:
            calyx = True
            name = string.replace(name, 'calyx', '')

        comps = []
        for comp in re.findall(r'(\W\'{0,1}\d{0,1}\w*)', name):

            cs = []
            for c in comp:
                if c == "'":
                    cs[-1] = cs[-1] + c
                elif c in self.comps.keys():
                    cs.append(c)
                else:
                    cs.append(c)

            if len(cs) > 3:
                for c in cs[2:]:
                    if c.isdigit():
                        cs2 = [cs[0]]
                    if 'cs2' in locals():
                        cs2.append(c)
                if 'cs2' in locals():
                    for c in cs2[1:]:
                        cs.remove(c)
                    if len(cs2) > 1 and cs2[1].isdigit():
                        cs2[1] = int(cs2[1])
            if len(cs) > 1 and cs[1].isdigit():
                cs[1] = int(cs[1])
            comps.append(cs)
            if 'cs2' in locals():
                comps.append(cs2)
        if pedc:
            comps.append(['pedc'])
        if calyx:
            comps.append(['calyx'])

        return comps

    def slice(self, types=None, loc=None, odours=None):

        gens_ret = self.copy()

        if types is not None:
            if not isinstance(types, list):
                types = [types]
            types_ = types[:]

            for type_i in types_:
                if type_i.lower() in ["dan"]:
                    types.remove(type_i)
                    types.append("PAM")
                    types.append("PPL1")
                if type_i.lower() in ["mbon"]:
                    types.remove(type_i)
                    types.append("MBON-glu")
                    types.append("MBON-ach")
                    types.append("MBON-gaba")
                elif type_i.lower() in ["glutamine", "glu"]:
                    types.remove(type_i)
                    types.append("MBON-glu")
                elif type_i.lower() in ["cholimergic", "ach"]:
                    types.remove(type_i)
                    types.append("MBON-ach")
                elif type_i.lower() in ["gaba"]:
                    types.remove(type_i)
                    types.append("MBON-gaba")

            gens = gens_ret.copy()
            gens_ret.clear()
            for type_i in types:
                for genotype in gens.keys():
                    if gens[genotype]['type'].lower() == type_i.lower():
                        gens_ret[genotype] = gens[genotype]

        if odours is not None:
            if not isinstance(odours, list):
                odours = [odours]
            odours_ = odours[:]

            for odour in odours_:
                if "ROI" not in odour:
                    odours.append(odour + "_bigROI")
                    odours.append(odour + "_bROI")
                    odours.append(odour + "_sROI")

            gens = gens_ret.copy()
            gens_ret.clear()
            for odour in odours:
                for genotype in gens.keys():
                    if odour in gens[genotype].keys():
                        gens_ret[genotype] = gens[genotype]

        if loc is not None:
            locs = self.__name2location(loc)

            for loc in locs:
                for l, _ in enumerate(loc):
                    gens = gens_ret.copy()
                    gens_ret.clear()
                    for genotype in gens.keys():
                        for comp in gens[genotype]['loc']:
                            if np.all([loc[ll] in comp for ll in range(l+1)]):
                                gens_ret[genotype] = gens[genotype]
                                continue

        return DataFrame(**gens_ret)

    @property
    def unstacked(self):
        return DataFrame.pivot_trials(self)

    @property
    def normalised(self):
        return DataFrame.normalise(self)

    @staticmethod
    def pivot_trials(df):
        # drop the 'odour' index
        dff = df.reset_index(level='odour', drop=True)

        # reorder the levels of indices in the 'index' axis so that the 'trial' index is last
        dff = dff.reorder_levels(['type', 'name', 'genotype', 'id', 'condition', 'trial'], axis=0)  # type: DataFrame

        # unstack the last level: the last level changes axis for 'index' to 'column'
        dff = dff.unstack([-2, -1])  # type: DataFrame

        # reorder the levels of indices in the 'columns' axis
        dff = dff.reorder_levels(['trial', 'condition', 'time', 'shock'], axis=1)  # type: DataFrame

        # sort the indices in the 'columns' axis
        dff = dff.sort_index(axis=1, level=['trial', 'condition', 'time'], ascending=[True, False, True])

        # sort the indices in the 'index' axis
        dff = dff.sort_index(axis=0, level=['type', 'name', 'genotype', 'id'])  # type: DataFrame

        # drop the last trial (CS+ trial 9) which does not exist
        dff = dff.T[:-100].T

        return dff  # type: DataFrame

    @staticmethod
    def normalise(df):
        x_max = np.max([df.max(axis=1), -df.min(axis=1)], axis=0)
        return (df.T / (x_max + eps)).T

    @staticmethod
    def load_data():
        data = {}
        for filename in os.listdir(__data_dir__):
            pattern = r'realSCREEN_([\d\w\W]+)\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv'
            details = re.findall(pattern, filename)
            if len(details) == 0:
                continue

            genotype, odour, trial = details[0]
            trial = int(trial)

            # print genotype, odour, trial

            timepoint = None
            filename = os.path.join(__data_dir__, filename)
            with open(filename, 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
                for row in reader:
                    if timepoint is None:
                        timepoint = row
                    else:
                        timepoint = np.vstack([timepoint, row])  # {timepoint} x {datapoint}

            name = "%s-%s" % (genotype, odour)
            if name not in data.keys():
                data[name] = [[]] * 9
            data[name][trial-1] = timepoint

        for name in data.keys():
            data[name] = np.array(data[name])

        return data


def plot_data(save=False, show=True):

    import matplotlib.pyplot as plt

    recs = DataFrame.load_data()
    names = recs.keys()
    figs = []
    genotypes = []

    for name in names:
        print name
        genotype, odour = re.findall(r'([\d\w\W]+)-([\w\W]+)', name)[0]

        fig = plt.figure("%s" % genotype, figsize=(18, 6))
        if genotype not in genotypes:
            genotypes.append(genotype)
            figs.append(fig)

        for i, trial in enumerate(recs[name]):
            if "M+S" in odour:
                plt.subplot(2, 9, i + 10)
            else:
                plt.subplot(2, 9, i + 1)
            plt.plot(trial)
            plt.ylim([-10, 70])
            # plt.title("%s %s" % (genotype, odour))

    for genotype, fig in zip(genotypes, figs):
        fig.suptitle(genotype)
        if save:
            fig.savefig("%s.eps" % genotype)
        if show:
            plt.show()

    return genotypes, figs


def hierarchical_clustering(normalise=False, plot_pca_2d=False):
    from sklearn.cluster import AgglomerativeClustering

    data_dict = sort(DataFrame.dataset(DataFrame()), ['type', 'name'])

    x = data_dict['traces'].reshape((-1, data_dict['traces'].shape[-1]))
    if normalise:
        x_max = x.max(axis=1)
        x = (x.T / (x_max + eps)).T

    cluster = AgglomerativeClustering(n_clusters=7)
    cluster.fit(x.T)

    if plot_pca_2d:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        pca = PCA(x.shape[1], whiten=False)
        pca.fit(x)
        x_proj = pca.transform(x)

        plt.figure("pca-types", figsize=(9, 9))
        colours = ["black", "grey", "green", "blue", "red", "cyan", "magenta"]

        for row in xrange(10):
            for col in xrange(10):
                plt.subplot(10, 10, row * 10 + col + 1)
                for i, c in enumerate(colours):
                    x0 = x_proj.T[cluster.labels_ == i]
                    plt.scatter(x0[:, row], x0[:, col], color=c, marker=".", label="cluster-%d" % (i + 1))
                plt.xlim([-3., 3.])
                plt.ylim([-3., 3.])
        plt.legend()

        plt.show()


def sort(dset, orders):
    """

    :param dset:
    :type dset: np.DataFrame
    :param orders:
    :type orders: list
    :return: np.ndarray
    """

    if not orders:
        return dset

    keys = dset[orders[0]].values.astype('unicode')
    j = np.argsort(keys)
    dset = dset.T[j].T

    if len(orders) < 2:
        return dset

    keys0 = np.unique(keys)
    keys1 = dset[orders[1]].values.astype('unicode')
    for key0 in keys0:
        i = dset[orders[0]].values == key0
        ii = np.argsort(keys1[i])
        for key in dset.columns:
            dset[key][i] = dset[key][i][ii]
    return dset


def sort_corr(C, names=None, types=None):
    C['id'] = range(len(names))
    C = C.set_index('id').T
    C['id'] = range(len(names))
    C = C.set_index('id').T

    C_sorted = C.copy().abs()
    indices = C.index.values

    C_sorted = C_sorted.sort_values(by=list(indices), axis=0, ascending=False)
    columns = C.columns.values
    C_sorted = C_sorted.sort_values(by=list(columns), axis=1, ascending=False)

    i = C_sorted.index.tolist()
    C_sorted['name'] = names[i].astype('unicode')
    C_sorted = C_sorted.set_index('name').T
    C_sorted['name'] = names[i].astype('unicode')
    C_sorted = C_sorted.set_index('name').T


if __name__ == "__main__":
    from visualisation import *
    import matplotlib.pyplot as plt
    # plot_covariance()
    # plot_f_score(normalise=True)
    # plot_mutual_information(normalise=True)
    # hierarchical_clustering(plot_pca_2d=True)

    # data_dict = sort(FruitflyData.dataset(FruitflyData()), ['type', 'name'])
    #
    # F = np.load("f-score-norm.npz")["F"]
    # print F.max(), F.min()
    #
    # v = 1e+2
    # plot_matrix(F, title="fly-f-score-norm",
    #             labels1=data_dict['name'], labels2=data_dict['type'], vmin=-v, vmax=v)
    # plt.show()

    df = DataFrame()
    dff = df.unstacked

    # A tale of the three MBONs
    cond = []
    for group in [[r'PPL1', u'\u03b31pedc'], [r'MBON-GABA', u'\u03b31pedc'],
                  [r'PAM', u"\u03b2'2m"], [r'MBON-Glu', u"\u03b2'2mp"],
                  [r'PAM', u"\u03b2'2a"], [r'MBON-Glu', u"\u03b35\u03b2'2a"]]:
        c = np.all([dff.index.get_level_values(t) == g for t, g in zip(["type", "name"], group)], axis=0)
        cond.append(c)
    dff = dff.iloc[np.any(cond, axis=0)]
    print dff

    # print dff
    # plot_traces(dff.sort_index(axis=0, level=['type', 'name']), diff=False, normalise=True)
    # plot_overall_response(dff.sort_index(axis=0, level=['type', 'name']), diff=False, normalise=True)

    # for i in range(0, 9):
    #     corr_matrix(dff.sort_index(axis=0, level=['type', 'name']), diff=False, shock=False,
    #                 mode="iter-%d" % (i * 2 + 1))
    #
    # for i in range(0, 8):
    #     corr_matrix(dff.sort_index(axis=0, level=['type', 'name']), diff=False, shock=False,
    #                 mode="iter-%d" % (i * 2 + 2))

    # for i in range(1, 8):
    #     cross_corr_matrix(dff.sort_index(axis=0, level=['type', 'name']), diff=True,
    #                       mode1="iter-%d" % (i * 2 + 1), mode2="iter-%d" % (i * 2 + 3))
    #
    # for i in range(1, 7):
    #     cross_corr_matrix(dff.sort_index(axis=0, level=['type', 'name']), diff=True,
    #                       mode1="iter-%d" % (i * 2 + 2), mode2="iter-%d" % (i * 2 + 4))

    # C, names, types = corr_matrix(dff.sort_index(axis=0, level=['type', 'name']),
    #                               mode="all", avg=True, diff=True, show=True)  # type: DataFrame
    for i in range(17):
        corr_matrix(dff.sort_index(axis=0, level=['type', 'name']),
                    mode=["iter-%d" % (i + 1)],
                    shock=False, diff=True, avg=True, figsize=(3.8, 3))
    # plot_mutual_information(dff.sort_index(axis=0, level=['type', 'name']), diff=True)
    # plot_f_score(dff.sort_index(axis=0, level=['type', 'name']), diff=True)
    # plot_corr_over_time(dff, shock=False)
    # plot_cross_corr_over_time(dff, shock=False)
    # plot_iter_corr_matrix(dff, shock=False)
    # plot_iter_corr_matrix(dff, sort_by=['condition', 'trial'], ascending=[False, True], shock=False)
    # pairplot(df.sort_values(by=['type', 'name']))

    # x = df[5:].astype(float)
    # pp = sns.pairplot(x[::150], size=1.8, aspect=1.8, plot_kws=dict(edgecolor="k", linewidth=.5),
    #                   diag_kind="kde", diag_kws=dict(shade=True))
    # fig = pp.fig
    # fig.subplots_adjust(top=0.93, wspace=0.3)

    # plot_matrix(C_sorted, title="cc-matrix-sorted",
    #             labels1=C_sorted.index.values.astype('unicode'))

    # for group in ["KC", "MBON-ND", "MBON-Glu", "MBON-GABA", "MBON-ACh", "PAM", "PPL1"]:
    #     plot_traces_over_time(dff, diff=True, group=group, normalise=True, shock=False, merge=7)

    # for group in [[[r'PPL1', u'\u03b31pedc']], [[r'MBON-GABA', u'\u03b31pedc']],
    #               [[r'PAM', u"\u03b2'2m"]], [[r'MBON-Glu', u"\u03b2'2mp"]],
    #               [[r'PAM', u"\u03b2'2a"]], [[r'MBON-Glu', u"\u03b35\u03b2'2a"]]]:
    #     plot_traces_over_time(dff, diff=True, group=group, normalise=False, shock=False, types=["type", "name"])
    #
    # plt.tight_layout()
    plt.show()
