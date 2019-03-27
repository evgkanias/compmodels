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
        names = recs.keys()
        odours = []
        raw_data = {}
        for name in names:
            genotype, odour = re.findall(r'([\d\w\W]+)-([\w\W]+)', name)[0]
            if genotype not in meta.keys():
                continue
            if genotype not in raw_data.keys():
                raw_data[genotype] = meta[genotype]
            if 'M+S' in odour:
                raw_data[genotype]['cs+'] = odour
                raw_data[genotype]['cs+traces'] = recs[name]
            else:
                raw_data[genotype]['cs-'] = odour
                raw_data[genotype]['cs-traces'] = recs[name]
            if odour not in odours:
                odours.append(odour)
            raw_data[genotype]['loc'] = self.__name2location(meta[genotype]['name'])

        super(DataFrame, self).__init__(raw_data)

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

        return FruitflyData(**gens_ret)

    @staticmethod
    def dataset(data):

        csp, csm, csp_types, csm_types = [], [], [], []
        genotypes, names, types = [], [], []
        for name in data.keys():
            csp.append(data[name]['cs+traces'])
            csm.append(data[name]["cs-traces"])
            csp_types.append([data[name]["cs+"]] * csp[-1].shape[-1])
            csm_types.append([data[name]["cs-"]] * csp[-1].shape[-1])
            genotypes.append([name] * csp[-1].shape[-1])
            names.append([data[name]['name']] * csp[-1].shape[-1])
            types.append([data[name]['type']] * csp[-1].shape[-1])

        csp = np.concatenate(csp, axis=2)
        csm = np.concatenate(csm, axis=2)

        keys = ["genotype", "name", "type", "CS+ type", "CS- type"] + [s % (i/200 + 1) for i, s in enumerate((
             ["%s.%d-" % ("%d", t) for t in xrange(100)] + ["%s.%d+" % ("%d", t) for t in xrange(100)]
        ) * 8)] + ["9.%d-" % t for t in xrange(100)]

        return pd.DataFrame(np.concatenate([
            np.concatenate(genotypes)[np.newaxis], np.concatenate(names)[np.newaxis], np.concatenate(types)[np.newaxis],
            np.concatenate(csp_types)[np.newaxis], np.concatenate(csm_types)[np.newaxis],
            csm[0, ...], csp[0, ...], csm[1, ...], csp[1, ...], csm[2, ...], csp[2, ...],
            csm[3, ...], csp[3, ...], csm[4, ...], csp[4, ...], csm[5, ...], csp[5, ...],
            csm[6, ...], csp[6, ...], csm[7, ...], csp[7, ...], csm[8, ...]
        ], axis=0), index=keys).T

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

    recs = load_data()
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

    data_dict = sort(FruitflyData.dataset(FruitflyData()), ['type', 'name'])

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

    df = DataFrame.dataset(DataFrame())
    # plot_traces(df.sort_values(by=['type', 'name']), normalise=True)
    # print sort(df, ['type', 'name'])

    plot_corr_over_time(df)
    # pairplot(df.sort_values(by=['type', 'name']))

    # x = df[5:].astype(float)
    # pp = sns.pairplot(x[::150], size=1.8, aspect=1.8, plot_kws=dict(edgecolor="k", linewidth=.5),
    #                   diag_kind="kde", diag_kws=dict(shade=True))
    # fig = pp.fig
    # fig.subplots_adjust(top=0.93, wspace=0.3)
    plt.show()
