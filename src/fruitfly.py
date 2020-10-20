import os
import re
import csv
import pandas as pd
import numpy as np
import yaml
import string

from itertools import product
from plot3mbon import *


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
                                         codes=[range(100), [0] * 44 + [1] * 5 + [0] * 51],
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

    def plot6neuron(self, version=2):
        plot_hist_func = plot_hist_old if version < 2 else plot_hist
        plot_hist_func(**_features2hist(_frame2features(self.unstacked, version=version), version=version))

    @property
    def unstacked(self):
        return DataFrame.pivot_trials(self)

    @property
    def normalised(self):
        return DataFrame.normalise(self)

    @property
    def dataset6neuron(self):
        return _frame2dataset(self.unstacked, version=2)

    @property
    def dataset6neuron_old(self):
        return _frame2dataset(self.unstacked, version=1)

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


def _frame2features(df, version=2):
    dan_types = [r"PPL1", r"PAM", r"PAM"]
    dan_names = [u"\u03b31pedc", u"\u03b2'2m", u"\u03b2'2a"]
    mbon_types = [r"MBON-GABA", r"MBON-Glu", r"MBON-Glu"]
    mbon_names = [u"\u03b31pedc", u"\u03b2'2mp", u"\u03b35\u03b2'2a"]

    cond = []
    for group in [[dan_types[0], dan_names[0]], [mbon_types[0], mbon_names[0]],
                  [dan_types[1], dan_names[1]], [mbon_types[1], mbon_names[1]],
                  [dan_types[2], dan_names[2]], [mbon_types[2], mbon_names[2]]]:
        c = np.all([df.index.get_level_values(t) == g for t, g in zip(["type", "name"], group)], axis=0)
        cond.append(c)

    data_i = df.iloc[np.any(cond, axis=0)]
    t0 = np.concatenate([np.ones(25), np.zeros(75)] * 17).astype(bool)
    t1 = np.concatenate([np.zeros(25), np.ones(20), np.zeros(55)] * 17).astype(bool)
    m2 = np.concatenate([np.zeros(45), np.ones(5), np.zeros(50)] * 17).astype(bool)
    if version > 1:
        t3 = np.concatenate([np.zeros(50), np.ones(25), np.zeros(25)] * 17).astype(bool)

    data_t0 = data_i.T.iloc[t0].groupby(["trial", "condition"], axis=0).mean().T
    data_t1 = data_i.T.iloc[t1].groupby(["trial", "condition"], axis=0).mean().T
    data_t2 = data_i.T.iloc[m2].groupby(["trial", "condition"], axis=0).mean().T
    data = [data_t0, data_t1, data_t2]
    keys = ["pre-odour", "odour", "shock"]
    if version > 1:
        data_t3 = data_i.T.iloc[t3].groupby(["trial", "condition"], axis=0).mean().T
        data += [data_t3]
        keys += ["post-odour"]
    names = ["time", "trial", "condition"]

    dff = pd.concat(data, axis=1, names=names, keys=keys).reorder_levels(
        ["trial", "condition", "time"], axis=1).sort_index(
        axis=1, level=["trial", "condition"], ascending=[True, False])  # pd.DataFrame

    return dff


def _features2hist(df, version=2):
    dan_types = [r"PPL1", r"PAM", r"PAM"]
    dan_names = [u"\u03b31pedc", u"\u03b2'2m", u"\u03b2'2a"]
    mbon_types = [r"MBON-GABA", r"MBON-Glu", r"MBON-Glu"]
    mbon_names = [u"\u03b31pedc", u"\u03b2'2mp", u"\u03b35\u03b2'2a"]

    hist = {
        "names": {},
        "m1": [], "m2": [], "m3": [],
        "d1": [], "d2": [], "d3": [],
        "m1s": [], "m2s": [], "m3s": [],
        "d1s": [], "d2s": [], "d3s": [],
    }

    df_avg = df.groupby(["type", "name"]).mean()
    ts = 3 + int(version > 1)

    for n, nt, nn in zip(["m1", "m2", "m3", "d1", "d2", "d3"],
                         mbon_types + dan_types, mbon_names + dan_names):
        nv = df_avg.T[nt, nn].T.to_numpy(dtype=float)
        hist["names"][n] = u"%s-%s" % (nt if '-' not in nt else 'MBON', nn)
        hist[n] = np.zeros((17, 2))
        hist[n][0::2, 0] = nv[2::ts][0::2]
        hist[n][1::2, 0] = (nv[2::ts][0:-2:2] + nv[2::ts][2::2]) / 2.
        hist[n][0, 1], hist[n][-1, 1] = nv[2::ts][1], nv[2::ts][-2]
        hist[n][1::2, 1] = nv[2::ts][1::2]
        hist[n][2:-2:2, 1] = (nv[2::ts][1:-2:2] + nv[2::ts][3::2]) / 2.

        hist[n + "s"] = np.zeros((17, ts, 2))
        hist[n + "s"][0::2, :, 0] = nv.reshape((17, ts))[0::2]
        hist[n + "s"][1::2, :, 0] = nv.reshape((17, ts))[0:-1:2][:, 0][..., np.newaxis]
        hist[n + "s"][1::2, :, 1] = nv.reshape((17, ts))[1::2]
        hist[n + "s"][0, :, 1] = nv.reshape((17, ts))[3, 0]
        hist[n + "s"][2::2, :, 1] = nv.reshape((17, ts))[1::2][:, 0][..., np.newaxis]
        hist[n + "s"] = hist[n + "s"].reshape((-1, 2))

    return hist


def _frame2dataset(df, version=2):
    dan_types = [r"PPL1", r"PAM", r"PAM"]
    dan_names = [u"\u03b31pedc", u"\u03b2'2m", u"\u03b2'2a"]
    mbon_types = [r"MBON-GABA", r"MBON-Glu", r"MBON-Glu"]
    mbon_names = [u"\u03b31pedc", u"\u03b2'2mp", u"\u03b35\u03b2'2a"]

    df = _frame2features(df, version=version)

    dff = None  # pd.DataFrame
    for t, n in zip(mbon_types + dan_types, mbon_names + dan_names):
        dfff = df.T[t, n].T  # pd.DataFrame
        dfff = dfff.T.set_index(np.repeat(t, dfff.shape[1]), append=True).T
        dfff = dfff.T.set_index(np.repeat(n, dfff.shape[1]), append=True).T
        dfff = dfff.reorder_levels([3, 4, 0, 1, 2], axis=1)
        dfff.columns.set_names([u'type', u'name', u'trial', u'condition', u'time'], inplace=True)
        dfff['key'] = np.ones(dfff.shape[0])
        if dff is None:
            dff = dfff
        else:
            dff = pd.merge(dff, dfff, on=["key"])
    dff.drop(columns="key", inplace=True)
    return dff


class ThreeMBONNet(object):

    def __init__(self):
        self.dan_types = [r"PPL1", r"PAM", r"PAM"]
        self.dan_names = [u"\u03b31pedc", u"\u03b2'2m", u"\u03b2'2a"]
        self.mbon_types = [r"MBON-GABA", r"MBON-Glu", r"MBON-Glu"]
        self.mbon_names = [u"\u03b31pedc", u"\u03b2'2mp", u"\u03b35\u03b2'2a"]
        self.m_dan2mbon = np.array([
            [1, 1],
            [1, 1],
            [1, 1]
        ], dtype=float)
        self.w_dan2mbon = np.zeros_like(self.m_dan2mbon)
        self.m_mbon2dan = np.array([
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 1]
        ], dtype=float)
        self.w_mbon2dan = np.zeros_like(self.m_mbon2dan)
        self.m_mbon2mbon = np.array([
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=float)
        self.w_mbon2mbon = np.zeros_like(self.m_mbon2mbon)
        self.m_kc2mbon = np.ones((2, 3), dtype=float)
        self.w_kc2mbon = np.zeros_like(self.m_kc2mbon)
        self.m_odour2kc = np.eye(2, 2, dtype=float)
        self.w_odour2kc = np.copy(self.m_odour2kc)
        self.m_odour2dan = np.ones((2, 3), dtype=float)
        self.w_odour2dan = np.zeros_like(self.m_odour2dan)
        self.m_shock2kc = np.zeros((2, 2), dtype=float)
        self.w_shock2kc = np.zeros_like(self.m_shock2kc)
        self.m_shock2dan = np.array((2, 3), dtype=float)
        self.w_shock2dan = np.zeros_like(self.m_shock2dan)
        self.m_shock2mbon = np.array((2, 3), dtype=float)
        self.w_shock2mbon = np.zeros_like(self.m_shock2mbon)
        self.b_mbon = np.ones(3, dtype=float)
        self.b_dan = np.ones(3, dtype=float)
        self.b_kc = np.ones(2, dtype=float)

        # self.f_kc = lambda x: x
        self.f_kc = lambda x: np.maximum(x, 0)
        # self.f_dan = lambda x: x
        self.f_dan = lambda x: np.maximum(x, 0)
        self.f_mbon = lambda x: x

        self.kc = np.zeros(2, dtype=float)
        self.dan = np.ones(3, dtype=float)
        self.mbon = np.zeros(3, dtype=float)

        self.eta = np.ones((2, 3), dtype=float)
        self.eta[0] *= 0.90
        self.eta[1] *= 0.05

    def __call__(self, *args, **kwargs):
        self.kc, self.dan, self.mbon = self._fprop(*args, **kwargs)

    def _fprop(self, shock=np.zeros(2), odour=np.zeros(2)):
        # eta = shock[np.newaxis].dot([[0.90, 0.9, 0.5], [0.90, 0.9, 0.50]])[0]  # routine1
        eta = shock[np.newaxis].dot([[1., 1., 1.], [1., 1., 1.]])[0]  # routine2
        # print eta
        o = odour.dot(self.w_odour2kc)
        s_M = shock[np.newaxis].T.dot(odour[np.newaxis])  # [[SM, SP], [NM, NP]]
        s = np.sum(s_M * self.w_shock2kc, axis=0)
        b = self.b_kc
        kc = self.f_kc(o + s + b)
        # kc = (1 - eta) * self.kc + eta * kc
        print "KC   ", kc, ": O", o, "+ S", s, "+ B", b

        s = shock.dot(s_M).dot(self.w_shock2mbon)
        k = kc.dot(self.w_kc2mbon)
        # d = np.maximum(self.dan.dot(self.w_dan2mbon), 0)
        b = self.b_mbon
        mbon = self.f_mbon(s + k + b)
        m = mbon.dot(self.w_mbon2mbon)
        mbon = self.f_mbon(mbon + m)
        # mbon = (1 - eta) * self.mbon + eta * mbon
        print "MBON ", mbon, ": S", s, "+ K", k, "+ M", m, "+ B", b

        o = odour.dot(self.w_odour2dan)
        s = (1 - eta) * self.dan + eta * np.array([1, 0]).dot(s_M).dot(self.w_shock2dan)
        m = mbon.dot(self.w_mbon2dan)
        b = self.b_dan
        dan = self.f_dan(o + s + m + b)
        # dan = (1 - eta) * self.dan + eta * dan
        print "DAN  ", dan, ": O", o, "+ S", s, "+ M", m, "+ B", b

        return kc, dan, mbon

    def update(self, w_name, data):
        cond = []
        for group in [[self.dan_types[0], self.dan_names[0]], [self.mbon_types[0], self.mbon_names[0]],
                      [self.dan_types[1], self.dan_names[1]], [self.mbon_types[1], self.mbon_names[1]],
                      [self.dan_types[2], self.dan_names[2]], [self.mbon_types[2], self.mbon_names[2]]]:
            c = np.all([data.index.get_level_values(t) == g for t, g in zip(["type", "name"], group)], axis=0)
            cond.append(c)
        data_i = data.iloc[np.any(cond, axis=0)]
        data_avg = data_i.groupby(["type", "name", "id"], axis=0).mean()  # type: pd.DataFrame

        r_mbon = np.array([data_avg.T[self.mbon_types[0], self.mbon_names[0]],
                           data_avg.T[self.mbon_types[1], self.mbon_names[1]],
                           data_avg.T[self.mbon_types[2], self.mbon_names[2]]])
        r_dan = np.array([data_avg.T[self.dan_types[0], self.dan_names[0]],
                          data_avg.T[self.dan_types[1], self.dan_names[1]],
                          data_avg.T[self.dan_types[2], self.dan_names[2]]])

        def rm_all(r, trial=1, cs='-'):
            r_trial = []
            for rr in r:
                rr = rr.to_numpy()
                rrr = rr[((trial-1)*200 + (0 if '-' in cs else 100)):(
                          (trial-1)*200 + 100 + (0 if '-' in cs else 100))]
                r_trial.append(rrr.mean(axis=0))
            return np.array(list(product(*r_trial)))

        def rm_odour(r, trial=1, cs='-'):
            r_trial = []
            for rr in r:
                rr = rr.to_numpy()
                rrr = rr[((trial-1)*200 + (0 if '-' in cs else 100)):(
                          (trial-1)*200 + 100 + (0 if '-' in cs else 100))]
                r_odour = (rrr[:44].sum(axis=0) + rrr[49:].sum(axis=0)) / 95.
                r_trial.append(r_odour)
            return np.array(list(product(*r_trial)))

        def rm_shock(r, trial=1, cs='-'):
            r_trial = []
            for rr in r:
                rr = rr.to_numpy()
                rrr = rr[((trial-1)*200 + (0 if '-' in cs else 100)):(
                          (trial-1)*200 + 100 + (0 if '-' in cs else 100))]
                r_odour = (rrr[:44].sum(axis=0) + rrr[49:].sum(axis=0)) / 95.
                r_shock = rrr[44:49].mean(axis=0) - r_odour
                r_trial.append(r_shock - r_odour)
            return np.array(list(product(*r_trial)))

        # odour : 25-50
        # shock : 45

        start_t = 1
        if w_name == "odour2kc":
            raise TypeError("Changes in these connections are not allowed.")
        elif w_name == "odour2dan":
            csm1 = rm_odour(r_dan, start_t, cs='-')
            csp1 = rm_odour(r_dan, start_t, cs='+')
            self.w_odour2dan[0] = np.linalg.pinv(csm1).dot(np.ones((csm1.shape[0], 1))).flatten() * self.m_odour2dan[0]
            self.w_odour2dan[1] = np.linalg.pinv(csp1).dot(np.ones((csp1.shape[0], 1))).flatten() * self.m_odour2dan[1]
            print "odour2dan :", self.w_odour2dan
        elif w_name == "odour2mbon":
            csm1 = rm_odour(r_mbon, start_t, cs='-')
            csp1 = rm_odour(r_mbon, start_t, cs='+')
            self.w_odour2mbon[0] = np.linalg.pinv(csm1).dot(np.ones((csm1.shape[0], 1))).flatten() * self.m_odour2mbon[0]
            self.w_odour2mbon[1] = np.linalg.pinv(csp1).dot(np.ones((csp1.shape[0], 1))).flatten() * self.m_odour2mbon[1]
            print "odour2mbon :", self.w_odour2mbon
        elif w_name == "shock2kc":
            raise TypeError("Changes in these connections are not allowed.")
        elif w_name == "shock2dan":
            csm1shock = rm_shock(r_dan, start_t, cs='-')
            csp1shock = rm_shock(r_dan, start_t, cs='+')
            csm2shock = rm_shock(r_dan, start_t + 1, cs='-')
            csp2shock = rm_shock(r_dan, start_t + 1, cs='+')
            shock = csp2shock - csp1shock
            noshock = csm2shock - csm1shock
            x = np.concatenate([shock, noshock], axis=0).T
            y = np.concatenate([np.ones((shock.shape[0], 1)), np.zeros((noshock.shape[0], 1))], axis=0)
            self.w_shock2dan = np.linalg.pinv(x.T).dot(y).flatten() * self.m_shock2dan
            print "shock2dan :", self.w_shock2dan
        elif w_name == "shock2mbon":
            csm1shock = rm_shock(r_mbon, start_t, cs='-')
            csp1shock = rm_shock(r_mbon, start_t, cs='+')
            csm2shock = rm_shock(r_mbon, start_t + 1, cs='-')
            csp2shock = rm_shock(r_mbon, start_t + 1, cs='+')
            shock = csp2shock - csp1shock
            noshock = csm2shock - csm1shock
            x = np.concatenate([shock, noshock], axis=0).T
            y = np.concatenate([np.ones((shock.shape[0], 1)), np.zeros((noshock.shape[0], 1))], axis=0)
            self.w_shock2mbon = np.linalg.pinv(x.T).dot(y).T * self.m_shock2mbon
            print "shock2mbon :", self.w_shock2mbon
        elif w_name == "kc2kc":
            raise TypeError("Not supported connections.")
        elif w_name == "kc2dan":
            raise TypeError("Not supported connections.")
        elif w_name == "kc2mbon":
            raise TypeError("Not supported connections.")
        elif w_name == "dan2kc":
            raise TypeError("Not supported connections.")
        elif w_name == "dan2dan":
            raise TypeError("Not supported connections.")
        elif w_name == "dan2mbon":
            csm1d = rm_odour(r_dan, start_t, cs='-')
            csp1d = rm_odour(r_dan, start_t, cs='+')
            csm1m = rm_all(r_mbon, start_t, cs='-')
            csp1m = rm_all(r_mbon, start_t, cs='+')
            temp = np.array(list(product(csm1d, csm1m)))
            csm1d = temp[:, 0]
            csm1m = temp[:, 1]
            temp = np.array(list(product(csp1d, csp1m)))
            csp1d = temp[:, 0]
            csp1m = temp[:, 1]
            x = np.concatenate([csp1d, csm1d], axis=0)
            y = np.concatenate([csp1m, csm1m], axis=0)
            self.w_dan2mbon = np.linalg.pinv(x).dot(y) * self.m_dan2mbon
            print "dan2mbon :", self.w_dan2mbon
        elif w_name == "mbon2kc":
            raise TypeError("Not supported connections.")
        elif w_name == "mbon2dan":
            csm1d = rm_odour(r_dan, start_t, cs='-')
            csp1d = rm_odour(r_dan, start_t, cs='+')
            csm1m = rm_all(r_mbon, start_t, cs='-')
            csp1m = rm_all(r_mbon, start_t, cs='+')
            temp = np.array(list(product(csm1d, csm1m)))
            csm1d = temp[:, 0]
            csm1m = temp[:, 1]
            temp = np.array(list(product(csp1d, csp1m)))
            csp1d = temp[:, 0]
            csp1m = temp[:, 1]
            x = np.concatenate([csp1m, csm1m], axis=0)
            y = np.concatenate([csp1d, csm1d], axis=0)
            self.w_mbon2dan = np.linalg.pinv(x).dot(y) * self.m_mbon2dan
            print "mbon2dan :", self.w_mbon2dan
        elif w_name == "mbon2mbon":
            csm1 = rm_all(r_mbon, start_t, cs='-')
            csp1 = rm_all(r_mbon, start_t, cs='+')
            csm2 = rm_all(r_mbon, start_t + 1, cs='-')
            csp2 = rm_all(r_mbon, start_t + 1, cs='+')
            csp = csp2 - csp1
            csm = csm2 - csm1
            mbonm = np.dot(np.zeros((csm.shape[0], 1)), self.w_shock2mbon) + np.dot(
                np.array([[1, 0]] * csm.shape[0]), self.w_odour2mbon)
            mbonp = np.dot(np.ones((csp.shape[0], 1)), self.w_shock2mbon) + np.dot(
                np.array([[0, 1]] * csm.shape[0]), self.w_odour2mbon)
            x = np.concatenate([mbonp, mbonm], axis=0).T
            y = np.concatenate([csp, csm], axis=0)
            self.w_mbon2mbon = np.linalg.pinv(x.T).dot(y) * self.m_mbon2mbon
            print "mbon2mbon :", self.w_mbon2mbon
        else:
            raise NameError("Name not recognised.")


def visualise(df):
    from src.visualisation import *
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

    # A tale of the three MBONs
    # cond = []
    # for group in [[r'PPL1', u'\u03b31pedc'], [r'MBON-GABA', u'\u03b31pedc'],
    #               [r'PAM', u"\u03b2'2m"], [r'MBON-Glu', u"\u03b2'2mp"],
    #               [r'PAM', u"\u03b2'2a"], [r'MBON-Glu', u"\u03b35\u03b2'2a"]]:
    #     c = np.all([dff.index.get_level_values(t) == g for t, g in zip(["type", "name"], group)], axis=0)
    #     cond.append(c)
    # dff = dff.iloc[np.any(cond, axis=0)]
    # print dff
    plot_3_mbon_traces(df)
    # plot_3_mbon_shock(df)

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
    # for i in range(17):
    #     corr_matrix(dff.sort_index(axis=0, level=['type', 'name']),
    #                 mode=["iter-%d" % (i + 1)], show=True,
    #                 shock=False, diff=False, avg=True, figsize=(3.8, 3))
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = DataFrame()
    # dff = df.unstacked

    print df.dataset6neuron.T
    # df.plot6neuron(version=2)

    # print pd.merge(dfff.reset_index(level="id"), dfff.reset_index(level="id"), on="id")
    # dfff.reset_index(level=["type", "name"], inplace=True)
    # dfff.reset_index(level="id", inplace=True)


    # tmnet = ThreeMBONNet()
    # # tmnet.update("odour2dan", dff)
    # # # tmnet.w_odour2dan *= 0.
    # # tmnet.update("odour2mbon", dff)
    # # tmnet.update("shock2dan", dff)
    # # tmnet.update("shock2mbon", dff)
    # # # tmnet.w_shock2mbon *= 0.
    # # tmnet.update("mbon2mbon", dff)
    # # tmnet.update("mbon2dan", dff)
    # # tmnet.update("dan2mbon", dff)
    #
    # shock, noshock, csm, csp = np.array([1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, 1])
    # dans, mbons = [], []
    #
    # mbon_i = np.array([[6.2, 3.6, 0.1], [5.9, 0.9, -0.1]])
    # dan_i = np.array([[2.2, 5.1, 2.2], [0.5, 2.9, 9.5]])
    #
    # def routine1():
    #     tmnet.b_kc = np.array([0., 0., 0.])
    #     tmnet.b_mbon = np.array([0., 0., 0.])
    #     tmnet.b_dan = np.array([0., 0., 0.])
    #     tmnet.w_odour2kc = np.array([
    #         [6., 1., -1.],
    #         [6., 1., -1.]
    #     ])
    #     tmnet.w_shock2kc = np.array([
    #         [0., 0., 0.],
    #         [0., 0., 0.]
    #     ])
    #     tmnet.w_odour2dan = np.array([
    #         [1., 5., 2.],
    #         [1., 3., 4.]
    #     ])
    #     tmnet.w_shock2dan = np.array([
    #         [3.0, 2.0, 3.0],  # shock
    #         [-.7, -.5, -.7]  # no shock
    #     ])
    #     tmnet.w_shock2mbon = np.array([
    #         [0., 0., 0.],
    #         [0., 0., 0.]
    #     ])
    #     tmnet.w_kc2mbon = np.array([
    #         [1., 0., 0.],
    #         [0., 2., 0.],
    #         [0., 0., 1.]
    #     ])
    #     tmnet.w_dan2mbon = np.array([
    #         [1., 0., 0.],
    #         [0., .5, 0.],
    #         [0., 0., .5]
    #     ])
    #     tmnet.w_mbon2mbon = np.array([
    #         [0., -.2, -.2],
    #         [0., 0., 0.],
    #         [0., 0., 0.]
    #     ])
    #     tmnet.w_mbon2dan = np.array([
    #         [-.1, 0., 0.],
    #         [0., -1.5, 1.],
    #         [0., .5, -.9]
    #     ])
    #     mbonm = mbon_i[0] * 0.
    #     tmnet.mbon = mbonm
    #     # tmnet.dan = np.array([1.9, 5., 2.2])
    #     for t in xrange(17):
    #         odour = csm if t % 2 == 0 else csp
    #         sk = shock if t in [3, 5, 7, 9, 11, 14, 16] else noshock
    #         # if t % 2 == 0:
    #         tmnet(shock=sk, odour=odour)
    #         kc = tmnet.kc
    #         dan = tmnet.dan
    #         mbon = tmnet.mbon
    #         print "KC   CS%s | Trial %2d :" % ('-' if t % 2 == 0 else '+', t // 2 + 1), kc
    #         print "DAN  CS%s | Trial %2d :" % ('-' if t % 2 == 0 else '+', t // 2 + 1), dan
    #         print "MBON CS%s | Trial %2d :" % ('-' if t % 2 == 0 else '+', t // 2 + 1), mbon
    #         dans.append(dan)
    #         mbons.append(mbon)
    #         tmnet.w_dan2mbon += np.array([
    #             [-.1, 0., 0.],
    #             [0., -.2, 0.],
    #             [0., 0., -.1]
    #         ]) * np.sign(np.array(
    #             (kc.dot(tmnet.w_kc2mbon)[np.newaxis] + tmnet.b_kc).T.dot(tmnet.dan[np.newaxis] + tmnet.b_dan),
    #             dtype=float))
    #         tmnet.w_mbon2dan += np.array([
    #             [0., 0., 0.],
    #             [0., 0., 0.],
    #             [0., 0., 0.]
    #         ]) * np.sign(np.array(
    #             (kc.dot(tmnet.w_kc2mbon)[np.newaxis] + tmnet.b_kc).T.dot(tmnet.dan[np.newaxis] + tmnet.b_dan),
    #             dtype=float))
    #         tmnet.b_dan += np.array([0., 0., 0.])
    #         sk = (sk.dot([[1., 1.], [-1., -1.]])[np.newaxis] * odour[np.newaxis])
    #         print t, sk
    #         tmnet.w_odour2kc += np.minimum(np.array([
    #             [-.5, 0., -.5],  # CS- with shock
    #             [-.5, 0., -.5]  # CS+ with shock
    #         ]) * np.array(
    #             sk.T.dot(tmnet.dan[np.newaxis] + 1.) * (tmnet.kc + 1.),
    #             dtype=float), 1.7) - np.array([
    #                 [1., 0., 2.],
    #                 [1., 0., 2.]
    #             ])
    #         tmnet.w_shock2dan += np.array([
    #             [-.5, -.5, -.5],  # shock
    #             [-.5, -.5, -.5]  # no shock
    #         ]) * np.sign(np.array(np.maximum(sk, 0).T.dot(tmnet.kc[np.newaxis] + tmnet.b_kc), dtype=float))
    #
    # def routine2():
    #     tmnet.b_kc = np.array([0., 0.])
    #     tmnet.b_mbon = np.array([4., 0., 2.])
    #     tmnet.b_dan = np.array([0., 1., 7.])
    #     tmnet.w_shock2kc = np.array([
    #         [1., 1.],
    #         [0., 0.]
    #     ])
    #     tmnet.w_shock2dan = np.array([
    #         [3., 1., 2.],  # shock
    #         [3., 1., 2.]  # no shock
    #     ])
    #     tmnet.w_kc2mbon = np.array([
    #         [2.3, 6.8, 1.2],
    #         [1.6, 3.5, 0.6]
    #     ])
    #     tmnet.w_mbon2mbon = np.array([
    #         [0., -.5, -.5],
    #         [0., 0., 0.],
    #         [0., 0., 0.]
    #     ])
    #     tmnet.w_mbon2dan = np.array([
    #         [.3, 0., 0.],
    #         [0., 1., 0.],
    #         [0., 0., 0.]
    #     ])
    #     tmnet.w_dan2mbon = np.array([
    #         [-.5, -.5],
    #         [-.1, -.5],
    #         [.1, .1]
    #     ])
    #     mbonm = mbon_i[0] * 0.
    #     tmnet.mbon = mbonm
    #     for t in xrange(17):
    #         odour = csm if t % 2 == 0 else csp
    #         s = shock if t in [3, 5, 7, 9, 11, 14, 16] else noshock
    #         # if t % 2 == 0:
    #         tmnet(shock=s, odour=odour)
    #         kc = tmnet.kc
    #         dan = tmnet.dan
    #         mbon = tmnet.mbon
    #         sk = (s.dot([[1., 1.], [-1., -1.]])[np.newaxis] * odour[np.newaxis])
    #         sM = s[np.newaxis].T.dot(odour[np.newaxis])
    #         print t, sk
    #         print "KC   CS%s | Trial %2d :" % ('-' if t % 2 == 0 else '+', t // 2 + 1), kc
    #         print "DAN  CS%s | Trial %2d :" % ('-' if t % 2 == 0 else '+', t // 2 + 1), dan
    #         print "MBON CS%s | Trial %2d :" % ('-' if t % 2 == 0 else '+', t // 2 + 1), mbon
    #         # print dan[np.newaxis].T.dot(kc[np.newaxis])
    #         dans.append(dan)
    #         mbons.append(mbon)
    #         tmnet.w_kc2mbon += np.array([
    #             [.07, 1., -.07],  # CS- with shock
    #             [.07, .07, -.07]  # CS+ with shock
    #         ]) * (dan * mbon)[np.newaxis].T.dot(kc[np.newaxis]).T * tmnet.w_dan2mbon.T - .1
    #         # tmnet.w_dan2mbon += np.array([
    #         #     [.01, .01],
    #         #     [0., 0.],
    #         #     [0., 0.]
    #         # ]) * (dan * mbon)[np.newaxis].T.dot(kc[np.newaxis]) * tmnet.w_dan2mbon
    #         # tmnet.w_mbon2dan += np.array([
    #         #     [-.034, 0., 0.],
    #         #     [0., 0., 0.],
    #         #     [0., 0., 0.]
    #         # ]) * mbon[np.newaxis].T.dot(dan[np.newaxis])
    #         tmnet.w_shock2kc -= .1 * (sM - .0)
    #         tmnet.w_shock2dan -= .3 * np.sign(shock[np.newaxis].dot(sM).T.dot(dan[np.newaxis]))
    #         print "Odour", odour
    #         print "Shock", shock.dot(sM)
    #         print "SK -> KC", tmnet.w_shock2dan
    #         print "S       ", s[np.newaxis].T.dot(odour[np.newaxis])
    #
    # routine2()
    #
    # dans = np.array(dans)
    # mbons = np.array(mbons)
    # plt.figure("reconstruct", figsize=(10, 5))
    # plt.subplot(231)
    # plt.plot(mbons[::2, 0], 'bo-')
    # plt.title(u"MBON-\u03b31pedc")
    # plt.subplot(232)
    # plt.plot(mbons[::2, 1], 'bo-')
    # plt.title(u"MBON-\u03b2'2mp")
    # plt.subplot(233)
    # plt.plot(mbons[::2, 2], 'bo-')
    # plt.title(u"MBON-\u03b35\u03b2'2a")
    # plt.subplot(234)
    # plt.plot(dans[::2, 0], 'bo-')
    # plt.title(u"PPL1-\u03b31pedc")
    # plt.subplot(235)
    # plt.plot(dans[::2, 1], 'bo-')
    # plt.title(u"PAM-\u03b2'2m")
    # plt.subplot(236)
    # plt.plot(dans[::2, 2], 'bo-')
    # plt.title(u"PAM-\u03b2'2a")
    #
    # plt.tight_layout()
    #
    # dans = np.array(dans)
    # mbons = np.array(mbons)
    # plt.subplot(231)
    # plt.plot(np.arange(.5, 8, 1), mbons[1::2, 0], 'ro-')
    # plt.title(u"MBON-\u03b31pedc")
    # plt.ylim([-1, 15])
    # plt.xticks(np.arange(0, 9, .5), [""] * 17)
    # plt.xlim([-.2, 8.2])
    # plt.subplot(232)
    # plt.plot(np.arange(.5, 8, 1), mbons[1::2, 1], 'ro-')
    # plt.title(u"MBON-\u03b2'2mp")
    # plt.ylim([-1, 15])
    # plt.xticks(np.arange(0, 9, .5), [""] * 17)
    # plt.xlim([-.2, 8.2])
    # plt.subplot(233)
    # plt.plot(np.arange(.5, 8, 1), mbons[1::2, 2], 'ro-')
    # plt.title(u"MBON-\u03b35\u03b2'2a")
    # plt.ylim([-1, 15])
    # plt.xticks(np.arange(0, 9, .5), [""] * 17)
    # plt.xlim([-.2, 8.2])
    # plt.subplot(234)
    # plt.plot(np.arange(.5, 8, 1), dans[1::2, 0], 'ro-')
    # plt.title(u"PPL1-\u03b31pedc")
    # plt.ylim([-1, 25])
    # plt.xticks(np.arange(0, 9, .5),
    #            ["%d%s" % ((i // 2) + 1, "-" if i % 2 == 0 else "+") for i in np.arange(17)])
    # plt.xlim([-.2, 8.2])
    # plt.subplot(235)
    # plt.plot(np.arange(.5, 8, 1), dans[1::2, 1], 'ro-')
    # plt.title(u"PAM-\u03b2'2m")
    # plt.ylim([-1, 25])
    # plt.xticks(np.arange(0, 9, .5),
    #            ["%d%s" % ((i // 2) + 1, "-" if i % 2 == 0 else "+") for i in np.arange(17)])
    # plt.xlim([-.2, 8.2])
    # plt.subplot(236)
    # plt.plot(np.arange(.5, 8, 1), dans[1::2, 2], 'ro-')
    # plt.title(u"PAM-\u03b2'2a")
    # plt.ylim([-1, 25])
    # plt.xticks(np.arange(0, 9, .5),
    #            ["%d%s" % ((i // 2) + 1, "-" if i % 2 == 0 else "+") for i in np.arange(17)])
    # plt.xlim([-.2, 8.2])
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # danm = np.dot(noshock, tmnet.w_shock2dan) + np.dot(csm, tmnet.w_odour2dan)
    # # danp = np.dot(noshock, tmnet.w_shock2dan) + np.dot(csp, tmnet.w_odour2dan)
    # # mbonm = np.dot(noshock, tmnet.w_shock2mbon) + np.dot(csm, tmnet.w_odour2mbon)
    # # mbonp = np.dot(noshock, tmnet.w_shock2mbon) + np.dot(csp, tmnet.w_odour2mbon)

