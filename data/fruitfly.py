import os
import re
import csv
import numpy as np
import yaml
import string


__dir__ = os.path.dirname(os.path.abspath(__file__))
__data_dir__ = os.path.join(__dir__, "FruitflyMB")

with open(os.path.join(__data_dir__, 'meta.yaml'), 'rb') as f:
    meta = yaml.load(f, Loader=yaml.BaseLoader)


class FruitflyData(dict):

    comps = {
        u"\u03b1": [['p', 'c', 's']] * 3,
        u"\u03b2": [['p', 'c', 's']] * 2,
        u"\u03b1'": [['m', 'p', 'a']] * 3,
        u"\u03b2'": [['m', 'p', 'a']] * 2,
        u"\u03b3": [['m', 'd']] * 5
    }

    def __init__(self, **kwargs):
        super(FruitflyData, self).__init__(**kwargs)

        recs = load_data()
        names = recs.keys()
        self._odours = []
        self._compartments = []
        for name in names:
            genotype, odour = re.findall(r'([\d\w\W]+)-([\w\W]+)', name)[0]
            if genotype not in meta.keys():
                continue
            if genotype not in self.keys():
                self[genotype] = meta[genotype]
            self[genotype][odour] = recs[name]
            if odour not in self._odours:
                self._odours.append(odour)
            self[genotype]['loc'] = self.__name2location(meta[genotype]['name'])

            # print meta[genotype]['type'], meta[genotype]['name'], self[genotype]['loc']

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

        return gens_ret

    @staticmethod
    def dataset(data):

        csp, csm = [], []
        genotypes, names, types = [], [], []
        for name in data.keys():
            keys = data[name].keys()
            for key in keys:
                if "O" in key or "A" in key or "B" in key or "M+S" in key:
                    res = data[name][key]
                    if "M+S" in key:
                        csp.append(res)
                    else:
                        csm.append(res)
            genotypes.append([name] * res.shape[-1])
            names.append([data[name]['name']] * res.shape[-1])
            types.append([data[name]['type']] * res.shape[-1])

        return {
            'genotype': np.concatenate(genotypes),
            'name': np.concatenate(names),
            'type': np.concatenate(types),
            'CS+': np.concatenate(csp, axis=2),
            'CS-': np.concatenate(csm, axis=2)
        }


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    dataset = FruitflyData()
    # gens = dataset.slice(types=["DAN", "MBON"], loc=u"\u03b1'")
    gens = dataset.slice()

    data_dict = dataset.dataset(gens)

    types = data_dict['type']
    traces = np.concatenate([data_dict['CS-'], data_dict['CS+']], axis=1)

    x = traces.reshape((-1, traces.shape[-1]))
    print x.shape

    i = np.argsort(types)
    types = types[i]
    x = x[:, i]

    pca = PCA(x.shape[1], whiten=False)
    pca.fit(x)
    x_proj = pca.transform(x)

    plt.figure("Covariance matrix", figsize=(10, 10))
    plt.imshow(pca.get_covariance(), vmin=-3, vmax=3, cmap="coolwarm")
    plt.yticks(np.arange(x.shape[1])[::7], types[::7])
    # plt.colorbar()

    # plt.figure("pca-types", figsize=(9, 9))
    # types_u = np.unique(types)
    # colours = {
    #     "KC": "black",
    #     "MBON-ND": "grey",
    #     "MBON-glu": "green",
    #     "MBON-gaba": "blue",
    #     "MBON-ach": "red",
    #     "PAM": "cyan",
    #     "PPL1": "magenta"
    # }
    # for t in types_u:
    #     x0 = x_proj.T[types == t]
    #     print x0.shape
    #     plt.scatter(x0[:, 0], x0[:, 1], color=colours[t], edgecolor='black', marker=".", label=t)
    # plt.xlim([-5, 5])
    # plt.ylim([-5, 5])
    # plt.legend()

    # plt.figure("pca-location", figsize=(9, 9))
    # colours = {
    #     u"\u03b1": "red",
    #     u"\u03b2": "green",
    #     u"\u03b1'": "pink",
    #     u"\u03b2'": "greenyellow",
    #     u"\u03b3": "blue"
    # }
    # for loc in colours.keys():
    #     gens = dataset.slice(loc=loc)
    #
    #     data_dict = dataset.dataset(gens)
    #     traces = np.concatenate([data_dict['CS-'], data_dict['CS+']], axis=1)
    #
    #     x = traces.reshape((-1, traces.shape[-1]))
    #     x_proj = pca.transform(x)
    #     plt.scatter(x_proj[:, 0], x_proj[:, 1], color=colours[loc], marker=".", label=loc)
    # plt.xlim([-5, 5])
    # plt.ylim([-5, 5])
    # plt.legend()

    plt.show()
