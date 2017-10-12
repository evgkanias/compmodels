import numpy as np
import yaml
import os

# get path of the script
cpath = os.path.dirname(os.path.abspath(__file__)) + '/'

# load parameters
with open(cpath + 'params.yaml', 'rb') as f:
    params = yaml.safe_load(f)

GAIN = params['gain']


class Willshaw(object):

    def __init__(self, gain=GAIN, learning_rate=.1, tau=1.7, dtype=np.float32):
        self.dtype = dtype
        self.learning_rate = learning_rate
        self.gain = gain
        self._tau = tau

        self.nb_pn = params['neurons']['PN']
        self.nb_kc = params['neurons']['KC']
        self.nb_en = params['neurons']['EN']

        self.w_pn2kc = generate_pn2kc_weights(self.nb_pn, self.nb_kc, dtype=self.dtype)
        self.w_kc2en = np.ones((self.nb_kc, self.nb_en), dtype=self.dtype)
        self.params = [self.w_pn2kc, self.w_kc2en]

        self.f_pn = lambda x: np.maximum(x.astype(dtype) / x.max(), 0)
        self.f_kc = lambda x: np.float32(x > tau)
        self.f_en = lambda x: np.maximum(x, 0)

        self.__update = False

    @property
    def update(self):
        return self.__update

    @update.setter
    def update(self, value):
        self.__update = value

    def __call__(self, *args, **kwargs):
        pn, kc, en = self._fprop(args[0])
        print "PN", pn.max()
        print "KC", kc.sum()
        print "EN", en.sum()
        if self.__update:
            self._update(kc)
        return en

    def _fprop(self, pn):
        a_pn = self.f_pn(pn)
        kc = a_pn.dot(self.w_pn2kc)
        a_kc = self.f_kc(kc)
        en = a_kc.dot(self.w_kc2en)
        a_en = self.f_en(en)
        return a_pn, a_kc, a_en

    def _update(self, kc):
        """
            THE LEARNING RULE:
        ----------------------------

          KC  | KC2EN(t)| KC2EN(t+1)
        ______|_________|___________
           1  |    1    |=>   0
           1  |    0    |=>   0
           0  |    1    |=>   1
           0  |    0    |=>   0

        :param kc: the KC activation
        :return:
        """
        learning_rule = (kc >= self.w_kc2en[:, 0]).astype(bool)
        self.w_kc2en[:, 0][learning_rule] = np.maximum(self.w_kc2en[:, 0][learning_rule] - self.learning_rate, 0)


def generate_pn2kc_weights(nb_pn, nb_kc, min_pn=5, max_pn=21, aff_pn2kc=None, nb_trials=100000, baseline=25000,
                           dtype=np.float32):
    """
    Create the synaptic weights among the Projection Neuros (PNs) and the Kenyon Cells (KCs).
    Choose the first sample that has dispersion below the baseline (early stopping), or the
    one with the lower dispersion (in case non of the samples' dispersion is less than the
    baseline).

    :param nb_pn:       the number of the Projection Neurons (PNs)
    :param nb_kc:       the number of the Kenyon Cells (KCs)
    :param aff_pn2kc:   the number of the PNs connected to every KC (usually 28-34)
                        if the number is less than or equal to zero it creates random values
                        for each KC in range [28, 34]
    :param nb_trials:   the number of trials in order to find a acceptable sample
    :param baseline:    distance between max-min number of projections per PN
    """

    dispersion = np.zeros(nb_trials)
    best_pn2kc = None

    for trial in range(nb_trials):
        pn2kc = np.zeros((nb_pn, nb_kc), dtype=dtype)

        if aff_pn2kc is None or aff_pn2kc <= 0:
            vaff_pn2kc = np.random.randint(min_pn, max_pn + 1, size=nb_pn)
        else:
            vaff_pn2kc = np.ones(nb_pn) * aff_pn2kc

        # go through every kenyon cell and select a nb_pn PNs to make them afferent
        for i in range(nb_pn):
            pn_selector = np.random.permutation(nb_kc)
            pn2kc[i, pn_selector[:vaff_pn2kc[i]]] = 1

        # This selections mechanism can be used to restrict the distribution of random connections
        #  compute the sum of the elements in each row giving the number of KCs each PN projects to.
        pn2kc_sum = pn2kc.sum(axis=0)
        dispersion[trial] = pn2kc_sum.max() - pn2kc_sum.min()
        # pn_mean = pn2kc_sum.mean()

        # Check if the number of projections per PN is balanced (min max less than baseline)
        #  if the dispersion is below the baseline accept the sample
        if dispersion[trial] <= baseline: return pn2kc

        # cache the pn2kc with the least dispersion
        if best_pn2kc is None or dispersion[trial] < dispersion[:trial].min():
            best_pn2kc = pn2kc

    # if non of the samples have dispersion lower than the baseline,
    # return the less dispersed one
    return best_pn2kc

if __name__ == "__main__":
    from world import load_world, load_routes

    world = load_world()
    routes = load_routes()
    world.add_route(routes[0])

    nn = Willshaw()
    nn.update = True

    tau_step = 5
    for xyz, phi in zip(world.routes[-1].xyz[::tau_step], world.routes[-1].phi[::tau_step]):
        img, _ = world.draw_panoramic_view(xyz[0], xyz[1], xyz[2], phi)
        inp = np.array(img).reshape((-1, 3))
        en = nn(inp.max(axis=-1))
