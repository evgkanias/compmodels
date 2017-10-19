import numpy as np


class NoneCondition(object):

    def valid(self, d_x, d_phi):
        return True


class Stepper(NoneCondition):

    def __init__(self, step):
        super(Stepper, self).__init__()
        self.__step = np.abs(step)

    def valid(self, d_x, d_phi):
        return d_x > self.__step

    @property
    def _Route__step(self):
        return self.__step

    @_Route__step.setter
    def _Route__step(self, value):
        self.__step = value


class Turner(NoneCondition):

    def __init__(self, tau_phi):
        super(Turner, self).__init__()
        self.__tau_phi = np.abs(tau_phi)

    def valid(self, d_x, d_phi):
        return np.abs(d_phi) > self.__tau_phi

