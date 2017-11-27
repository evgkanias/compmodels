import numpy as np
from scipy.special import expit
from base import Network, params

GAIN = -.1 / params['gain']
N_COLUMNS = params['central-complex']['columns']  # 8
x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)

cxparams = params['central-complex']
cxrate_params = params['central-complex-rate']


class CX(Network):

    def __init__(self, tn_prefs=np.pi/4, gain=GAIN, noise=.1, **kwargs):

        super(CX, self).__init__(gain=gain, **kwargs)

        self.tn_prefs = tn_prefs
        self.smoothed_flow = 0.
        self.noise = noise

        self.nb_tb1 = cxparams['TB1']  # 8
        self.nb_tn1 = cxparams['TN1']  # 2
        self.nb_tn2 = cxparams['TN2']  # 2
        self.nb_cpu4 = cxparams['CPU4']  # 16
        nb_cpu1a = cxparams['CPU1A']  # 14
        nb_cpu1b = cxparams['CPU1B']  # 2
        self.nb_cpu1 = nb_cpu1a + nb_cpu1b  # 16

        self.tb1 = np.zeros(self.nb_tb1)
        self.tn1 = np.zeros(self.nb_tn1)
        self.tn2 = np.zeros(self.nb_tn2)
        self.__cpu4 = np.zeros(self.nb_cpu4)
        self.cpu1 = np.zeros(self.nb_cpu1)

        # Weight matrices based on anatomy (These are not changeable!)
        self.w_cl12tb1 = np.tile(np.eye(self.nb_tb1), 2).T
        self.w_tb12tb1 = gen_tb_tb_weights(self.nb_tb1)
        self.w_tb12cpu1a = np.tile(np.eye(self.nb_tb1), (2, 1))[1:nb_cpu1a+1, :]
        self.w_tb12cpu1b = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 0, 0, 0]])
        self.w_tb12cpu4 = np.tile(np.eye(self.nb_tb1), (2, 1))
        self.w_tn2cpu4 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        self.w_cpu42cpu1a = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        self.w_cpu42cpu1b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 9
        ])
        self.w_cpu1a2motor = np.array([
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])
        self.w_cpu1b2motor = np.array([[0, 1],
                                       [1, 0]])

        # The cell properties (for sigmoid function)
        self.tl2_slope = cxrate_params['tl2-tuned']['slope']
        self.tl2_bias = cxrate_params['tl2-tuned']['bias']
        self.tl2_prefs = -np.tile(np.linspace(0, 2 * np.pi, self.nb_tb1, endpoint=False), 2)
        self.cl1_slope = cxrate_params['cl1-tuned']['slope']
        self.cl1_bias = cxrate_params['cl1-tuned']['bias']
        self.tb1_slope = cxrate_params['tb1-tuned']['slope']
        self.tb1_bias = cxrate_params['tb1-tuned']['bias']
        self.cpu4_slope = cxrate_params['cpu4-tuned']['slope']
        self.cpu4_bias = cxrate_params['cpu4-tuned']['bias']
        self.cpu1_slope = cxrate_params['cpu1-tuned']['slope']
        self.cpu1_bias = cxrate_params['cpu1-tuned']['bias']
        self.motor_slope = cxrate_params['motor-tuned']['slope']
        self.motor_bias = cxrate_params['motor-tuned']['bias']

    @property
    def cpu4(self):
        return self.__cpu4

    def __call__(self, *args, **kwargs):
        compass, flow = args[:2]
        self.tb1, self.tn1, self.tn2, _, self.cpu1 = self._fprop(compass, flow)
        return self.f_motor(self.cpu1)

    def f_tl2(self, theta):
        """
        Just a dot product with the preferred angle and current heading.
        :param theta:
        :type theta: float
        :return:
        """
        output = np.cos(theta - self.tl2_prefs)
        return noisy_sigmoid(output, self.tl2_slope, self.tl2_bias, self.noise)

    def f_cl1(self, tl2):
        """
        Takes input from the TL2 neurons and gives output.
        :param tl2:
        :return:
        """
        return noisy_sigmoid(-tl2, self.cl1_slope, self.cl1_bias, self.noise)

    def f_tb1(self, cl1, tb1=None):
        """
        Sinusoidal response to solar compass.

        :param cl1:
        :type cl1: np.ndarray
        :param tb1:
        :type tb1: np.ndarray
        :return:
        """
        if tb1 is None:
            output = cl1
        else:
            p = .667  # Proportion of input from CL1 vs TB1
            cl1_out = cl1.dot(self.w_cl12tb1)
            tb1_out = tb1.dot(self.w_tb12tb1)
            output = p * cl1_out - (1. - p) * tb1_out

        return noisy_sigmoid(output, self.tb1_slope, self.tb1_bias, self.noise)

    def f_tn1(self, flow):
        """
        Linearly inverse sensitive to forwards and backwards motion.

        :param flow:
        :type flow: np.ndarray
        :return:
        """
        noise = self.rng.normal(scale=self.noise, size=flow.shape)
        return np.clip((1. - flow) / 2. + noise, 0, 1)

    def f_tn2(self, flow):
        """
        Linearly sensitive to forwards motion only.

        :param flow:
        :type flow: np.ndarray
        :return:
        """
        return np.clip(flow, 0, 1)

    def f_cpu4(self, tb1, tn1, tn2):
        """
        Output activity based on memory.

        :param tb1:
        :type tb1: np.ndarray
        :param tn1:
        :type tn1: np.ndarray
        :param tn2:
        :type tn2: np.ndarray
        :return:
        """

        # Idealised setup, where we can negate the TB1 sinusoid for memorising backwards motion
        update = np.clip((.5 - tn1).dot(self.w_tn2cpu4), 0, 1)

        update *= self.gain * (1. - tb1).dot(self.w_tb12cpu4.T)

        # Both CPU4 waves must have same average
        # If we don't normalise get drift and weird steering
        update -= self.gain * .25 * tn2.dot(self.w_tn2cpu4)

        # Constant purely to visualise same as rate-based model
        self.__cpu4 = np.clip(self.__cpu4 + update, 0., 1.)

        return noisy_sigmoid(self.__cpu4, self.cpu4_slope, self.cpu4_bias, self.noise)

    def f_cpu1a(self, tb1, cpu4):
        """
        The memory and direction used together to get population code for heading.

        :param tb1:
        :type tb1: np.ndarray
        :param cpu4:
        :type cpu4: np.ndarray
        :return:
        """
        inputs = np.dot(self.w_cpu42cpu1a, cpu4) * np.dot(self.w_tb12cpu1a, 1. - tb1)
        return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias, self.noise)

    def f_cpu1b(self, tb1, cpu4):
        """
        The memory and direction used together to get population code for heading.

        :param tb1:
        :type tb1: np.ndarray
        :param cpu4:
        :type cpu4: np.ndarray
        :return:
        """
        inputs = np.dot(self.w_cpu42cpu1b, cpu4) * np.dot(self.w_tb12cpu1b, 1. - tb1)
        return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias, self.noise)

    def f_cpu1(self, tb1, cpu4):
        """
        Offset CPU4 columns by 1 column (45 degrees) left and right wrt TB1.

        :param tb1:
        :type tb1: np.ndarray
        :param cpu4:
        :type cpu4: np.ndarray
        :return:
        """
        cpu1a = self.f_cpu1a(tb1, cpu4)
        cpu1b = self.f_cpu1b(tb1, cpu4)
        return np.hstack([cpu1b[-1], cpu1a, cpu1b[0]])

    def f_motor(self, cpu1):
        """
        Outputs a scalar where sign determines left or right turn.
        :param cpu1:
        :type cpu1: np.ndarray
        :return:
        """

        cpu1a = cpu1[1:-1]
        cpu1b = np.array([cpu1[-1], cpu1[0]])
        motor = np.dot(self.w_cpu1a2motor, cpu1a)
        motor += np.dot(self.w_cpu1b2motor, cpu1b)
        output = (motor[1] - motor[0]) * .25  # to kill the noise a bit!
        return output

    def _fprop(self, phi, flow):
        if isinstance(phi, np.ndarray) and phi.size == 8:
            cl1 = np.tile(phi, 2)
        else:
            tl2 = self.f_tl2(phi)
            cl1 = self.f_cl1(tl2)
        tb1 = self.f_tb1(cl1, self.tb1)
        tn1 = self.f_tn1(flow)
        tn2 = self.f_tn2(flow)
        cpu4 = self.f_cpu4(tb1, tn1, tn2)
        cpu1 = self.f_cpu1(tb1, cpu4)

        return tb1, tn1, tn2, cpu4, cpu1

    def get_flow(self, heading, velocity, filter_steps=0):
        """
        Calculate optic flow depending on preference angles. [L, R]
        """
        A = np.array([[np.sin(heading - self.tn_prefs),
                       np.cos(heading - self.tn_prefs)],
                      [np.sin(heading + self.tn_prefs),
                       np.cos(heading + self.tn_prefs)]])
        flow = velocity.dot(A)

        # If we are low-pass filtering speed signals (fading memory)
        if filter_steps > 0:
            self.smoothed_flow = (1.0 / filter_steps * flow + (1.0 -
                                  1.0 / filter_steps) * self.smoothed_flow)
            flow = self.smoothed_flow
        return flow


def gen_tb_tb_weights(nb_tb1, weight=1.):
    """
    Weight matrix to map inhibitory connections from TB1 to other neurons
    """

    W = np.zeros([nb_tb1, nb_tb1])
    sinusoid = -(np.cos(np.linspace(0, 2*np.pi, nb_tb1, endpoint=False)) - 1)/2  # type: np.ndarray
    for i in range(nb_tb1):
        values = np.roll(sinusoid, i)
        W[i, :] = values
    return weight * W


def noisy_sigmoid(v, slope=1.0, bias=0.5, noise=0.01):
    """
    Takes a vector v as input, puts through sigmoid and adds Gaussian noise. Results are clipped to return rate
    between 0 and 1.

    :param v:
    :type v: np.ndarray
    :param slope:
    :type slope: float
    :param bias:
    :type bias: float
    :param noise:
    :type noise: float
    """
    sig = expit(v * slope - bias)
    if noise > 0:
        sig += np.random.normal(scale=noise, size=len(v))
    return np.clip(sig, 0, 1)
