import numpy as np
import numpy.linalg as la
from datetime import datetime, timedelta


def vec2sph(vec):
    """
    Transforms a cartessian vector to spherical coordinates.
    :param vec:     the cartessian vector
    :return theta:  elevation
    :return __phi:    azimuth
    :return rho:    radius
    """
    rho = la.norm(vec, axis=-1)  # length of the radius

    if vec.ndim == 1:
        vec = vec[np.newaxis, ...]
        if rho == 0:
            rho = 1.
    else:
        rho = np.concatenate([rho[..., np.newaxis]] * vec.shape[1], axis=-1)
        rho[rho == 0] = 1.
    v = vec / rho  # normalised vector

    phi = np.arctan2(v[:, 1], v[:, 0])  # azimuth
    theta = np.arccos(v[:, 2])  # elevation

    # theta, __phi = sphadj(theta, __phi)  # bound the spherical coordinates
    return np.asarray([theta, phi, rho[:, -1]])


def shifted_datetime(roll_back_days=153, lower_limit=7.5, upper_limit=19.5):
    date_time = datetime.now() - timedelta(days=roll_back_days)
    if lower_limit is not None and upper_limit is not None:
        uhours = int(upper_limit // 1)
        uminutes = timedelta(minutes=(upper_limit % 1) * 60)
        lhours = int(lower_limit // 1)
        lminutes = timedelta(minutes=(lower_limit % 1) * 60)
        if (date_time - uminutes).hour > uhours or (date_time - lminutes).hour < lhours:
            date_time = date_time + timedelta(hours=12)
    return date_time
