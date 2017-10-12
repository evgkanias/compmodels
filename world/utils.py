import numpy as np
import numpy.linalg as la


def vec2sph(vec):
    """
    Transforms a cartessian vector to spherical coordinates.
    :param vec:     the cartessian vector
    :return theta:  elevation
    :return phi:    azimuth
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

    # theta, phi = sphadj(theta, phi)  # bound the spherical coordinates
    return np.asarray([theta, phi, rho[:, -1]])
