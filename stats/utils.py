import numpy as np
from world.geometry import Route


def distance_from_route(route, point):
    """

    :param route: the route
    :type route: Route
    :param point:
    :type point: np.ndarray
    :return:
    """

    xyz = np.array([route.x, route.y, route.z]).T
    if point.ndim == 1:
        point = point.reshape((1, -1))
    if point.shape[1] < 2:
        raise AttributeError()
    if point.shape[1] < 3:
        point = np.append(point, route.z.mean(), axis=1)

    d = np.sqrt(np.square(xyz - point).sum(axis=1))

    return d.min()


if __name__ == "__main__":
    import os
    import yaml

    from world import load_route
    import matplotlib.pyplot as plt

    # get path of the script
    cpath = os.path.dirname(os.path.abspath(__file__)) + '/'
    logpath = cpath + "../data/tests.yaml"

    # load tests
    with open(logpath, 'rb') as f:
        tests = yaml.safe_load(f)


    def get_name(sky_type, j):
        date = tests[sky_type][j]["date"]
        time = tests[sky_type][j]["time"]
        step = tests[sky_type][j]["step"]  # cm

        return "%s_%s_s%02d-%s-sky" % (date, time, step, sky_type)

    nb_columns = 5
    nb_rows = 2

    skies = ["uniform", "fixed", "fixed-no-pol", "live", "live-no-pol",
             "uniform-rgb", "fixed-rgb", "fixed-no-pol-rgb", "live-rgb", "live-no-pol-rgb"]

    plt.figure(figsize=(30, 20))
    for i, sky in enumerate(skies):
        nb_trials = len(tests[sky])
        plt.subplot(nb_rows, nb_columns, i + 1)
        for id in xrange(nb_trials):
            name = get_name(sky, id)
            r = load_route("homing-1-2-%s" % name)
            h_xyz = np.array([r.x, r.y, r.z]).T

            r = load_route("learned-1-1-%s" % name)

            dist = []
            for p in h_xyz:
                dist.append(distance_from_route(r, p))
            dist = np.array(dist)
            plt.plot(dist, label="%s %s" % (tests[sky][id]["date"], tests[sky][id]["time"]))
        plt.title(sky)
        plt.xlim([0, 100])
        plt.ylim([0, 5])
        if i < 5:
            plt.xticks(np.linspace(0, 100, 5), [""] * 5)
        else:
            plt.xticks(np.linspace(0, 100, 5), np.linspace(0, 2, 5))
            plt.xlabel("Time (sec)")
        if i not in [0, 5]:
            plt.yticks(np.linspace(0, 5, 6), [""] * 6)
        else:
            plt.yticks(np.linspace(0, 5, 6))
            plt.ylabel("Distance (m)")
        plt.legend()
        plt.grid()
    plt.tight_layout(pad=5)
    plt.show()