import numpy as np
import os
import yaml
from datetime import datetime

# get path of the script
cpath = os.path.dirname(os.path.abspath(__file__)) + '/'
logpath = cpath + "../data/tests.yaml"
datestr = "%Y-%m-%d_%H-%M"

# load tests
with open(logpath, 'rb') as f:
    tests = yaml.safe_load(f)


def create_agent_name(date, sky_type, step=.1, gfov=-np.pi/2, sfov=np.pi/2):
    """

    :param date: the date of the trial
    :type date: datetime
    :param sky_type: the sky-type
    :type sky_type: basestring
    :param step: the step size (default 10 cm)
    :type step: float
    :param gfov: the ground field of view in rads (default -90 degrees - full view)
    :type gfov: float
    :param sfov: the sky field of view in rads (default 90 degrees - full view)
    :type sfov: float
    :return:
    """
    agent_name = "%s_s%02d-%s-sky" % (date.strftime(datestr), step * 100, sky_type)
    if np.abs(gfov) < np.pi / 2:
        agent_name += "_gfov%02d" % np.abs(np.rad2deg(gfov))
    if np.abs(sfov) < np.pi / 2:
        agent_name += "_sfov$02d" % np.abs(np.rad2deg(sfov))

    return agent_name


def get_agent_features(sky_type, j=-1):
    """

    :param sky_type: the sky-type
    :type sky_type: basestring
    :param j: the index of the trial
    :type j: int
    :return:
    """
    if sky_type not in tests.keys():
        raise AttributeError("There is not key named '%s' in the tests records." % sky_type)
    if j >= len(tests[sky_type]):
        raise AttributeError("Index %d out of range. List length = %d" % (j, len(tests[sky_type])))

    date = datetime.strptime("%s_%s" % (tests[sky_type][j]["date"], tests[sky_type][j]["time"]), datestr)
    step = tests[sky_type][j]["step"] / 100.  # cm --> m

    if "gfov" in tests[sky_type][j].keys():
        gfov = -np.deg2rad(tests[sky_type][j]["gfov"])  # degrees --> rad
    else:
        gfov = -np.pi / 2
    if "sfov" in tests[sky_type][j].keys():
        sfov = np.deg2rad(tests[sky_type][j]["sfov"])  # degrees --> rad
    else:
        sfov = np.pi / 2

    return date, step, gfov, sfov


def get_agent_name(sky_type, j):
    """

    :param sky_type: the sky-type
    :type sky_type: basestring
    :param j: the index of the trial
    :type j: int
    :return:
    """
    date, step, gfov, sfov = get_agent_features(sky_type, j)
    date_str = date.strftime(datestr)

    agent_name = "%s_s%02d-%s-sky" % (date_str, step * 100, sky_type)
    if np.abs(gfov) < np.pi / 2:
        gfov = np.abs(np.rad2deg(gfov))  # degrees
        agent_name += "_gfov%02d" % gfov
    if np.abs(sfov) < np.pi / 2:
        sfov = np.abs(np.rad2deg(sfov))  # degrees
        agent_name += "_sfov%02d" % sfov

    return agent_name


def update_tests(sky_type, date, step, gfov=-np.pi/2, sfov=np.pi/2):
    """

    :param sky_type: the sky-type
    :type sky_type:
    :param date: the date of the trial
    :type date:
    :param step: the step size (in meters)
    :type step:
    :param gfov: the ground field of view (in rads)
    :param sfov: the sky field of view (in rads)
    :return:
    """

    if sky_type not in tests.keys():
        tests[sky_type] = []
    date_str = date.strftime("%Y-%m-%d_%H-%M")
    tests[sky_type].append({
        "date": date_str.split("_")[0],
        "time": date_str.split("_")[1],
        "step": int(step * 100)
    })
    if np.abs(gfov) < np.pi / 2:
        tests[sky_type][-1]["gfov"] = np.rad2deg(gfov)
    if np.abs(sfov) < np.pi / 2:
        tests[sky_type][-1]["sfov"] = np.rad2deg(sfov)

    try:
        # save/update tests
        with open(logpath, 'wb') as f:
            yaml.safe_dump(tests, f, default_flow_style=False, allow_unicode=False)
        return True
    except Exception, e:
        print e.message
        return False
