import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

from world import load_world, load_route

# get path of the script
cpath = os.path.dirname(os.path.abspath(__file__)) + '/'
logpath = cpath + "tests.yaml"

# load tests
with open(logpath, 'rb') as f:
    tests = yaml.safe_load(f)


def get_name(sky_type, j):
    date = tests[sky_type][j]["date"]
    time = tests[sky_type][j]["time"]
    step = tests[sky_type][j]["step"]  # cm

    return "%s_%s_s%02d-%s-sky" % (date, time, step, sky_type)


# sky_type = "fixed"

if 'sky_type' in locals():
    plt.figure(sky_type, figsize=(30, 20))

    w = load_world()
    labels = []
    name = get_name(sky_type, 0)
    r = load_route("learned-1-1-%s" % name)
    w.add_route(r)
    for j in xrange(len(tests[sky_type])):
        name = get_name(sky_type, j)
        labels.append(name.split("_")[0] + " " + name.split("_")[1])
        r = load_route("homing-1-2-%s" % name)
        r.route_no = j+2
        w.add_route(r)

    img, _ = w.draw_top_view(width=800, length=800)

    plt.figure("Sky: %s" % sky_type, figsize=(30, 30))
    plt.imshow(img)
    for i, label in enumerate(labels):
        plt.text(810, 15 * (i + 1), label)
    plt.show()
else:
    nb_columns = 5
    nb_rows = 2

    plt.figure(figsize=(30, 20))
    sky_types = ["uniform", "fixed", "fixed-no-pol", "live", "live-no-pol",
                 "uniform-rgb", "fixed-rgb", "fixed-no-pol-rgb", "live-rgb", "live-no-pol-rgb"]
    for i, sky_type in enumerate(sky_types):
        w = load_world()
        name = get_name(sky_type, 0)
        r = load_route("learned-1-1-%s" % name)
        w.add_route(r)
        labels = []
        for j in xrange(len(tests[sky_type])):
            name = get_name(sky_type, j)
            labels.append(name.split("_")[0] + " " + name.split("_")[1])
            r = load_route("homing-1-2-%s" % name)
            r.route_no = j+2
            w.add_route(r)
        img, _ = w.draw_top_view(width=300, length=300)
        plt.subplot(nb_rows, nb_columns, i + 1)
        plt.imshow(img)
        plt.title(sky_type)
        plt.xticks([])
        plt.yticks([])

        for j, label in enumerate(labels):
            x = 300 * (j // 5) + 15
            y = 300 + 15 * (j - (5 * (j // 5)) + 1)
            plt.text(x, y, label)
    plt.tight_layout(pad=5)
    plt.show()
