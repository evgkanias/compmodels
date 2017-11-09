import numpy as np
import matplotlib.pyplot as plt

from world import load_world, load_route
from utils import *


# sky_type = "fixed"
fov = True

if 'sky_type' in locals():
    plt.figure(sky_type, figsize=(30, 20))

    w = load_world()
    labels = []
    name = get_agent_name(sky_type, 0, fov)
    r = load_route("learned")
    w.add_route(r)
    for j in xrange(len(tests[sky_type])):
        name = get_agent_name(sky_type, j, fov)
        labels.append(name.split("_")[0] + " " + name.split("_")[1])
        r = load_route(name)
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
        print i, sky_type,
        w = load_world()
        try:
            name = get_agent_name(sky_type, 0, fov)
            print ""
        except AttributeError, e:
            print "aboard"
            continue
        r = load_route("learned")
        r.agent_no = 1
        r.route_no = 2
        w.add_route(r)
        labels = []
        test_ = fov_tests if fov else tests
        for j in xrange(len(test_[sky_type])):
            name = get_agent_name(sky_type, j, fov)
            labels.append(name.split("_")[0] + " " + name.split("_")[1])
            r = load_route(name)
            r.agent_no = j+2
            r.route_no = 2
            w.add_route(r)
        img, _ = w.draw_top_view(width=300, length=300)
        plt.subplot(nb_rows, nb_columns, i + 1)
        plt.imshow(img)
        plt.title(sky_type)
        plt.xticks([])
        plt.yticks([])

        for j, label in enumerate(labels):
            x = 300 * (j // 5) / 2 + 15
            y = 300 + 15 * (j - (5 * (j // 5)) + 1)
            plt.text(x, y, label)
    plt.tight_layout(pad=5)
    plt.show()
