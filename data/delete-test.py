import os
import yaml

# get path of the script
cpath = os.path.dirname(os.path.abspath(__file__)) + '/'
enpath = cpath + "EN/"
rtpath = cpath + "routes/"
ripath = cpath + "routes-img/"
logpath = cpath + "tests.yaml"

# load tests
with open(logpath, 'rb') as f:
    tests = yaml.safe_load(f)


def get_name(sky_type, j):
    date = tests[sky_type][j]["date"]
    time = tests[sky_type][j]["time"]
    step = tests[sky_type][j]["step"]  # cm

    return "%s_%s_s%02d-%s-sky" % (date, time, step, sky_type)


sky_type = "fixed-no-pol"
id = -1

name = get_name(sky_type, id)

enname = enpath + name + ".npz"
lroute = rtpath + "learned-1-1-" + name + ".npz"
hroute = rtpath + "homing-1-2-" + name + ".npz"
imname = ripath + name + ".png"

files = [enname, lroute, hroute, imname]

print "Are you sure you want to delete '%s'? ([Y]/n)" % name
s = raw_input()

if s in ["Y", "y", ""]:
    for f in files:
        try:
            os.remove(f)
            print "'%s' successfully deleted." % f
        except OSError as e:
            print e.message

    tests[sky_type].remove(tests[sky_type][id])
    with open(logpath, 'wb') as f:
        yaml.safe_dump(tests, f, default_flow_style=False, allow_unicode=False)
        print "Testes log updated successfully."

else:
    print "Canceled."
