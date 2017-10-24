import numpy as np
import matplotlib.pyplot as plt

from world import load_world, load_route

name = "60-deg"
en = np.load("%s.npz" % name)["en"].T
min_en = np.argmin(en, axis=0)

w = load_world()
r = load_route("learned-1-1-%s" % name)
w.add_route(r)
r = load_route("homing-1-2-%s" % name)
w.add_route(r)
img, _ = w.draw_top_view(width=500, length=500)

plt.figure("ENs activation", figsize=(20, 7))
plt.subplot(121)
plt.imshow(np.log(en), cmap="Greys", vmin=0, vmax=2)
plt.plot(min_en, 'r.-')
plt.yticks(np.linspace(0, 60, 7), np.linspace(-60, 60, 7))
plt.colorbar()
plt.xlabel("time (steps)")
plt.ylabel("Turning (degrees)")
plt.subplot(122)
plt.imshow(np.array(img))
plt.show()
