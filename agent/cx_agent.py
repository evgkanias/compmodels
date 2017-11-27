import numpy as np
from agent import Agent
from world import Hybrid
from net import CXRate, update_cells
from compoundeye import CompassSensor
from datetime import datetime


class CXAgent(Agent):
    FOV = 60

    def __init__(self, init_pos=np.zeros(3), init_rot=np.zeros(2), condition=Hybrid(),
                 live_sky=True, rgb=False, compass=CompassSensor(nb_lenses=60, fov=np.deg2rad(FOV)),
                 visualiser=None, name=None):
        """

        :param init_pos: the initial position
        :type init_pos: np.ndarray
        :param init_rot: the initial orientation
        :type init_rot: np.ndarray
        :param condition:
        :type condition: Hybrid
        :param live_sky: flag to update the sky with respect to the time
        :type live_sky: bool
        :param rgb: flag to set as input to the network all the channels (otherwise use only green)
        :type rgb: bool
        :param compass: the compass sensor
        :type compass: CompassSensor
        :param visualiser:
        :type visualiser: Visualiser
        :param name: a name for the agent
        :type name: string
        """
        super(CXAgent, self).__init__(init_pos=init_pos, init_rot=init_rot, condition=condition,
                                      live_sky=live_sky, rgb=rgb, visualiser=visualiser, name=name)

        self.__net = CXRate()
        self.compass = compass

        if name is None:
            self.name = "cx_agent_%02d" % self.id

    def read_sensor(self):
        self.compass.facing_direction = self.rot[1]
        return self.compass(self.world.sky, decode=False)

    def start_homing(self, reset=True):
        if self.world is None:
            # TODO: warn about not setting the world
            return None

        if reset:
            print "Resetting..."
            self.reset()

        # initialise the visualisation
        if self.visualiser is not None:
            self.visualiser.reset()

        # add a copy of the current route to the world to visualise the path
        xs, ys, zs, phis = [self.pos[0]], [self.pos[1]], [self.pos[2]], [self.rot[1]]
        ens = []
        self.world.routes.append(route_like(
            self.world.routes[0], xs, ys, zs, phis, agent_no=self.id, route_no=len(self.world.routes) + 1)
        )

        d_nest = lambda: np.sqrt(np.square(self.pos[:2] - self.nest).sum())
        d_feeder = 0
        counter = 0
        start_time = datetime.now()
        while d_nest() > 0.1:
            x, y, z = self.pos
            phi = self.rot[1]
            # if d_feeder // .1 > counter:
            en, snaps = [], []

            if self.visualiser is not None and self.visualiser.is_quit():
                break

            sun = self.read_sensor()

            self.__net.
            # make a forward pass from the network
            en.append(self._net(pn))

            if self.visualiser is not None:
                now = datetime.now() - start_time
                min = now.seconds // 60
                sec = now.seconds % 60
                self.visualiser.update_thumb(snap, pn=self._net.pn, pn_mode="L",
                                             caption="Elapsed time: %02d:%02d" % (min, sec))

            en = np.array(en).flatten()
            ens.append(en)
            # show preference to the least turning angle
            en += np.append(np.linspace(.01, 0., 30, endpoint=False), np.linspace(0., .01, 31))
            phi += np.deg2rad(2 * (en.argmin() - 30))

            counter += 1

            self.rot[1] = phi
            self.pos[:] = x + self.dx * np.cos(phi), y + self.dx * np.sin(phi), z
            xs.append(self.pos[0])
            ys.append(self.pos[1])
            zs.append(self.pos[2])
            phis.append(self.rot[1])

            self.world.routes[-1] = route_like(self.world.routes[-1], xs, ys, zs, phis)

            # update view
            img_func = None
            if self.visualiser.mode == "top":
                img_func = self.world.draw_top_view
            elif self.visualiser.mode == "panorama":
                img_func = self.world_snapshot
            if self.visualiser is not None:
                names = self.name.split('_')
                names[0] = self.world.date.strftime(datestr)
                names.append(counter)
                names.append(2 * (en.argmin() - 30))
                names.append(en.min())
                names.append(d_feeder)
                names.append(d_nest())

                capt_format = "%s " * (len(names) - 5) + "| C: % 2d, EN: % 3d (%.2f), D: %.2f, D_nest: %.2f"
                self.visualiser.update_main(img_func, en=en, thumbs=snaps, caption=capt_format % tuple(names))

            if d_feeder > 15:
                break
            d_feeder += self.dx
        self.world.routes.remove(self.world.routes[-1])
        np.savez(__data__ + "EN/%s.npz" % self.name, en=np.array(ens))
        return Route(xs, ys, zs, phis, condition=self.condition, agent_no=self.id, route_no=len(self.world.routes) + 1)

