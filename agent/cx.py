import numpy as np
from base import Agent
from world import Hybrid, Route, route_like, __data__
from net import CX
from compoundeye import CompassSensor, decode_sun
from datetime import datetime
from utils import datestr


class CXAgent(Agent):
    FOV = (-np.pi/6, np.pi/3)
    COMPASS_FOV = 60

    def __init__(self, compass=CompassSensor(nb_lenses=60, fov=np.deg2rad(COMPASS_FOV)), *args, **kwargs):
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
        if 'fov' in kwargs.keys() and kwargs['fov'] is None:
            kwargs['fov'] = CXAgent.FOV

        super(CXAgent, self).__init__(*args, **kwargs)

        self._net = CX(noise=.01)
        self.compass = compass

        if 'name' in kwargs.keys() and kwargs['name'] is None:
            self.name = "cx_agent_%02d" % self.id

    def reset(self):
        if super(CXAgent, self).reset():
            # reset to the nest instead of the feeder
            self.pos[:2] = self.nest.copy()
            self.rot[1] = (self.homing_routes[-1].phi[-2] + np.pi) % (2 * np.pi)
            return True
        else:
            return False

    def start_learning_walk(self):

        if self.world is None:
            # TODO: warn about not setting the world
            return None
        elif len(self.homing_routes) == 0:
            # TODO: warn about not setting the homing route
            return None

        print "Resetting..."
        self.reset()

        # initialise visualisation
        if self.visualiser is not None:
            self.visualiser.reset()

        # follow a reverse homing route
        rt = self.homing_routes[-1].reverse()

        # add a copy of the current route to the world to visualise the path
        xs, ys, zs, phis = [self.pos[0]], [self.pos[1]], [self.pos[2]], [self.rot[1]]
        self.world.routes.append(
            route_like(rt, xs, ys, zs, phis, self.condition, agent_no=self.id, route_no=len(self.world.routes) + 1)
        )
        counter = 0         # count the steps
        pphi = self.rot[1]  # initialise the last orientation to the current

        for x, y, z, phi in rt:
            # stop the loop when we close the visualisation window
            if self.visualiser is not None and self.visualiser.is_quit():
                break

            # update the agent position
            self.pos[:] = x, y, z
            self.rot[1] = phi
            # calculate the distance from the start position (feeder)
            distance = np.sqrt(np.square(self.pos[:2] - self.feeder[:2]).sum())

            # update the route in the world
            xs.append(x)
            ys.append(y)
            zs.append(z)
            phis.append(phi)
            self.world.routes[-1] = route_like(self.world.routes[-1], xs, ys, zs, phis)

            d_phi = phi - pphi
            sun = self.read_sensor()
            heading = self.get_heading(sun, phi, pphi)
            flow = self._net.get_flow(heading, np.array([np.cos(d_phi), np.sin(d_phi)]) * self.dx)

            # make a forward pass from the network
            motor = self._net(sun, flow)
            print "D_phi: % 2.2f" % np.rad2deg(d_phi),
            print "MTR: % 2.2f" % motor,
            if isinstance(sun, np.ndarray) and sun.size == 8:
                sun = decode_sun(sun)[0]
            print "lon: % 2.2f" % np.rad2deg(sun)

            counter += 1

            # update view
            img_func = None
            if self.visualiser is not None and self.visualiser.mode == "top":
                img_func = self.world.draw_top_view
            elif self.visualiser is not None and self.visualiser.mode == "panorama":
                img_func = self.world_snapshot
            if self.visualiser is not None:
                names = self.name.split('_')
                names[0] = self.world.date.strftime(datestr)
                names.append(counter)
                names.append(motor)
                names.append(distance)
                names.append(np.rad2deg(d_phi))

                capt_format = "%s " * (len(names) - 4) + "| C: % 2d MTR: %.2f Distance: %.2f D_phi: % 2.2f"
                self.visualiser.update_main(img_func, caption=capt_format % tuple(names))

            # update last orientation
            pphi = phi

        # remove the copy of the route from the world
        self.world.routes.remove(self.world.routes[-1])
        return rt     # return the learned route

    def start_homing(self, reset=True):
        if self.world is None:
            # TODO: warn about not setting the world
            return None

        if reset:
            print "Resetting..."
            super(CXAgent, self).reset()

        # initialise the visualisation
        if self.visualiser is not None:
            self.visualiser.reset()

        # add a copy of the current route to the world to visualise the path
        xs, ys, zs, phis = [self.pos[0]], [self.pos[1]], [self.pos[2]], [self.rot[1]]
        self.world.routes.append(route_like(
            self.world.routes[0], xs, ys, zs, phis, agent_no=self.id, route_no=len(self.world.routes) + 1)
        )

        d_nest = lambda: np.sqrt(np.square(self.pos[:2] - self.nest).sum())
        d_feeder = 0
        counter = 0
        start_time = datetime.now()
        phi = self.rot[1]
        pphi = phi
        while d_nest() > 0.1:
            x, y, z = self.pos
            phi = self.rot[1]
            d_phi = phi - pphi
            self.compass.facing_direction = phi

            if self.visualiser is not None and self.visualiser.is_quit():
                break

            sun = self.read_sensor()
            heading = self.get_heading(sun, phi, phi - d_phi)
            flow = self._net.get_flow(heading, np.array([np.cos(d_phi), np.sin(d_phi)]) * self.dx)

            # make a forward pass from the network
            motor = self._net(sun, flow)
            print "MTR: % 2.2f" % motor,
            print "D_phi: % 2.2f" % np.rad2deg(d_phi),
            if isinstance(sun, np.ndarray) and sun.size == 8:
                sun = decode_sun(sun)[0]
            print "lon: % 2.2f" % np.rad2deg(sun)

            d_phi = 10 * motor * np.pi
            # d_phi = np.sign(motor) * np.deg2rad(15)
            pphi = phi
            phi += d_phi

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
            if self.visualiser is not None and self.visualiser.mode == "top":
                img_func = self.world.draw_top_view
            # elif self.visualiser.mode == "panorama":
            #     img_func = self.world_snapshot
            if self.visualiser is not None:
                now = datetime.now() - start_time
                names = self.name.split('_')
                names[0] = self.world.date.strftime(datestr)
                names.append(counter)
                names.append(np.rad2deg(d_phi))
                names.append(d_feeder)
                names.append(d_nest())
                names.append(now.seconds // 60)
                names.append(now.seconds % 60)

                capt_format = "%s " * (len(names) - 5) + "| C: % 2d, D_phi: % 3d, D: %.2f, D_nest: %.2f | " \
                                                         "Elapsed time: %02d:%02d"
                self.visualiser.update_main(img_func, caption=capt_format % tuple(names))

            if d_feeder > 15:
                break
            d_feeder += self.dx
        self.world.routes.remove(self.world.routes[-1])
        return Route(xs, ys, zs, phis, condition=self.condition, agent_no=self.id, route_no=len(self.world.routes) + 1)

    def get_heading(self, sun, phi, pphi):
        if isinstance(sun, np.ndarray) and sun.size == 8:
            sun = decode_sun(sun)[0]
        return -(phi - pphi)

    def read_sensor(self, decode=False):
        self.compass.facing_direction = self.rot[1]
        return self.world.sky.lon - self.compass.facing_direction
        # if decode:
        #     return self.compass(self.world.sky, decode=decode)[0]
        # else:
        #     return self.compass(self.world.sky, decode=decode)


if __name__ == "__main__":
    from world import load_world, load_routes, save_route
    from world.utils import shifted_datetime
    from utils import create_agent_name, update_tests
    from visualiser import Visualiser

    exps = [
        # (True, False, True, False, None),     # live
        # (True, False, True, True, None),      # live-rgb
        # (True, False, False, False, None),    # live-no-pol
        # (True, False, False, True, None),     # live-no-pol-rgb
        # (False, True, True, False, np.random.RandomState(2018)),  # uniform
        # (False, True, True, True, np.random.RandomState(2018)),  # uniform-rgb
        # (False, False, True, False, None),    # fixed
        # (False, False, True, True, None),     # fixed-rgb
        # (False, False, False, False, None),    # fixed-no-pol
        (False, False, False, True, None),     # fixed-no-pol-rgb
    ]

    bin = True

    for update_sky, uniform_sky, enable_pol, rgb, rng in exps:
        date = shifted_datetime()
        if rng is None:
            rng = np.random.RandomState(2018)
        RND = rng
        fov = (-np.pi/2, np.pi/2)
        # fov = (-np.pi/6, np.pi/2)
        sky_type = "uniform" if uniform_sky else "live" if update_sky else "fixed"
        if not enable_pol and "uniform" not in sky_type:
            sky_type += "-no-pol"
        if rgb:
            sky_type += "-rgb"
        step = .1       # 10 cm
        tau_phi = np.pi    # 60 deg
        condition = Hybrid(tau_x=step, tau_phi=tau_phi)
        agent_name = create_agent_name(date, sky_type, step, fov[0], fov[1])
        print agent_name

        world = load_world()
        world.enable_pol_filters(enable_pol)
        world.uniform_sky = uniform_sky
        routes = load_routes()
        world.add_route(routes[0])

        agent = CXAgent(condition=condition, live_sky=update_sky,
                        # visualiser=Visualiser(),
                        rgb=rgb, fov=fov, name=agent_name)
        agent.set_world(world)
        print agent.homing_routes[0]

        if agent.visualiser is not None:
            agent.visualiser.set_mode("panorama")
        route = agent.start_learning_walk()
        print "Learned route:", route

        if agent.visualiser is not None:
            agent.visualiser.set_mode("top")
        route = agent.start_homing(reset=False)
        print route
        # if route is not None:
        #     save_route(route, agent_name)

        # if not update_tests(sky_type, date, step, gfov=fov[0], sfov=fov[1], bin=bin):
        #     break
        agent.world.routes.append(route)
        img, _ = agent.world.draw_top_view(1000, 1000)
        # img.save(__data__ + "routes-img/%s.png" % agent_name, "PNG")
        img.show(title="Testing route")
