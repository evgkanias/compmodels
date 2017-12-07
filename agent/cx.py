import numpy as np
from base import Agent, Logger
from world import Hybrid, Route, route_like
from net import CX
from compoundeye import CompassSensor, decode_sun
from datetime import datetime
from utils import datestr


# ACCELERATION = .15  # a good value because keeps speed under 1
ACCELERATION = 1e-03
DRAG = .0


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

        self._net = CX(noise=0., pontin=False)
        self.compass = compass
        self.bump_shift = 0.
        self.log = CXLogger()

        try:
            self.compass.load_weights()
        except Exception, e:
            print "No parameters found for compass."
            print e.message

        if 'name' in kwargs.keys() and kwargs['name'] is None:
            self.name = "cx_agent_%02d" % self.id

    def reset(self):
        if super(CXAgent, self).reset():
            # reset to the nest instead of the feeder
            self.pos[:2] = self.nest.copy()
            self.rot[1] = (self.homing_routes[-1].phi[-2] + np.pi) % (2 * np.pi)
            self._net.update = True

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
        self.log.set_stage("training")

        # initialise visualisation
        if self.visualiser is not None:
            self.visualiser.reset()

        # follow a reverse homing route
        rt = self.homing_routes[-1].reverse()  # type: Route

        # add a copy of the current route to the world to visualise the path
        self.log.add(self.pos[:3], self.rot[1])
        self.world.routes.append(
            route_like(rt, self.log.x, self.log.y, self.log.z, self.log.phi,
                       self.condition, agent_no=rt.agent_no, route_no=rt.route_no)
        )
        counter = 0         # count the steps

        phi_ = (np.array([np.pi - phi for _, _, _, phi in rt]) + np.pi) % (2 * np.pi) - np.pi
        # phi_ = np.roll(phi_, 1)  # type: np.ndarray

        for phi in phi_:
            # stop the loop when we close the visualisation window
            if self.visualiser is not None and self.visualiser.is_quit():
                break

            phi, v = self.update_state(phi)
            # calculate the distance from the start position (feeder)
            distance = np.sqrt(np.square(self.pos[:2] - self.feeder[:2]).sum())

            # update the route in the world
            self.world.routes[-1] = route_like(self.world.routes[-1], self.log.x, self.log.y, self.log.z, self.log.phi)

            sun = self.read_sensor()
            if isinstance(sun, np.ndarray) and sun.size == 8:
                heading = decode_sun(sun)[0]
            else:
                heading = sun
            v_trans = self.transform_velocity(heading, v)
            flow = self._net.get_flow(heading, v_trans)
            # flow = self._net.get_flow(__phi, v)

            # make a forward pass from the network
            motor = self._net(sun, flow)

            self.log.update_hist(tl2=self._net.tl2, cl1=self._net.cl1, tb1=self._net.tb1, cpu4=self._net.cpu4_mem,
                                 cpu1=self._net.cpu1, tn1=self._net.tn1, tn2=self._net.tn2, motor0=motor,
                                 flow0=flow, v0=v, v1=v_trans, phi=phi, sun=heading)
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
                names.append(np.rad2deg(phi))

                capt_format = "%s " * (len(names) - 4) + "| C: % 2d MTR: %.2f Distance: %.2f D_phi: % 2.2f"
                self.visualiser.update_main(img_func, caption=capt_format % tuple(names))

        self.log.update_outbound_end()
        # remove the copy of the route from the world
        rt = self.world.routes[-1]
        self.world.routes.remove(self.world.routes[-1])

        return rt     # return the learned route

    def start_homing(self, reset=True):
        if self.world is None:
            # TODO: warn about not setting the world
            return None

        if reset:
            print "Resetting..."
            super(CXAgent, self).reset()
        self.log.set_stage("homing")

        # initialise the visualisation
        if self.visualiser is not None:
            self.visualiser.reset()

        phi, v = self.update_state(self.rot[1])
        # add a copy of the current route to the world to visualise the path
        self.world.routes.append(route_like(
            self.world.routes[0], self.log.x, self.log.y, self.log.z, self.log.phi,
            agent_no=self.id, route_no=len(self.world.routes) + 1)
        )

        d_nest = lambda: np.sqrt(np.square(self.pos[:2] - self.nest).sum())
        d_feeder = 0
        counter = 0
        start_time = datetime.now()
        while d_nest() > 0.1:

            if self.visualiser is not None and self.visualiser.is_quit():
                break

            sun = self.read_sensor()
            if isinstance(sun, np.ndarray) and sun.size == 8:
                heading = decode_sun(sun)[0]
            else:
                heading = sun
            v_trans = self.transform_velocity(heading, v)
            flow = self._net.get_flow(heading, v_trans)

            # make a forward pass from the network
            motor = self._net(sun, flow)
            self.log.update_hist(tl2=self._net.tl2, cl1=self._net.cl1, tb1=self._net.tb1, cpu4=self._net.cpu4_mem,
                                 cpu1=self._net.cpu1, tn1=self._net.tn1, tn2=self._net.tn2, motor0=motor,
                                 flow0=flow, v0=v, v1=v_trans, phi=phi, sun=heading)

            phi, v = self.update_state(phi, rotation=motor)
            counter += 1

            self.world.routes[-1] = route_like(self.world.routes[-1], self.log.x, self.log.y, self.log.z, self.log.phi)

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
                names.append(np.rad2deg(motor))
                names.append(d_feeder)
                names.append(d_nest())
                names.append(now.seconds // 60)
                names.append(now.seconds % 60)

                capt_format = "%s " * (len(names) - 5) + "| C: % 2d, D_phi: % 3d, D: %.2f, D_nest: %.2f | " \
                                                         "Elapsed time: %02d:%02d"
                self.visualiser.update_main(img_func, caption=capt_format % tuple(names))

            if d_feeder > 15:
                break
            d_feeder += np.sqrt(np.square(v).sum())
        self.world.routes.remove(self.world.routes[-1])
        return Route(self.log.x, self.log.y, self.log.z, self.log.phi, condition=self.condition,
                     agent_no=self.id, route_no=len(self.world.routes) + 1)

    def read_sensor(self, decode=True):
        self.compass.facing_direction = np.pi - self.rot[1]
        if decode:
            # sun = self.compass.facing_direction - self.world.sky.lon
            sun = self.compass(self.world.sky, decode=decode)[0]
            sun = (sun + np.pi) % (2 * np.pi) - np.pi
        else:
            sun = self.compass(self.world.sky, decode=decode)

        return sun

    def transform_velocity(self, heading, velocity):
        """

        :param heading:
        :param velocity:
        :type velocity: np.ndarray
        :return:
        """
        # R = np.array([[np.sin(heading), -np.cos(heading)],
        #               [np.cos(heading), np.sin(heading)]]).T

        r = np.sqrt(np.square(velocity).sum())
        return self.get_velocity(heading, r)

    def update_state(self, heading, rotation=0):
        phi, v = self.translate(heading, rotation, self.dx)

        # update the agent position
        self.pos[:] += np.array([v[0], -v[1], 0.])
        self.rot[1] = np.pi - phi
        self.log.add(self.pos[:3], self.rot[1])

        return phi, v

    @staticmethod
    def translate(heading, rotation, acceleration):
        phi = CXAgent.rotate(heading, rotation)
        v = CXAgent.get_velocity(phi, acceleration)
        return phi, v

    @staticmethod
    def rotate(heading, rotation):
        return ((heading + rotation + np.pi) % (2 * np.pi)) - np.pi

    @staticmethod
    def get_velocity(phi, acceleration):
        return np.array([np.sin(phi), np.cos(phi)]) * acceleration


class CXLogger(Logger):

    def reset(self):
        super(CXLogger, self).reset()

        self.hist["cpu1"] = []
        self.hist["cpu4"] = []
        self.hist["tb1"] = []
        self.hist["tl2"] = []
        self.hist["cl1"] = []
        self.hist["tn1"] = []
        self.hist["tn2"] = []
        self.hist["flow0"] = []
        self.hist["v0"] = []
        self.hist["v1"] = []
        self.hist["phi"] = []
        self.hist["sun"] = []
        self.hist["motor0"] = []
        self.hist["outbound_end"] = -1

    def update_outbound_end(self):
        self.hist["outbound_end"] = len(self.hist["flow0"]) - 1


if __name__ == "__main__":
    from world import load_world, load_routes
    from world.utils import shifted_datetime
    from utils import create_agent_name
    from visualiser import Visualiser

    exps = [
        (False, False, True, False, None),    # fixed
        (False, False, True, True, None),     # fixed-rgb
        (False, False, False, False, None),    # fixed-no-pol
        (False, False, False, True, None),     # fixed-no-pol-rgb
    ]

    bin = True
    show = True
    i = 0

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
        step = .01         # 1 cm
        tau_phi = np.pi    # 180 deg
        condition = Hybrid(tau_x=step, tau_phi=tau_phi)
        agent_name = create_agent_name(date, sky_type, step, fov[0], fov[1])
        print agent_name

        world = load_world()
        world.enable_pol_filters(enable_pol)
        world.uniform_sky = uniform_sky
        routes = load_routes()
        route = routes[i]
        route.agent_no = 1
        route.route_no = 2
        world.add_route(route)
        i += 1

        agent = CXAgent(condition=condition, live_sky=update_sky,
                        # visualiser=Visualiser(),
                        rgb=rgb, fov=fov, name=agent_name)
        agent.id = i + 1
        agent.set_world(world)
        print agent.homing_routes[0]

        if agent.visualiser is not None:
            agent.visualiser.set_mode("panorama")
        route1 = agent.start_learning_walk()
        print "Learned route:", route1

        # if agent.visualiser is not None:
        #     agent.visualiser.set_mode("top")
        route2 = agent.start_homing(reset=False)
        print "Homing route: ", route2
        # if route2 is not None:
        #     save_route(route2, agent_name)

        # if not update_tests(sky_type, date, step, gfov=fov[0], sfov=fov[1], bin=bin):
        #     break
        agent.world.routes.remove(agent.world.routes[0])
        agent.world.routes.append(route1)
        agent.world.routes.append(route2)
        img, _ = agent.world.draw_top_view(1000, 1000)
        # img.save(__data__ + "routes-img/%s.png" % agent_name, "PNG")
        img.show(title="Testing route")

        if show:
            import matplotlib.pyplot as plt
            from PIL import Image

            comp = agent.compass.facing_direction

            T = len(agent.log.hist["flow0"])
            T_ob = agent.log.hist["outbound_end"] + 1
            T_ib = T - T_ob

            N, N_U = 38., 40.
            ticks = int(T * N / T_ob)
            xticks = 2 * np.arange(0, ticks, step=10) / N_U
            y = 2 * np.arange(0, ticks, step=10) / (10 * agent.dx)
            x = np.interp(np.arange(T), y, xticks)
            xlim = [0, x[-1]]
            # xlim = [0, np.sum(y < 2)]

            plt.subplot(5, 2, 1)
            plt.grid()
            # plt.plot(x, np.array(agent.hist["flow0"])[:, 0], label=r"flow_x")
            # plt.plot(x, np.array(agent.hist["flow0"])[:, 1], label=r"flow_y")
            plt.plot(x, np.array(agent.log.hist["v0"])[:, 0], label=r"v_x")
            plt.plot(x, np.array(agent.log.hist["v0"])[:, 1], label=r"v_y")
            plt.plot(x, np.array(agent.log.hist["v1"])[:, 0], label=r"v'_x")
            plt.plot(x, np.array(agent.log.hist["v1"])[:, 1], label=r"v'_y")
            plt.legend()
            plt.xticks(xticks)  # , [""])
            plt.yticks([-agent.dx, 0., agent.dx])
            plt.xlim(xlim)
            plt.ylim([-agent.dx, agent.dx])

            plt.subplot(5, 2, 3)
            plt.grid()
            plt.plot(x, np.array(agent.log.hist["phi"]), label="phi")
            plt.plot(x, np.array(agent.log.hist["sun"]), label="rel sun")
            plt.plot(x, np.ones_like(x) * world.sky.lon, label="abs sun")
            plt.legend()
            plt.xticks(xticks, [""])
            plt.xlim(xlim)
            plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ["-pi", "-pi/2", "0", "pi/2", "pi"])
            plt.ylim([-np.pi, np.pi])
            plt.ylabel("phi")

            plt.subplot(5, 2, 5)
            plt.grid()
            plt.plot(x, np.array(agent.log.hist["tn1"])[:, 0], label="TN1_x")
            plt.plot(x, np.array(agent.log.hist["tn1"])[:, 1], label="TN1_y")
            plt.plot(x, np.array(agent.log.hist["tn2"])[:, 0], label="TN2_x")
            plt.plot(x, np.array(agent.log.hist["tn2"])[:, 1], label="TN2_y")
            plt.legend()
            plt.xticks(xticks, [""])
            plt.yticks([0., .5, 1])
            plt.xlim(xlim)
            plt.ylim([0, 1])

            plt.subplot(5, 2, 7)
            plt.grid()
            plt.plot(x, np.array(agent.log.hist["motor0"]))
            plt.xticks(xticks, [""])
            plt.xlim(xlim)
            plt.yticks([-np.pi/4, 0, np.pi/4], ["-pi/4", "0", "pi/4"])
            plt.ylim([-np.pi/3, np.pi/3])
            plt.ylabel("motor")

            plt.subplot(5, 2, 9)
            tb1 = np.array(agent.log.hist["tb1"]).T
            tb1_img = Image.fromarray(tb1)
            tb1_img = tb1_img.resize((ticks, tb1.shape[0]))
            plt.imshow(np.array(tb1_img), vmin=0, vmax=1)
            plt.xticks(N_U * xticks / 2., xticks)
            # plt.xlim(xlim)
            plt.yticks([0, 7], [1, 8])
            plt.xlabel("Time (sec)")
            plt.ylabel("TB1")

            plt.subplot(5, 2, 2)
            tl2 = np.array(agent.log.hist["tl2"]).T
            tl2_img = Image.fromarray(tl2)
            tl2_img = tl2_img.resize((ticks, tl2.shape[0]))
            plt.imshow(np.array(tl2_img), vmin=0, vmax=1)
            # plt.xlim(xlim)
            plt.xticks(N_U * xticks / 2., [""])
            plt.yticks([0, 15], [1, 16])
            plt.ylabel("TL2")

            plt.subplot(5, 2, 4)
            cl1 = np.array(agent.log.hist["cl1"]).T
            cl1_img = Image.fromarray(cl1)
            cl1_img = cl1_img.resize((ticks, cl1.shape[0]))
            plt.imshow(np.array(cl1_img), vmin=0, vmax=1)
            # plt.xlim(xlim)
            plt.xticks(N_U * xticks / 2., [""])
            plt.yticks([0, 15], [1, 16])
            plt.ylabel("CL1")

            plt.subplot(5, 2, 6)
            cpu4 = np.array(agent.log.hist["cpu4"]).T
            cpu4_img = Image.fromarray(cpu4)
            cpu4_img = cpu4_img.resize((ticks, cpu4.shape[0]))
            plt.imshow(np.array(cpu4_img), vmin=0, vmax=1)
            # plt.xlim(xlim)
            plt.xticks(N_U * xticks / 2., [""])
            plt.yticks([0, 15], [1, 16])
            plt.ylabel("CPU4")

            ax = plt.subplot(5, 2, 8)
            cpu1 = np.array(agent.log.hist["cpu1"]).T
            cpu1_img = Image.fromarray(cpu1)
            cpu1_img = cpu1_img.resize((ticks, cpu1.shape[0]))
            plt.imshow(np.array(cpu1_img), vmin=0, vmax=1)
            # plt.xlim(xlim)
            plt.xticks(N_U * xticks / 2., xticks)
            plt.yticks([0, 15], [1, 16])
            plt.xlabel("Time (sec)")
            plt.ylabel("CPU1")

            cax = plt.subplot(15, 2, 28)
            plt.colorbar(cax=cax, ax=ax, orientation="horizontal")

            plt.show()

