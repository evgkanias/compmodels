import numpy as np
from world import Route, World, save_route
from net import Willshaw


class Agent(object):
    __latest_agent_id__ = 0

    def __init__(self, init_pos=np.zeros(3), init_rot=np.zeros(2), live_sky=True, name=None):
        """

        :param init_pos: the initial position
        :param init_rot: the initial orientation
        :param live_sky: flag to update the sky with respect to the time
        :param name: a name for the agent
        """
        self.pos = init_pos
        self.rot = init_rot
        self.nest = np.zeros(2)
        self.feeder = np.zeros(2)
        self.live_sky = live_sky

        self.homing_routes = []
        self.world = None
        self._net = Willshaw()  # learning_rate=1)
        self.__is_foraging = False
        self.__is_homing = False
        self.dx = 0

        Agent.__latest_agent_id__ += 1
        self.id = Agent.__latest_agent_id__
        if name is None:
            self.name = "agent_%02d" % Agent.__latest_agent_id__
        else:
            self.name = name

    def reset(self):
        """
        Resets the agent at the feeder

        :return: a boolean notifying whether the update of the position and orientation is done or not
        """
        self.__is_foraging = False
        self.__is_homing = True
        self._net.update = False

        if len(self.homing_routes) > 0:
            self.pos[:2] = self.feeder.copy()
            self.rot[1] = self.homing_routes[-1].phi[-1]
            return True
        else:
            # TODO: warn about the existence of the route
            return False

    def add_homing_route(self, rt):
        """
        Updates the homing route, home and nest points.

        :param rt: The route from the feeder to the nest
        :type rt: Route
        :return: a boolean notifying whether the update is done or not
        """
        if not isinstance(rt, Route):
            return False

        if rt not in self.homing_routes:
            self.homing_routes.append(rt)
            self.nest = np.array(rt.xy[-1])
            self.feeder = np.array(rt.xy[0])
            self.dx = rt.dx
            return True
        return False

    def set_world(self, w):
        """
        Update the world of the agent.

        :param w: the world to be placed in
        :return: a boolean notifying whether the update is done or not
        """
        if not isinstance(w, World):
            return False

        self.world = w
        for rt in self.world.routes:
            self.add_homing_route(rt)
        self.world.routes = self.homing_routes
        return True

    def start_learning_walk(self, visualise=None):
        if self.world is None:
            # TODO: warn about not setting the world
            yield None
            return
        elif len(self.homing_routes) == 0:
            # TODO: warn about not setting the homing route
            yield None
            return

        if visualise in ["top", "panorama"]:
            import pygame

            pygame.init()
            done = False
            if visualise == "top":
                screen = pygame.display.set_mode((1000, 1000))
            elif visualise == "panorama":
                screen = pygame.display.set_mode((1000, 500))

        self._net.update = True

        for r in self.homing_routes:
            xs, ys, zs, phis = [self.pos[0]], [self.pos[1]], [self.pos[2]], [self.rot[1]]
            self.world.routes.append(Route(xs, ys, zs, phis, self.id, len(self.world.routes) + 1))
            counter = -1
            pphi = r.phi[0]
            for x, y, z, phi in r:
                if visualise in ["top", "panorama"]:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                    if done:
                        break
                self.pos[:] = x, y, z
                self.rot[1] = phi
                dx = np.sqrt(np.square(self.pos[:2] - self.feeder[:2]).sum())
                distance = dx * self.world.ratio2meters

                if np.abs(phi - pphi) > np.pi / 32 or distance // 1 > counter:
                    img, _ = self.world.draw_panoramic_view(x, y, z, phi, update_sky=self.live_sky)
                    inp = np.array(img).reshape((-1, 3))[:, 1].flatten()
                    en = self._net(inp)
                    print "Distance:", distance, "d_phi:", np.rad2deg(np.abs(phi - pphi))
                    print counter, "EN:", en[0]
                    counter += 1
                pphi = phi

                xs.append(self.pos[0])
                ys.append(self.pos[1])
                zs.append(self.pos[2])
                phis.append(self.rot[1])
                self.world.routes[-1] = Route(xs, ys, zs, phis, self.id, len(self.world.routes))

                if visualise == "top":
                    img, _ = self.world.draw_top_view(width=1000, length=1000)
                elif visualise == "panorama":
                    img, _ = self.world.draw_panoramic_view(
                        (self.pos[0] + .5) * self.world.ratio2meters,
                        (self.pos[1] + .5) * self.world.ratio2meters,
                        (self.pos[2] + .5) * self.world.ratio2meters, self.rot[1],
                        width=1000, length=1000, height=500)
                if visualise in ["top", "panorama"]:
                    screen.blit(pygame.image.fromstring(img.tobytes("raw", "RGB"), img.size, "RGB"), (0, 0))
                    pygame.display.flip()

                    if done:
                        break
            self.world.routes.remove(self.world.routes[-1])
            yield r

        self._net.update = False

    def start_homing(self, reset=True, visualise=None):
        if self.world is None:
            # TODO: warn about not setting the world
            return None

        if reset:
            print "Resetting..."
            self.reset()

        if visualise in ["top", "panorama"]:
            import pygame

            pygame.init()
            done = False
            if visualise == "top":
                screen = pygame.display.set_mode((1000, 1000))
            elif visualise == "panorama":
                screen = pygame.display.set_mode((1000, 500))

        xs, ys, zs, phis = [self.pos[0]], [self.pos[1]], [self.pos[2]], [self.rot[1]]
        self.world.routes.append(Route(xs, ys, zs, phis, self.id, len(self.world.routes) + 1))
        d_nest = lambda: np.sqrt(np.square(self.pos[:2] - self.nest).sum()) * self.world.ratio2meters
        d_feeder = 0
        counter = 0
        while d_nest() > 0.1:
            x, y, z = self.pos
            phi = self.rot[1]
            if d_feeder // .1 > counter:
                en = []
                for dphi in np.linspace(-np.pi / 6, np.pi / 6, 61):
                    if visualise in ["top", "panorama"]:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                done = True
                        if done:
                            break

                    img, _ = self.world.draw_panoramic_view(
                        (x + .5) * self.world.ratio2meters,
                        (y + .5) * self.world.ratio2meters,
                        (z + .5) * self.world.ratio2meters, phi + dphi, update_sky=self.live_sky)
                    inp = np.array(img).reshape((-1, 3))[:, 1].flatten()
                    en.append(self._net(inp))
                en = np.array(en).flatten()
                # show preference to the least turning angle
                en += np.append(np.linspace(.01, 0., 30, endpoint=False), np.linspace(0., .01, 31))
                # print "EN:", en
                print "C: %03d, EN: % 3d (%.2f), D: %.2f, D_nest: %.2f" % (
                    counter, en.argmin() - 30, en.min(), d_feeder, d_nest())
                print "EN:" + " %.2f" * 61 % tuple(en)
                phi += np.deg2rad(en.argmin() - 30)

                counter += 1

            self.rot[1] = phi
            self.pos[:] = x + self.dx * np.cos(phi), y + self.dx * np.sin(phi), z
            xs.append(self.pos[0])
            ys.append(self.pos[1])
            zs.append(self.pos[2])
            phis.append(self.rot[1])

            self.world.routes[-1] = Route(xs, ys, zs, phis, self.id, len(self.world.routes))

            if visualise == "top":
                img, _ = self.world.draw_top_view(width=1000, length=1000)
            elif visualise == "panorama":
                img, _ = self.world.draw_panoramic_view(
                    (self.pos[0] + .5) * self.world.ratio2meters,
                    (self.pos[1] + .5) * self.world.ratio2meters,
                    (self.pos[2] + .5) * self.world.ratio2meters, self.rot[1],
                    width=1000, length=1000, height=500)
            if visualise in ["top", "panorama"]:
                screen.blit(pygame.image.fromstring(img.tobytes("raw", "RGB"), img.size, "RGB"), (0, 0))
                pygame.display.flip()

                if done:
                    break

            if d_feeder > 9:
                break
            d_feeder += self.dx * self.world.ratio2meters
        return Route(xs, ys, zs, phis, nant=self.id, nroute=len(self.world.routes) + 1)


if __name__ == "__main__":
    from world import load_world, load_routes

    update_sky = False
    uniform_sky = False

    world = load_world()
    world.uniform_sky = uniform_sky
    routes = load_routes()
    routes[0].dx = .1  # 10cm
    world.add_route(routes[0])
    print world.routes[0]

    img, _ = world.draw_top_view(1000, 1000)
    img.save("training-route.png", "PNG")
    # img.show(title="Training route")

    agent = Agent()
    agent.live_sky = update_sky
    agent.set_world(world)
    for route in agent.start_learning_walk(visualise="panorama"):
        print "Learned route:", route
        if route is not None:
            save_route(route, "learned-%d-%d" % (route.nant, route.nroute))

    route = agent.start_homing(visualise="top")
    print route
    if route is not None:
        save_route(route, "homing-%d-%d" % (route.nant, route.nroute))

    del world.routes[:]
    world.routes.append(route)
    img, _ = world.draw_top_view(1000, 1000)
    img.save("testing-route.png", "PNG")
    img.show(title="Testing route")
