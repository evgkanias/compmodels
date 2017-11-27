import numpy as np

from world import World, Route, Hybrid
from utils import *


class Agent(object):
    __latest_agent_id__ = 0

    def __init__(self, init_pos=np.zeros(3), init_rot=np.zeros(2), condition=Hybrid(),
                 live_sky=True, rgb=False, visualiser=None, name=None):
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
        :param visualiser:
        :type visualiser: Visualiser
        :param name: a name for the agent
        :type name: string
        """

        self.pos = init_pos
        self.rot = init_rot
        self.nest = np.zeros(2)
        self.feeder = np.zeros(2)
        self.live_sky = live_sky
        self.rgb = rgb
        self.visualiser = visualiser

        self.homing_routes = []
        self.world = None  # type: World
        self.__is_foraging = False
        self.__is_homing = False
        self.dx = 0.  # type: float
        self.condition = condition

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

        if len(self.homing_routes) > 0:
            self.pos[:2] = self.feeder.copy()
            self.rot[1] = self.homing_routes[-1].phi[0]
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
            rt.condition = self.condition
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

    def start_homing(self, reset=True):
        raise NotImplementedError()
