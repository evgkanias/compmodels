import numpy as np
import numpy.linalg as la
from copy import copy
from datetime import datetime
from numbers import Number
from PIL import ImageDraw, Image
from colorsys import hsv_to_rgb
from utils import vec2sph
from ephem import Observer
from sky import get_seville_observer, ChromaticitySkyModel
from compoundeye import CompoundEye
from conditions import *

WIDTH = 36
HEIGHT = 10
LENGTH = 36

GRASS_COLOUR = (0, 255, 0)
GROUND_COLOUR = (229, 183, 90)
SKY_COLOUR = (13, 135, 201)


class World(object):

    def __init__(self, observer=None, polygons=None, width=WIDTH, length=LENGTH, height=HEIGHT, uniform_sky=False):
        """
        Creates a world.

        :param observer: a reference to an observer
        :type observer: Observer
        :param polygons: polygons of the objects in the world
        :type polygons: PolygonList
        :param width: the width of the world
        :type width: int
        :param length: the length of the world
        :type length: int
        :param height: the height of the world
        :type height: int
        :param uniform_sky: flag that indicates if there is a uniform sky or not
        :type uniform_sky: bool
        """
        # normalise world
        xmax = np.array([polygons.x.max(), polygons.y.max(), polygons.z.max()]).max()
        polygons.normalise(*((xmax,) * 3))

        # polygons = polygons * [width, length, height]
        # polygons = polygons + [width / 2, length / 2, height / 2]

        # default observer is in Seville (where the data come from)
        if observer is None:
            observer = get_seville_observer()
        observer.date = datetime.now()

        # create and generate a sky instance
        self.sky = ChromaticitySkyModel(observer=observer, nside=1)
        self.sky.generate()

        # create ommatidia positions with respect to the resolution
        # (this is for the sky drawing on the panoramic images)
        thetas = np.linspace(-np.pi, np.pi, width, endpoint=False)
        phis = np.linspace(np.pi/2, 0, height, endpoint=False)
        thetas, phis = np.meshgrid(phis, thetas)
        ommatidia = np.array([thetas.flatten(), phis.flatten()]).T

        # create a compound eye model for the sky pixels
        self.eye = CompoundEye(ommatidia)
        self.eye.set_sky(self.sky)

        # store the polygons and initialise the parameters
        self.polygons = polygons
        self.routes = []
        self.width = width
        self.length = length
        self.height = height
        self.__normalise_factor = xmax
        self.uniform_sky = uniform_sky

    @property
    def ratio2meters(self):
        return self.__normalise_factor

    def add_route(self, route):
        """
        Adds an ant-route in the world

        :param route: the new route
        :type route: Route
        :return: None
        """
        route.normalise(*((self.__normalise_factor,) * 3))
        self.routes.append(route)

    def draw_top_view(self, width=None, length=None, height=None):
        """
        Draws a top view of the world and all the added paths in it.

        :param width: the width of the world
        :type width: int
        :param length: the length of the world
        :type length: int
        :param height: the height of the world
        :type height: int
        :return: an image of the top view
        """

        # set the default values to the dimensions of the world
        if width is None:
            width = self.width
        if length is None:
            length = self.length
        if height is None:
            height = self.height

        # create new image and drawer
        image = Image.new("RGB", (width, length), GROUND_COLOUR)
        draw = ImageDraw.Draw(image)

        # rescale the polygons and put them in the centre of the world
        polygons = self.polygons * [width, length, height]
        polygons = polygons + [width / 2, length / 2, height / 2]

        # draw the polygons
        for p in polygons:
            draw.polygon(p.xy, fill=p.c_int32)

        # draw the routes
        nants = int(np.array([r.nant for r in self.routes]).max())      # the ants' ID
        nroutes = int(np.array([r.nroute for r in self.routes]).max())  # the routes' ID
        for route in self.routes:
            # transform the routes similarly to the polygons
            rt = route * [width, length, height]
            rt = rt + [width / 2, length / 2, height / 2]
            h = np.linspace(0, 1, nants)[rt.nant-1]
            s = np.linspace(0, 1, nroutes)[rt.nroute-1]
            v = .5
            r, g, b = hsv_to_rgb(h, s, v)
            draw.line(rt.xy, fill=(int(r * 255), int(g * 255), int(b * 255)))

        return image, draw

    def draw_panoramic_view(self, x=None, y=None, z=None, r=0, width=None, length=None, height=None, update_sky=True):
        """
        Draws a panoramic view of the world

        :param x: The x coordinate of the agent in the world
        :type x: float
        :param y: The y coordinate of the agent in the world
        :type y: float
        :param z: The z coordinate of the agent in the world
        :type z: float
        :param r: The orientation of the agent in the world
        :type r: float
        :param width: the width of the world
        :type width: int
        :param length: the length of the world
        :type length: int
        :param height: the height of the world
        :type height: int
        :param update_sky: flag that specifies if we want to update the sky
        :type update_sky: bool
        :return: an image showing the 360 degrees view of the agent
        """

        # set the default values for the dimensions of the world
        if width is None:
            width = self.width
        if length is None:
            length = self.length
        if height is None:
            height = self.height
        if x is None:
            x = width / 2.
        if y is None:
            y = length / 2.
        if z is None:
            z = height / 2. + .06 * height

        # create ommatidia positions with respect to the resolution
        # (this is for the sky drawing on the panoramic images)
        thetas = np.linspace(-np.pi, np.pi, width, endpoint=False)
        phis = np.linspace(np.pi/2, 0, height / 2, endpoint=False)
        thetas, phis = np.meshgrid(phis, thetas)
        ommatidia = np.array([thetas.flatten(), phis.flatten()]).T

        image = Image.new("RGB", (width, height), GROUND_COLOUR)
        draw = ImageDraw.Draw(image)

        if self.uniform_sky:
            draw.rectangle((0, 0, width, height/2), fill=SKY_COLOUR)
        else:
            # create a compound eye model for the sky pixels
            self.eye = CompoundEye(ommatidia)
            if update_sky:
                self.sky.obs.date = datetime.now()
                self.sky.generate()
            self.eye.facing_direction = -r
            self.eye.set_sky(self.sky)

            pix = image.load()
            for i, c in enumerate(self.eye.L):
                pix[i // (height / 2), i % (height / 2)] = tuple(np.int32(255 * c))

        R = np.array([
            [np.cos(r), -np.sin(r), 0],
            [np.sin(r), np.cos(r), 0],
            [0, 0, 1]
        ])
        thetas, phis, rhos = [], [], []
        polygons = self.polygons * [width, length, height]
        polygons = polygons + [width / 2, length / 2, height / 2]
        pos = np.array([x, y, z]) / self.ratio2meters - .5
        pos *= np.array([width, length, height])
        pos += np.array([width / 2, length / 2, height / 2])
        for p in polygons:
            theta, phi, rho = vec2sph((p.xyz - pos).dot(R))
            thetas.append(theta)
            phis.append(phi)
            rhos.append(rho)

        thetas = height * ((np.array(thetas) % np.pi) / np.pi)
        phis = width * ((np.pi + np.array(phis)) % (2 * np.pi)) / (2 * np.pi)
        rhos = la.norm(np.array(rhos), axis=-1)
        ind = np.argsort(rhos)[::-1]
        for theta, phi, c in zip(thetas[ind], phis[ind], polygons.c_int32[ind]):
            if phi.max() - phi.min() < width/2:  # normal conditions
                p = tuple((b, a) for a, b in zip(theta, phi))
                draw.polygon(p, fill=tuple(c))
            else:   # in case that the object is on the edge of the screen
                phi0, phi1 = phi.copy(), phi.copy()
                phi0[phi < width/2] += width
                phi1[phi >= width/2] -= width
                p = tuple((b, a) for a, b in zip(theta, phi0))
                draw.polygon(p, fill=tuple(c))
                p = tuple((b, a) for a, b in zip(theta, phi1))
                draw.polygon(p, fill=tuple(c))

            # draw visible polygons

        return image, draw


class PolygonList(list):

    def __init__(self, polygons=None):
        items = []
        ext = 0
        if polygons is not None:
            for p in polygons:
                if isinstance(p, Polygon):
                    items.append(p)
                else:
                    ext += 1
        if len(items) > 0:
            super(PolygonList, self).__init__(items)
        else:
            super(PolygonList, self).__init__()

        if ext > 0:
            print "Warning: %d elements extracted from the list because of wrong type." % ext

    @property
    def x(self):
        return np.array([p.x for p in self])

    @property
    def y(self):
        return np.array([p.y for p in self])

    @property
    def z(self):
        return np.array([p.z for p in self])

    @property
    def c(self):
        return np.array([p.c for p in self])

    @property
    def c_int32(self):
        return np.array([p.c_int32 for p in self])

    def normalise(self, xmax=None, ymax=None, zmax=None):
        for p in self:
            p.normalise(xmax, ymax, zmax)

    def __add__(self, other):
        ps = []
        for p in self:
            ps.append(p.__add__(other))
        return PolygonList(ps)

    def __sub__(self, other):
        ps = []
        for p in self:
            ps.append(p.__sub__(other))
        return PolygonList(ps)

    def __mul__(self, other):
        ps = []
        for p in self:
            ps.append(p.__mul__(other))
        return PolygonList(ps)

    def __div__(self, other):
        ps = []
        for p in self:
            ps.append(p.__div__(other))
        return PolygonList(ps)

    def __str__(self):
        s = "["
        if len(self) < 10:
            for p in self:
                s += p.__str__() + ",\n "
        else:
            for p in self[:3]:
                s += p.__str__() + ",\n "
            s += "   . . .\n "
            s += self[-1].__str__() + ",\n "
        return s[:-3] + "]"


class Polygon(object):

    def __init__(self, xs, ys, zs, colour=(0, 0, 0)):
        """

        :param xs: x coodrinates in meters
        :param ys: y coordinates in meters
        :param zs: z coordinates in meters
        :param colour: colour (R, G, B) in [0, 1] ^ 3
        """
        self.x = xs
        self.y = ys
        self.z = zs
        self._c = np.array(colour)

    @property
    def xyz(self):
        return tuple((x, y, z) for x, y, z in zip(self.x, self.y, self.z))

    @property
    def xy(self):
        return tuple((x, y) for x, y in zip(self.x, self.y))

    @property
    def xz(self):
        return tuple((x, z) for x, z in zip(self.x, self.z))

    @property
    def yx(self):
        return tuple((y, x) for y, x in zip(self.y, self.x))

    @property
    def yz(self):
        return tuple((y, z) for y, z in zip(self.y, self.z))

    @property
    def zx(self):
        return tuple((z, x) for z, x in zip(self.z, self.x))

    @property
    def zy(self):
        return tuple((z, y) for z, y in zip(self.z, self.y))

    @property
    def c(self):
        return tuple(self._c)

    @property
    def c_int32(self):
        return tuple(np.int32(self._c * 255))

    def normalise(self, xmax=None, ymax=None, zmax=None):
        if xmax is not None:
            self.x = (self.x / xmax - .5)
        if ymax is not None:
            self.y = (self.y / ymax - .5)
        if zmax is not None:
            self.z = (self.z / zmax - .5)

    def __add__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x += other[0]
            if len(other) > 1:
                p.y += other[1]
            elif len(other) > 0:
                p.y += other[0]
            if len(other) > 2:
                p.z += other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z += other[0]
        if isinstance(other, Number):
            p.x += other
            p.y += other
            p.z += other
        return p

    def __sub__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x -= other[0]
            if len(other) > 1:
                p.y -= other[1]
            elif len(other) > 0:
                p.y -= other[0]
            if len(other) > 2:
                p.z -= other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z -= other[0]
        if isinstance(other, Number):
            p.x -= other
            p.y -= other
            p.z -= other
        return p

    def __mul__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x *= other[0]
            if len(other) > 1:
                p.y *= other[1]
            elif len(other) > 0:
                p.y *= other[0]
            if len(other) > 2:
                p.z *= other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z *= other[0]
        if isinstance(other, Number):
            p.x *= other
            p.y *= other
            p.z *= other
        return p

    def __div__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x /= other[0]
            if len(other) > 1:
                p.y /= other[1]
            elif len(other) > 0:
                p.y /= other[0]
            if len(other) > 2:
                p.z /= other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z /= other[0]
        if isinstance(other, Number):
            p.x /= other
            p.y /= other
            p.z /= other
        return p

    def __copy__(self):
        return Polygon(self.x.copy(), self.y.copy(), self.z.copy(), self._c.copy())

    def __str__(self):
        s = "C: (%.2f, %.2f, %.2f), P: " % self.c
        for x, y, z in self.xyz:
            s += "(%.2f, %.2f, %.2f) " % (x, y, z)
        return s[:-1]


class Route(object):

    def __init__(self, xs, ys, zs=None, phis=None, condition=NoneCondition(), nant=None, nroute=None):
        self.x = np.array(xs)
        self.y = np.array(ys)
        if isinstance(zs, Number):
            self.z = np.ones_like(xs) * zs
        elif isinstance(zs, list) or isinstance(zs, np.ndarray):
            self.z = np.array(zs)
        else:
            self.z = np.ones_like(xs) * .01

        if isinstance(phis, Number):
            self.phi = np.ones_like(xs) * phis
        elif isinstance(phis, list) or isinstance(phis, np.ndarray):
            self.phi = np.array(phis)
        else:
            self.phi = np.arctan2(ys, xs)
        self.nant = nant if nant is not None else -1
        self.nroute = nroute if nroute is not None else -1

        self.__condition = condition
        dx = np.sqrt(np.square(self.x[1:] - self.x[:-1]) + np.square(self.y[1:] - self.y[:-1]))
        self.__mean_dx = dx.mean() if dx.size > 0 else 0.
        self.dt = 2. / self.x.size  # the duration of the route is 2s
        self.__normalise_factor = 1.

    @property
    def dx(self):
        if isinstance(self.condition, Stepper):
            return self.condition.__step / self.__normalise_factor
        else:
            return self.__mean_dx / self.__normalise_factor

    @dx.setter
    def dx(self, value):
        if isinstance(self.condition, Stepper):
            self.condition.__step = value
        else:
            self.__mean_dx = value

    @property
    def condition(self):
        return self.__condition

    @condition.setter
    def condition(self, value):
        if isinstance(value, Stepper):
            self.__mean_dx = value.__step
        self.__condition = value

    @property
    def xyz(self):
        return tuple((x, y, z) for x, y, z, _ in self.__iter__())

    @property
    def xy(self):
        return tuple((x, y) for x, y, _, _ in self.__iter__())

    def normalise(self, xmax=None, ymax=None, zmax=None):
        if xmax is not None:
            self.x = (self.x / xmax - .5)
            self.__normalise_factor = xmax
        if ymax is not None:
            self.y = (self.y / ymax - .5)
            if xmax is None:
                self.__normalise_factor = ymax
        if zmax is not None:
            self.z = (self.z / zmax - .5)
            if xmax is None and ymax is None:
                self.__normalise_factor = zmax

    def __iter__(self):
        px, py, pz, p_phi = self.x[0], self.y[0], self.z[0], self.phi[0]
        phi = 0.

        for x, y, z in zip(self.x[1:], self.y[1:], self.z[1:]):
            dv = np.array([x - px, y - py, z - pz])
            d = np.sqrt(np.square(dv).sum())
            phi = np.arctan2(dv[1], dv[0])
            if self.condition.valid(d * self.__normalise_factor, np.abs(phi - p_phi)):
                yield px, py, pz, phi
                px, py, pz, pphi = x, y, z, phi

        yield px, py, pz, phi

    def __add__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x += other[0]
            if len(other) > 1:
                p.y += other[1]
            elif len(other) > 0:
                p.y += other[0]
            if len(other) > 2:
                p.z += other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z += other[0]
        if isinstance(other, Number):
            p.x += other
            p.y += other
            p.z += other
        return p

    def __sub__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x -= other[0]
            if len(other) > 1:
                p.y -= other[1]
            elif len(other) > 0:
                p.y -= other[0]
            if len(other) > 2:
                p.z -= other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z -= other[0]
        if isinstance(other, Number):
            p.x -= other
            p.y -= other
            p.z -= other
        return p

    def __mul__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x *= other[0]
                p.__normalise_factor /= other[0]
            if len(other) > 1:
                p.y *= other[1]
                p.__normalise_factor *= other[0]
                p.__normalise_factor /= np.sqrt(np.square(other[:2]).sum())
            elif len(other) > 0:
                p.y *= other[0]
            if len(other) > 2:
                p.z *= other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z *= other[0]
        if isinstance(other, Number):
            p.x *= other
            p.y *= other
            p.z *= other
            p.__normalise_factor /= other
        return p

    def __div__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x /= other[0]
                p.__normalise_factor *= other[0]
            if len(other) > 1:
                p.y /= other[1]
                p.__normalise_factor /= other[0]
                p.__normalise_factor *= np.sqrt(np.square(other[:2]).sum())
            elif len(other) > 0:
                p.y /= other[0]
            if len(other) > 2:
                p.z /= other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z /= other[0]
        if isinstance(other, Number):
            p.x /= other
            p.y /= other
            p.z /= other
            p.__normalise_factor *= other
        return p

    def __copy__(self):
        r = Route(xs=self.x.copy(), ys=self.y.copy(), zs=self.z.copy(), phis=self.phi.copy(),
                  condition=self.condition, nant=self.nant, nroute=self.nroute)
        r.__normalise_factor = self.__normalise_factor
        return r

    def __str__(self):
        if self.nant > 0 and self.nroute > 0:
            s = "Ant: %02d, Route %02d," % (self.nant, self.nroute)
        elif self.nant > 0 >= self.nroute:
            s = "Ant: %02d," % self.nant
        elif self.nant <= 0 < self.nroute:
            s = "Route: %02d," % self.nroute
        else:
            s = "Route:"
        for x, y, z in self.xyz:
            s += " (%.2f, %.2f)" % (x, y)
        s += ", Step: % 2.2f" % self.dx
        return s[:-1]

    def save(self, filename):
        np.savez_compressed(filename,
                            ant=self.nant, route=self.nroute, dt=self.dt,
                            x=self.x, y=self.y, z=self.z, phi=self.phi)

    @classmethod
    def from_file(cls, filename):
        data = np.load(filename)
        new_route = Route(
            xs=data['x'], ys=data['y'], zs=data['z'], phis=data['phi'],
            nant=data['ant'], nroute=data['route'])
        new_route.dt = data['dt']
        return new_route


def route_like(r, xs=None, ys=None, zs=None, phis=None, condition=None, nant=None, nroute=None):
    new_route = copy(r)
    if xs is not None:
        new_route.x = np.array(xs)
    if ys is not None:
        new_route.y = np.array(ys)
    if zs is not None:
        new_route.z = np.array(zs)
    if phis is not None:
        new_route.phi = np.array(phis)
    if condition is not None:
        new_route.condition = condition
    if nant is not None:
        new_route.nant = nant
    if nroute is not None:
        new_route.nroute = nroute
    return new_route
