import numpy as np
import numpy.linalg as la
from datetime import datetime
from numbers import Number
from PIL import ImageDraw, Image
from colorsys import hsv_to_rgb
from utils import vec2sph

from sky import get_seville_observer, ChromaticitySkyModel
from compoundeye import CompoundEye

WIDTH = 1500
HEIGHT = 400
LENGTH = 1500

GRASS_COLOUR = (0, 255, 0)
GROUND_COLOUR = (229, 183, 90)
SKY_COLOUR = (13, 135, 201)


class World(object):

    def __init__(self, observer=None, polygons=None, width=WIDTH, length=LENGTH, height=HEIGHT):
        # normalise world
        xmax = np.array([polygons.x.max(), polygons.y.max(), polygons.z.max()]).max()
        polygons.normalise(*((xmax,) * 3))

        polygons = polygons * [.9 * width, .9 * length, .9 * height]
        polygons = polygons + [width / 2, length / 2, height / 2]

        if observer is None:
            observer = get_seville_observer()
        observer.date = datetime.now()
        self.sky = ChromaticitySkyModel(observer=observer, nside=1)
        self.sky.generate()
        thetas = np.linspace(-np.pi, np.pi, width, endpoint=False)
        phis = np.linspace(np.pi/2, 0, height, endpoint=False)
        thetas, phis = np.meshgrid(phis, thetas)
        ommatidia = np.array([thetas.flatten(), phis.flatten()]).T
        self.eye = CompoundEye(ommatidia)
        self.eye.set_sky(self.sky)

        self.polygons = polygons
        self.routes = []
        self.width = width
        self.length = length
        self.height = height
        self.__normalise_factor = xmax

    def add_route(self, route):
        route.normalise(*((self.__normalise_factor,) * 3))
        route = route * [.9 * self.width, .9 * self.length, .9 * self.height]
        route = route + [self.width / 2, self.length / 2, self.height / 2]
        self.routes.append(route)

    def draw_top_view(self):
        image = Image.new("RGB", (self.width, self.length), GROUND_COLOUR)
        draw = ImageDraw.Draw(image)

        for p in self.polygons:
            draw.polygon(p.xy, fill=p.c_int32)

        nants = np.float32(np.array([r.nant for r in self.routes]).max())
        nroutes = np.float32(np.array([r.nroute for r in self.routes]).max())
        for route in self.routes:
            h = np.float32(route.nant) / nants
            s = route.nroute / nroutes
            v = .5
            r, g, b = hsv_to_rgb(h, s, v)
            draw.line(route.xy, fill=(int(r * 255), int(g * 255), int(b * 255)))

        return image, draw

    def draw_panoramic_view(self, x=WIDTH/2, y=LENGTH/2, z=.06*HEIGHT, r=0):
        image = Image.new("RGB", (self.width, self.height * 2), GROUND_COLOUR)
        self.sky.obs.date = datetime.now()
        self.sky.generate()
        self.eye.facing_direction = r
        self.eye.set_sky(self.sky)
        draw = ImageDraw.Draw(image)

        pix = image.load()
        for i, c in enumerate(self.eye.L):
            pix[i // self.height, i % self.height] = tuple(np.int32(255 * c))

        pos = np.array([x, y, z])
        R = np.array([
            [np.cos(-r), -np.sin(-r), 0],
            [np.sin(-r), np.cos(-r), 0],
            [0, 0, 1]
        ])
        thetas, phis, rhos = [], [], []
        for p in self.polygons:
            theta, phi, rho = vec2sph((p.xyz - pos).dot(R))
            thetas.append(theta)
            phis.append(phi)
            rhos.append(rho)

        thetas = 2 * self.height * ((np.array(thetas) % np.pi) / np.pi)
        phis = self.width * ((np.pi + np.array(phis)) % (2 * np.pi)) / (2 * np.pi)
        rhos = la.norm(np.array(rhos), axis=-1)
        ind = np.argsort(rhos)[::-1]
        for theta, phi, c in zip(thetas[ind], phis[ind], self.polygons.c_int32[ind]):
            if phi.max() - phi.min() < WIDTH/2:  # normal conditions
                p = tuple((b, a) for a, b in zip(theta, phi))
                draw.polygon(p, fill=tuple(c))
            else:   # in case that the object is on the edge of the screen
                phi0, phi1 = phi.copy(), phi.copy()
                phi0[phi < WIDTH/2] += WIDTH
                phi1[phi >= WIDTH/2] -= WIDTH
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

    def __init__(self, xs, ys, zs=None, phis=None, nant=None, nroute=None):
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

    @property
    def xyz(self):
        return tuple((x, y, z) for x, y, z in zip(self.x, self.y, self.z))

    @property
    def xy(self):
        return tuple((x, y) for x, y in zip(self.x, self.y))

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
        return Route(self.x.copy(), self.y.copy(), self.z.copy(), self.phi.copy(), self.nant, self.nroute)

    def __str__(self):
        if self.nant > 0 and self.nroute > 0:
            s = "Ant: %02d, Route %02d: " % (self.nant, self.nroute)
        elif self.nant > 0 >= self.nroute:
            s = "Ant: %02d: " % self.nant
        elif self.nant <= 0 < self.nroute:
            s = "Route %02d: " % self.nroute
        else:
            s = "Route: "
        for x, y, z in self.xyz:
            s += "(%.2f, %.2f) " % (x, y)
        return s[:-1]
