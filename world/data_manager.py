import os
from scipy.io import loadmat
from model import *

__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = __dir__ + "/../data/"
__seville_2009__ = __data__ + "Seville2009_world/"
WORLD_FILENAME = "world5000_gray.mat"
ROUTES_FILENAME = "AntRoutes.mat"


def load_world(world_filename=WORLD_FILENAME, width=WIDTH, length=LENGTH, height=HEIGHT):
    mat = loadmat(__seville_2009__ + world_filename)
    polygons = PolygonList()
    for xs, ys, zs, col in zip(mat["X"], mat["Y"], mat["Z"], mat["colp"]):
        col[0] = col[2] = 0
        polygons.append(Polygon(xs, ys, zs, col))

    return World(polygons=polygons, width=width, length=length, height=height)


def load_routes(routes_filename=ROUTES_FILENAME):
    mat = loadmat(__seville_2009__ + routes_filename)
    ant, route, key = 1, 1, lambda a, r: "Ant%d_Route%d" % (a, r)
    routes = []
    while key(ant, route) in mat.keys():
        while key(ant, route) in mat.keys():
            mat[key(ant, route)][:, :2] /= 100.  # convert the route data to meters
            xs, ys, phis = mat[key(ant, route)].T
            r = Route(xs, ys, .01, phis=np.deg2rad(phis), nant=ant, nroute=route)
            routes.append(r)
            route += 1
        ant += 1
        route = 1
    return routes


if __name__ == "__main__":
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, 2*HEIGHT))
    done = False

    world = load_world()
    routes = load_routes()
    for route in routes:
        world.add_route(route)
        break

    for xyz, phi in zip(world.routes[-1].xyz, world.routes[-1].phi):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        img, draw = world.draw_panoramic_view(xyz[0], xyz[1], xyz[2], phi)
        # img.show()
        screen.blit(pygame.image.fromstring(img.tobytes("raw", "RGB"), img.size, "RGB"), (0, 0))
        pygame.display.flip()

        if done:
            break

    # img, draw = world.draw_top_view()
