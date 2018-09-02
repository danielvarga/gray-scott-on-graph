import math, random
import numpy as np


# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/26127012#26127012
# TODO trivial to vectorize
def fibonacci_sphere(n, randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * n

    points = []
    offset = 2./n
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(n):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % n) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return np.array(points)


def random_sphere(samples):
    a = np.random.normal(size=(samples, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


def bunny():
    ps = []
    with open("bunny_cloud.txt") as f:
        for l in f:
            try:
                a = map(float, l.split())
            except:
                print l
                raise
            xyz, confidence, luminosity = a[:3], a[3], a[4]
            ps.append(xyz)
    ps = np.array(ps)
    cog = ps.mean(axis=0, keepdims=True)
    ps -= cog
    print ps.shape
    print np.abs(ps).max()
    ps /= np.abs(ps).max()
    return ps
