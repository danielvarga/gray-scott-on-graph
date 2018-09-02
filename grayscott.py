import sys
import math, random
from collections import defaultdict
import numpy as np
import scipy.spatial
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/26127012#26127012
# TODO trivial to vectorize
def fibonacci_sphere(samples=1, randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

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


def test_tree_query_pairs(ps):
    s = 0
    for i, a in enumerate(ps):
        for j, b in enumerate(ps):
            if j<=i:
                continue
            if np.linalg.norm(a-b) < r:
                s += 1
    tree = scipy.spatial.cKDTree(ps)
    neis = tree.query_pairs(r)
    assert s == len(neis)


n = 50000
print "n", n
ps = fibonacci_sphere(samples=n, randomize=True)
# ps = random_sphere(n) # does not work, discrepancy too high
# ps = bunny() ; n = len(ps) # could be nicer, point cloud is not uniform density on surface


def find_hoods(edges):
    hoods = {}
    for a in range(n):
        hoods[a] = []
    for a, b in edges:
        hoods[a].append(b)
        hoods[b].append(a)
    return hoods


tree = scipy.spatial.cKDTree(ps)
# let's find the smallest radius that leads to a neighborhood graph with average degree > 6.
# too lazy to calculate it analytically for the Fibonacci sphere,
# too lazy to set radius on a per-point level for complex surfaces.
r = np.sqrt(1.0/n)
while True:
    edges = tree.query_pairs(r)
    hoods = find_hoods(edges)
    sizes = np.array(map(len, hoods.values()))
    avgdeg = sizes.mean()
    mindeg = sizes.min()
    maxdeg = sizes.max()
    print "radius", r, "average degree", avgdeg, "min", mindeg, "max", maxdeg
    if avgdeg >= 6:
        print "final radius", r
        break
    r *= 1.2


sizes = map(len, hoods.values())
print sizes[:20]
hist, bin_edges = np.histogram(sizes, bins=range(max(sizes)+1))
print hist, bin_edges


def sparse_laplacian(n, hoods):
    data = []
    row = []
    col = []
    for a, hood in hoods.iteritems():
        m = len(hood)
        if m > 0:
            data += [1.0 / m] * m
            row += [a] * m
            col += hood
    s = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
    # dealing with isolated vertices so that all-1 is still an eigenvector.
    s.setdiag([-1 if len(hoods[a])>0 else 0 for a in range(n)])
    return s


laplacian = sparse_laplacian(n, hoods)
assert np.allclose(laplacian.dot(np.ones(n)), np.zeros(n))


vis_fibonacci_polygon = False
if vis_fibonacci_polygon:
    segments = []
    for a, b in edges:
        if ps[a][2]>0 and ps[b][2]>0:
            segments.append([ps[a], ps[b]])
    segments = np.array(segments)
    print segments.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2])

    lc = Line3DCollection(segments, cmap=plt.get_cmap('copper'))
    lc.set_linewidth(2)
    ax.add_collection3d(lc)
    plt.show()
    sys.exit()


u = 0.2 * np.random.random(n) + 1
v = 0.2 * np.random.random(n)
z = np.array([u, v]).T
u = z[:, 0]
v = z[:, 1]

# *4 because the original formula summed rather than averaged.
Du, Dv, F, k = 0.16*4, 0.08*4, 0.060, 0.062 # Coral


fig = plt.figure(figsize=(10, 10), dpi=72.0, facecolor="white")
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c=v, cmap=plt.cm.gray_r)

n_iter = 5000
print "starting", n_iter, "iterations"
for i in range(n_iter):
    uvv = u*v*v
    Lu = laplacian.dot(u)
    Lv = laplacian.dot(v)
    u += (Du*Lu - uvv +  F   *(1-u))
    v += (Dv*Lv + uvv - (F+k)*v    )
    if i % 100 == 0:
        print i, v.mean()
        sys.stdout.flush()
        # ax.set_cdata(v)
        # fig.canvas.draw()

ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c=v, cmap=plt.cm.gray_r)
plt.show()

sys.exit()


from bokeh.plotting import figure, show, output_file
p = figure()
x = ps[:, 0][ps[:, 2]>0]
y = ps[:, 1][ps[:, 2]>0]
col = v[ps[:, 2]>0]
p.scatter(x, y, radius=np.ones(len(x))*0.02,
          fill_color=col, fill_alpha=0.6,
          line_color=None)
output_file("color_scatter.html", title="color_scatter.py example")
show(p)  # open a browser
