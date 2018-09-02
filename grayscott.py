import sys
from collections import defaultdict
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from pointclouds import *
from neighborhood import *


n = 50000
print "n", n
ps = fibonacci_sphere(samples=n, randomize=True)
# ps = random_sphere(n) # does not work, discrepancy too high
# ps = bunny() ; n = len(ps) # could be nicer, point cloud is not uniform density on surface


tree = scipy.spatial.cKDTree(ps)
r = set_radius(tree)
edges = tree.query_pairs(r)
hoods = find_hoods(edges, n)


sizes = map(len, hoods.values())
print sizes[:20]
hist, bin_edges = np.histogram(sizes, bins=range(max(sizes)+1))
print hist, bin_edges


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
