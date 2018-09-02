import sys
import numpy as np

from pointclouds import *
from neighborhood import *
from vis import *


n = 5000
print "n", n
ps = fibonacci_sphere(n, randomize=True)
# ps = random_sphere(n) # does not work, discrepancy too high
# ps = bunny() ; n = len(ps) # could be nicer, point cloud is not uniform density on surface


tree = scipy.spatial.cKDTree(ps)
r = optimize_radius(tree)
edges = tree.query_pairs(r)
hoods = find_hoods(edges, n)


sizes = map(len, hoods.values())
print "example graph degrees", sizes[:20]
hist, bin_edges = np.histogram(sizes, bins=range(max(sizes)+1))
print "graph degree histogram", hist, bin_edges


laplacian = sparse_laplacian(n, hoods)
assert np.allclose(laplacian.dot(np.ones(n)), np.zeros(n))


do_vis_neighborhood_polygon = (n<=1000)
if do_vis_neighborhood_polygon:
    vis_neighborhood_polygon(ps, edges)


# *4 because the original formula summed rather than averaged.
Du, Dv, F, k = 0.16*4, 0.08*4, 0.060, 0.062 # Coral

def gray_scott_update(u, v, laplacian):
    uvv = u*v*v
    Lu = laplacian.dot(u)
    Lv = laplacian.dot(v)
    u += (Du*Lu - uvv +  F   *(1-u))
    v += (Dv*Lv + uvv - (F+k)*v    )


u = 0.2 * np.random.random(n) + 1
v = 0.2 * np.random.random(n)

import matplotlib as mpl
plt.ion()
fig = plt.figure(figsize=(10, 10), dpi=72.0, facecolor="white")
ax = fig.add_subplot(111)
ax.set_xlim(-1, +1)
ax.set_ylim(-1, +1)
perm = np.argsort(ps[:, 2])
scatter = ax.scatter(ps[perm, 0], ps[perm, 1], c=v[perm], cmap=plt.cm.gray_r)
plt.draw()


n_iter = 3000
print "starting", n_iter, "iterations"
for i in range(n_iter):
    gray_scott_update(u, v, laplacian)
    if (i+1) % 100 == 0:
        print i+1, v.mean()
        sys.stdout.flush()

        values = v[perm]
        n = mpl.colors.Normalize(vmin = min(values), vmax = max(values))
        m = mpl.cm.ScalarMappable(norm=n, cmap=plt.cm.gray_r)
        scatter.set_facecolor(m.to_rgba(values))
        scatter.set_clim(vmin=min(values), vmax=max(values))
        fig.canvas.show()


plt.show()

# vis_pattern(ps, v)
