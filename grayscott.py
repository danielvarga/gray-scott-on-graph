import sys
import numpy as np

from pointclouds import *
from neighborhood import *
from vis import *

n = 1000
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


do_vis_neighborhood_polygon = True
if do_vis_neighborhood_polygon:
    vis_neighborhood_polygon(ps, edges)

u = 0.2 * np.random.random(n) + 1
v = 0.2 * np.random.random(n)


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
