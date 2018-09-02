import scipy.spatial
import numpy as np


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


def find_hoods(edges, n):
    hoods = {}
    for a in range(n):
        hoods[a] = []
    for a, b in edges:
        hoods[a].append(b)
        hoods[b].append(a)
    return hoods


def set_radius(kdtree):
    # let's find the smallest radius that leads to a neighborhood graph with average degree > 6.
    # too lazy to calculate it analytically for the Fibonacci sphere,
    # too lazy to set radius on a per-point level for complex surfaces.
    n = kdtree.n
    r = np.sqrt(1.0/n)
    for i in xrange(100):
        edges = kdtree.query_pairs(r)
        hoods = find_hoods(edges, n)
        sizes = np.array(map(len, hoods.values()))
        avgdeg = sizes.mean()
        mindeg = sizes.min()
        maxdeg = sizes.max()
        print "radius", r, "average degree", avgdeg, "min", mindeg, "max", maxdeg
        if avgdeg >= 6:
            print "final radius", r
            return r
        r *= 1.2
    assert False, "could not find proper radius"


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
