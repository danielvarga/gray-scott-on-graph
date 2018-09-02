import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Line3DCollection


def vis_neighborhood_polygon(ps, edges):
    segments = []
    for a, b in edges:
        if ps[a][2]>0 and ps[b][2]>0:
            segments.append([ps[a], ps[b]])
    segments = np.array(segments)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], s=1)

    lc = Line3DCollection(segments, cmap=plt.get_cmap('copper'))
    lc.set_linewidth(1)
    ax.add_collection3d(lc)
    plt.show()


def vis_pattern(ps, v):
    fig = plt.figure(figsize=(10, 10), dpi=72.0, facecolor="white")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c=v, cmap=plt.cm.gray_r)
    plt.show()


def save_frame(prefix, i, ps, values, rotation=0):
    c, s = np.cos(rotation), np.sin(rotation)
    j = np.matrix([[c, s], [-s, c]])
    ps_rotated = ps.copy()
    ps_rotated[:, [0, 2]] = np.dot(j, ps_rotated[:, [0, 2]].T).T
    perm = np.argsort(ps_rotated[:, 2])
    ps_rotated = ps_rotated[perm, :]
    values_rotated = values[perm]

    from PIL import Image, ImageDraw
    s = 500
    r = 2
    img = Image.new("RGB", (s, s), "lightblue")
    draw = ImageDraw.Draw(img)
    filename = "%s-%05d.png" % (prefix, i)
    for (x, y, z), value in zip(ps_rotated, values_rotated):
        br = int(value*255)
        br = np.clip(br, 0, 255)
        x = (x+1)/2*s
        y = (-y+1)/2*s # draw has larger ys down, unlike bunny point cloud
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(br, br, br))
    img.save(filename)


def bokeh_vis_broken(ps, v):
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
