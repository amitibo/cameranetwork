from __future__ import absolute_import, division, print_function
from mayavi import mlab
import numpy as np
import math

alpha = np.linspace(0, 2*math.pi, 100)

xs = np.cos(alpha)
ys = np.sin(alpha)
zs = np.zeros_like(xs)
s = np.random.rand(xs.size) + 1
sz = np.zeros_like(s)

plt = mlab.points3d(xs, ys, zs, sz, colormap="copper", scale_factor=.50)

index = 20

@mlab.animate(delay=1000)
def anim():
    f = mlab.gcf()
    while True:
        mask = np.zeros_like(xs, dtype=np.bool)
        global index
        index += 1
        mask[:index] = True
        print('Updating scene...')
        sz = np.zeros_like(s)
        sz[mask] = s[mask]
        plt.mlab_source.scalars=sz
        yield

anim()
mlab.show()