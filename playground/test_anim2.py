from __future__ import division
import mayavi.mlab as mlab
from CameraNetwork.visualization import calcSeaMask
import matplotlib.mlab as ml
import datetime
import glob
import json
import math
import numpy as np
import os
import pandas as pd
import pymap3d



FLIGHT_PATH = r"..\particle_spectrometer\data\2017_04_22_09_57_54_040000"
MAP_ZSCALE = 3
PARTICLE_SIZE = 0.28

COLUMNS = [0.25, 0.28, 0.3, 0.35, 0.4, 0.45, 0.5, 0.58, 0.65, 0.7, 0.8, 1, 1.3, 1.6, 2, 2.5, 3, 3.5, 4, 5, 6.5, 7.5, 8.5, 10, 12.5, 15, 17.5, 20, 25, 30, 32]


def load_path(flight_path=FLIGHT_PATH, lat0=32.775776, lon0=35.024963, alt0=229):
    """Load the flight path."""

    file_paths = sorted(glob.glob(os.path.join(flight_path, '*.json')))

    data = []
    indices = []
    lat = []
    lon = []
    alt = []
    relative_alt = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            d = json.load(f)
            if len(d['data'])==0 or d['coords'] is None or d['coords']['lat'] == 0:
                #
                # ignore corrupt data.
                #
                continue

            t = datetime.datetime(*[int(i) for i in os.path.split(file_path)[-1].split('.')[0].split('_')])
            indices.append(t)
            data.append(d['data'])
            lat.append(d['coords']['lat']*1e-7)
            lon.append(d['coords']['lon']*1e-7)
            alt.append(d['coords']['alt']*1e-3)
            relative_alt.append(d['coords']['relative_alt']*1e-3)

    data = np.array(data)[..., :-1]
    df = pd.DataFrame(data=data, index=indices, columns=COLUMNS)

    #
    # Convert lat/lon/height data to grid data
    #
    n, e, d = pymap3d.geodetic2ned(
        lat, lon, alt,
        lat0=lat0, lon0=lon0, h0=alt0)

    x, y, z = e, n, -d


    return df, x, y, z


df, xs, ys, zs = load_path()
s = df[PARTICLE_SIZE].values.astype(np.float)

plt = mlab.points3d(xs, ys, zs, s)

index = 0

@mlab.animate(delay=1000)
def anim():
    f = mlab.gcf()
    while True:
        mask = np.zeros_like(xs, dtype=np.bool)
        global index
        index += 1
        mask[:index] = True
        print('Updating scene...')
        plt.mlab_source.reset(x=xs[mask], y=ys[mask], z=zs[mask], scalars=s[mask])
        yield

anim()
mlab.show()