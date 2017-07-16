from __future__ import division
import mayavi.mlab as mlab
from CameraNetwork.visualization import calcSeaMask
import matplotlib.mlab as ml
import datetime
import glob
import json
import numpy as np
import os
import pandas as pd
import pymap3d


FLIGHT_PATH = r"data\2017_04_22_09_57_54_040000"
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


def loadMapData():
    """Load height data for map visualization."""

    path1 = r"..\data\reconstructions\N32E034.hgt"
    path2 = r"..\data\reconstructions\N32E035.hgt"
    with open(path1) as hgt_data:
        hgt1 = np.fromfile(hgt_data, np.dtype('>i2')).reshape((1201, 1201))[:1200, :1200]
    with open(path2) as hgt_data:
        hgt2 = np.fromfile(hgt_data, np.dtype('>i2')).reshape((1201, 1201))[:1200, :1200]
    hgt = np.hstack((hgt1, hgt2)).astype(np.float32)
    lon, lat = np.meshgrid(np.linspace(34, 36, 2400, endpoint=False), np.linspace(32, 33, 1200, endpoint=False)[::-1])

    x_slice = slice(100, 400)
    y_slice = slice(1100, 1400)

    return lat[x_slice, y_slice], lon[x_slice, y_slice], hgt[x_slice, y_slice]


def convertMapData(lat, lon, hgt, lat0=32.775776, lon0=35.024963, alt0=229):
    """Convert lat/lon/height data to grid data."""

    n, e, d = pymap3d.geodetic2ned(
        lat, lon, hgt,
        lat0=lat0, lon0=lon0, h0=alt0)

    x, y, z = e, n, -d

    xi = np.linspace(-10000, 10000, 300)
    yi = np.linspace(-10000, 10000, 300)
    X, Y = np.meshgrid(xi, yi)

    Z = ml.griddata(y.flatten(), x.flatten(), z.flatten(), yi, xi, interp='linear')
    Z_mask = calcSeaMask(Z)

    return X, Y, Z, Z_mask


def draw_map(mayavi_scene, map_coords):
    """Clear the map view and draw elevation map."""

    mlab.clf(figure=mayavi_scene)
    X, Y, Z, Z_mask = convertMapData(
        map_coords[0],
        map_coords[1],
        map_coords[2],
    )

    mlab.mesh(Y, X, MAP_ZSCALE * Z, figure=mayavi_scene, mask=Z_mask, color=(0.7, 0.7, 0.7))


@mlab.animate(delay=100)
def anim():
    f = mlab.gcf()
    while 1:
        f.scene.camera.azimuth(2)
        f.scene.render()
        yield


def main():
    #
    # Load the path data.
    #
    df, x, y, z = load_path()
    s = df[PARTICLE_SIZE].values

    #
    # Setup the map.
    #
    map_coords = loadMapData()
    mayavi_scene = mlab.figure(bgcolor=(1, 1, 1), size=(900, 900))
    draw_map(mayavi_scene, map_coords)

    #
    # Setup to the path.
    #
    s[s>5000] = 5000
    mask = (x>-10000) & (x<10000) & (y>-10000) & (y<10000)
    mlab.points3d(x[mask], y[mask], MAP_ZSCALE * z[mask], s[mask])

    a = anim()
    mlab.show()


if __name__ == "__main__":
    main()