import cPickle
from mayavi import mlab
import matplotlib.mlab as ml
import numpy as np
import os
import pymap3d

BASE_PATH = r'../scripts_client/reconstruction/2016_10_23_11_10_00/'

def loadData():
    path1 = r"..\data\reconstructions\N32E034.hgt"
    path2 = r"..\data\reconstructions\N32E035.hgt"
    with open(path1) as hgt_data:
        hgt1 = np.fromfile(hgt_data, np.dtype('>i2')).reshape((1201, 1201))[:1200, :1200]
    with open(path2) as hgt_data:
        hgt2 = np.fromfile(hgt_data, np.dtype('>i2')).reshape((1201, 1201))[:1200, :1200]
    hgt = np.hstack((hgt1, hgt2)).astype(np.float32)
    lon, lat = np.meshgrid(np.linspace(34, 36, 2400, endpoint=False), np.linspace(32, 33, 1200, endpoint=False)[::-1])
    return lat[100:400, 1100:1400], lon[100:400, 1100:1400], hgt[100:400, 1100:1400]


def convert(lat, lon, hgt, lat0=32.775776, lon0=35.024963, alt0=229):
    n, e, d = pymap3d.geodetic2ned(
        lat, lon, hgt,
        lat0=lat0, lon0=lon0, h0=alt0)

    x, y, z = e, n, -d

    xi = np.linspace(-10000, 10000, 100)
    yi = np.linspace(-10000, 10000, 100)
    X, Y = np.meshgrid(xi, yi)

    Z = ml.griddata(y.flatten(), x.flatten(), z.flatten(), yi, xi, interp='linear')

    return X, Y, Z


def viz_elev(X, Y, Z):
    mlab.surf(Y, X, Z)


def quiver(datas, length = 6000, skip=5):

    triangles = [
        (0, 1, 2),
        (0, 2, 4),
        (0, 4, 3),
        (0, 3, 1),
    ]

    for cam_id, data in datas.items():
        x0, y0, z0, phi, psi = \
            [data[i] for i in ('x', 'y', 'z', 'bounding_phi', 'bounding_psi')]

        x = x0 + length * np.sin(phi)
        y = y0 + length * np.cos(phi)
        z = z0 + length * np.cos(psi)

        mlab.triangular_mesh(
            np.insert(x, 0, x0),
            np.insert(y, 0, y0),
            np.insert(z, 0, z0),
            triangles,
            color=(0.5, 0.5, 0.5),
            opacity=0.2
        )
        mlab.text3d(x0, y0, z0, cam_id, color=(0, 0, 0), scale=100.)


if __name__ == '__main__':
    lat, lon, hgt = loadData()
    X, Y, Z = convert(lat, lon, hgt)
    mlab.figure()
    viz_elev(X, Y, Z)

    with open(os.path.join(BASE_PATH, 'Datas.pkl'), 'rb') as f:
        datas = cPickle.load(f)

    quiver(datas)


    mlab.show()
