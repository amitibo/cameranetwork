from __future__ import division
import cv2
import matplotlib.mlab as ml
import numpy as np
import pymap3d


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
    return lat[100:400, 1100:1400], lon[100:400, 1100:1400], hgt[100:400, 1100:1400]


def calcSeaMask(hgt_array):
    """Calc a masking to the sea.

    Note:
        This code is uses empirical magic number, and should be adjusted if
        grid sizes change.
    """

    hgt_u8 = (255 * (hgt_array - hgt_array.min()) / (hgt_array.max() - hgt_array.min())).astype(np.uint8)

    mask = (hgt_u8 > 7).astype(np.uint8)*255
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask[250:, 250:] = 255

    return mask < 255


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