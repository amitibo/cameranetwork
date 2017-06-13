##
## Copyright (C) 2017, Amit Aides, all rights reserved.
## 
## This file is part of Camera Network
## (see https://bitbucket.org/amitibo/cameranetwork_git).
## 
## Redistribution and use in source and binary forms, with or without modification,
## are permitted provided that the following conditions are met:
## 
## 1)  The software is provided under the terms of this license strictly for
##     academic, non-commercial, not-for-profit purposes.
## 2)  Redistributions of source code must retain the above copyright notice, this
##     list of conditions (license) and the following disclaimer.
## 3)  Redistributions in binary form must reproduce the above copyright notice,
##     this list of conditions (license) and the following disclaimer in the
##     documentation and/or other materials provided with the distribution.
## 4)  The name of the author may not be used to endorse or promote products derived
##     from this software without specific prior written permission.
## 5)  As this software depends on other libraries, the user must adhere to and keep
##     in place any licensing terms of those libraries.
## 6)  Any publications arising from the use of this software, including but not
##     limited to academic journal and conference publications, technical reports and
##     manuals, must cite the following works:
##     Dmitry Veikherman, Amit Aides, Yoav Y. Schechner and Aviad Levis, "Clouds in The Cloud" Proc. ACCV, pp. 659-674 (2014).
## 
## THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
## WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
## MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
## EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
## INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
## BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
## OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
## ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.##
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

        tm = mlab.triangular_mesh(
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
