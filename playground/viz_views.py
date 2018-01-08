import cPickle
from mayavi import mlab
import numpy as np
import os

BASE_PATH = r'../scripts_client/reconstruction/2016_10_23_11_10_00/'


def quiver3(datas, length = 6000, skip=5):
    black = (0,0,0)
    white = (1,1,1)
    mlab.figure(bgcolor=white)
    for cam_id, data in datas.items():
        if cam_id == '107':
            continue
        
        x0, y0, z0, phi, psi = \
            [data[i] for i in ('x', 'y', 'z', 'bounding_phi', 'bounding_psi')]

        x = x0 + length * np.sin(phi)
        y = y0 + length * np.cos(phi)
        z = z0 + length * np.cos(psi)
        
        for x1, y1, z1 in zip(x, y, z):
            print(cam_id, x1, y1, z1)
            mlab.plot3d([x0, x1], [y0, y1], [z0, z1], color=black, tube_radius=10.)
            print(([x0, x1], [y0, y1], [z0, z1]))
        mlab.text3d(x0, y0, z0, cam_id, color=black, scale=100.)
    
    mlab.show()


def quiver4(datas, length = 6000, skip=5):
    black = (0,0,0)
    white = (1,1,1)
    mlab.figure(bgcolor=white)

    triangles = [
        (0, 1, 2),
        (0, 2, 4),
        (0, 4, 3),
        (0, 3, 1),
    ]
    
    for cam_id, data in datas.items():
        if cam_id == '107':
            continue
        
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
            opacity=0.4
        )
        mlab.text3d(x0, y0, z0, cam_id, color=black, scale=100.)
    
    mlab.show()


def main():

    with open(os.path.join(BASE_PATH, 'Datas.pkl'), 'rb') as f:
        datas = cPickle.load(f)   
    
    quiver4(datas)


if __name__ == '__main__':
    main()