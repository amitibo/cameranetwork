from __future__ import division
import numpy as np
from numpy.polynomial import polynomial
from mayavi import mlab
import matplotlib.pyplot as plt
import cv2
import time
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp
from skimage.morphology import convex_hull_image
import fisheye
from skycameras import FisheyeProxy, Normalization, Radiometric
import cPickle
import os
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import math
import itertools
from tqdm import tqdm

COLORS = ('blue', 'green', 'red')
COLOR_INDICES = {'blue': 2, 'green': 1, 'red': 0}


def skFit(base_path, x, y, z, img_shape):
    model = make_pipeline(
        PolynomialFeatures(2),
        linear_model.RANSACRegressor(random_state=0, residual_threshold=5)
    )    

    #
    # Interpolate a second order polynomial
    #
    X = np.hstack([coord.reshape(-1, 1) for coord in (x, y)])
    model.fit(X, z)
    
    #
    # Visualize the error
    #
    ygrid, xgrid = np.mgrid[0:img_shape[0]:10, 0:img_shape[1]:10]
    grid = np.hstack([coord.reshape(-1, 1) for coord in (xgrid, ygrid)])
    
    zgrid = model.predict(grid).reshape(ygrid.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xgrid, ygrid, zgrid)
    ax.scatter(x, y, z)    


    z_estim = model.predict(X).reshape(y.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z-z_estim)

    plt.show()

    #
    # Visualize the error in the normalized image.
    #
    z_err = np.abs(z-z_estim)
    
    img_out = np.ones(shape=img_shape) * 100
    
    for threshold in np.logspace(-2, 2, 20)[::-1]:
        print threshold
        img_tmp = np.zeros(shape=img_shape)
        indices = z_err < threshold
        if not np.any(indices):
            break
        img_tmp[y[indices], x[indices]] = threshold
        chull = convex_hull_image(img_tmp)
        
        img_out[chull] = threshold
        
    plt.figure()
    plt.imshow(img_out)
    plt.colorbar()
    
    fe = fisheye.load_model(os.path.join(base_path, 'fisheye.pkl'))
    normalization = Normalization(1001, FisheyeProxy(fe))
    img_normalized = normalization.normalize(img_out)
    
    plt.figure()
    plt.imshow(img_normalized)
    plt.colorbar()
    plt.show()


def main(base_path):
    #
    # Load the measurements
    #
    base_path1 = r'vignetting_calibration\4102820388'
    
    color = 'blue'
    color_index = COLOR_INDICES[color]
    
    with open(os.path.join(base_path1, 'measurements_{}.pkl'.format(color)), 'rb') as f:
        measurements1 = cPickle.load(f)
    with open(os.path.join(base_path1, 'spec_{}.pkl'.format(color)), 'rb') as f:
        spec1 = cPickle.load(f)

    x1, y1, z1 = [np.array(a) for a in zip(*measurements1)]

    plt.figure()
    for c in COLORS:
        plt.plot(spec1[0], spec1[1], label='camera1')
        plt.legend()
    plt.show()
    
    for c in COLORS:
        mlab.figure(bgcolor=(1, 1, 1), )
        mlab.points3d(x1, y1, z1[..., COLOR_INDICES[c]], mode='sphere', scale_mode='none', scale_factor=5, color=(0, 0, 1))
        mlab.outline(color=(0, 0, 0), extent=(0, 1600, 0, 1200, 0, 255))
        mlab.title(c, color=(0, 0, 0))
    mlab.show()
    
    img_rgb = np.zeros(shape=(1200, 1600, 3))
    for x, y, val in measurements1:
        img_rgb[y, x, ...] = val
    
    img = img_rgb[..., 2][1::2, 1::2]
    y, x = np.nonzero(img)
    z = img[np.nonzero(img)]    
    
    skFit(base_path, x, y, z, img.shape)
