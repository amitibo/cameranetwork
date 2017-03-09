# -*- coding: utf-8 -*-
"""
This example demonstrates the use of ImageView, which is a high-level widget for
displaying and analyzing 2D and 3D data. ImageView provides:

  1. A zoomable region (ViewBox) for displaying the image
  2. A combination histogram and gradient editor (HistogramLUTItem) for
     controlling the visual appearance of the image
  3. A timeline for selecting the currently displayed frame (for 3D data only).
  4. Tools for very basic analysis of image data (see ROI and Norm buttons)

"""
## Add path to library (just for examples; you do not need this)
#import initExample

from CameraNetwork.image_utils import Normalization
from CameraNetwork.image_utils import FisheyeProxy
import fisheye
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import scipy.io as sio

print pg.__file__

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

app = QtGui.QApplication([])

win = pg.GraphicsLayoutWidget()
win.resize(800,800)

plot_area = win.addPlot()
plot_area.hideAxis('bottom')
plot_area.hideAxis('left')

img_item = pg.ImageItem()
plot_area.addItem(img_item)

win.show()

fe = fisheye.load_model('data/.calibration_data.dat', calib_img_shape=(1200, 1600))
R = np.load('data/.extrinsic_data.npy')
nm = Normalization(resolution=301, fisheye_model=FisheyeProxy(fe), Rot=R)

img = sio.loadmat('data/1476876720.0_2016_10_19_11_32_00_2.mat')['img_array']
img_item.setImage(nm.normalize(img).astype(np.float))


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
