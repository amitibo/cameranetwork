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
win.resize(800,400)

plot_area = win.addPlot()
plot_area.hideAxis('bottom')
plot_area.hideAxis('left')

img_item = pg.ImageItem()
plot_area.addItem(img_item)

plot_area1 = win.addPlot()
plot_area1.hideAxis('bottom')
plot_area1.hideAxis('left')

img_item1 = pg.ImageItem()
plot_area1.addItem(img_item1)

#
# Add ROI
#
ROI = pg.RectROI([20, 20], [20, 20], pen=(0,9))
ROI.addRotateHandle([1, 0], [0.5, 0.5])
ROI.setVisible(True)

#
# Callback when the user stopsmoving a ROI.
#
def ROI_updated(roi):
    crop = roi.getArrayRegion(normalized_img, img_item)
    print(crop.shape, crop.max(), crop.dtype)
    print(crop.shape, crop.max(), crop.dtype)
    img_item1.setImage(crop, levels=(0, normalized_img.max()))
    #v1b.autoRange()

ROI.sigRegionChangeFinished.connect(ROI_updated)
plot_area.vb.addItem(ROI)

win.show()

#
# Create the image.
#
fe = fisheye.load_model('data/.calibration_data.dat', calib_img_shape=(1200, 1600))
R = np.load('data/.extrinsic_data.npy')
nm = Normalization(resolution=301, fisheye_model=FisheyeProxy(fe), Rot=R)

img = sio.loadmat('data/1476876720.0_2016_10_19_11_32_00_2.mat')['img_array']
normalized_img = nm.normalize(img)
img_item.setImage(normalized_img)

#
# Callback of mouse click (used for updating the epipolar lines).
#
def mouseClicked(evt):

    #
    # Get the click position.
    #
    pos = evt.scenePos()

    if plot_area.sceneBoundingRect().contains(pos):
        #
        # Map the click to the image.
        #
        mp = plot_area.vb.mapSceneToView(pos)
        h, w = img.shape[:2]
        x, y = np.clip((mp.x(), mp.y()), 0, h-1).astype(np.int)

        print("x: {}, y: {}".format(x, y))
    else:
        print("Out of bounds.")

#
# Connect the click callback to the plot.
#
plot_area.scene().sigMouseClicked.connect(mouseClicked)



## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
