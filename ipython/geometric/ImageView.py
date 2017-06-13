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
