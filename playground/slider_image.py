from __future__ import division

from PyQt4 import QtCore
from PyQt4 import QtGui
#.QtCore import Qt, QRectF
#from PyQt4.QtGui import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    #QVBoxLayout, QWidget
#import QtCore.QString.fromUtf8 as asdf

import glob
import numpy as np
import os
import pandas as pd
import pyqtgraph as pg
import skimage.io as io
import sys


class Slider(QtGui.QWidget):
    def __init__(self, minimum, maximum, parent=None):
        super(Slider, self).__init__(parent=parent)

        #
        # Create a font for the label
        #
        font = QtGui.QFont()
        font.setFamily(QtCore.QString.fromUtf8("FreeMono"))
        font.setBold(True)

        #
        # Create the label
        #
        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.label = QtGui.QLabel(self)
        self.label.setFont(font)
        self.verticalLayout.addWidget(self.label)

        #
        # Create the Slider (centered)
        #
        self.horizontalLayout = QtGui.QHBoxLayout()
        spacerItem = QtGui.QSpacerItem(0, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.slider = QtGui.QSlider(self)
        self.slider.setOrientation(QtCore.Qt.Vertical)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QtGui.QSpacerItem(0, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.setLabelValue(self.slider.value())

    def setLabelValue(self, value):
        self.value = self.minimum + \
            float(value) / (self.slider.maximum() - self.slider.minimum()) * (self.maximum - self.minimum)
        self.label.setText("{0:.02g}".format(self.value))


class MainWindow(QtGui.QWidget):
    """main widget."""

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)

        #
        # Create the main window
        #
        self.win = pg.GraphicsWindow(title="Basic plotting examples")
        self.horizontalLayout = QtGui.QHBoxLayout(self)
        self.horizontalLayout.addWidget(self.win)
        self.view = self.win.addViewBox()

        #
        # lock the aspect ratio so pixels are always square
        #
        self.view.setAspectLocked(True)

        #
        # Load the thumbnails dataframes
        #
        dfs = pd.read_pickle(r"..\ipython\system\thumbnails_downloaded.pkl")
        self.thumbs = {}
        self.image_items = {}
        server_id_list, df_list = [], []
        for server_id, df in dfs.items():
            server_id_list.append(server_id)

            #
            # Load all the images.
            #
            print("Processing camera {}".format(server_id))
            images, indices = [], []
            index = 0
            for _, row in df.iterrows():
                try:
                    images.append(io.imread(os.path.join(r"..\ipython\system", row["thumbnail"])))
                    indices.append(index)
                    index += 1
                except:
                    images.append(None)
                    indices.append(None)

            self.thumbs[server_id] = images
            df["thumb_index"] = indices
            df_list.append(df)

            #
            # Create image widgets
            #
            image_item = pg.ImageItem()
            self.view.addItem(image_item)
            self.image_items[server_id] = image_item

        self.df = pd.concat(df_list, axis=1, keys=server_id_list)

        #
        # Create the thumbnail slider
        #
        self.w1 = Slider(0, len(self.df)-1)
        self.horizontalLayout.addWidget(self.w1)
        self.w1.slider.valueChanged.connect(lambda: self.update())

        self.update()

    def update(self):
        #
        # Get the current image time/index.
        #
        img_index = int(self.w1.value)
        row = self.df.iloc[img_index]

        coord = 0
        for server_id, image_item in self.image_items.items():
            coord += 50
            server_data = row[server_id]
            if not np.isfinite(server_data["thumb_index"]):
                image_item.hide()
                continue

            image_item.show()
            image_item.setImage(self.thumbs[server_id][int(server_data["thumb_index"])])
            image_item.setRect(QtCore.QRectF(coord, coord, 100, 100))


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())