"""PyQtGraph widget for the enaml framework

Based on the example by blink1073:
https://gist.github.com/blink1073/7411284
"""
from atom.api import Instance, Signal, Str, Int, observe, List, Bool
from enaml.qt import QtCore, QtGui
from enaml.widgets.api import *
import numpy as np
from pyqtgraph import opengl as gl
import pyqtgraph as pg


class PyQtGraphWidget(RawWidget):
    """A base for PyQtGraph Widgets for enaml."""
    
    def __init__(self, *params, **kwds):
        super(PyQtGraphWidget, self).__init__(*params, **kwds)
        
    def create_widget(self, parent):
        """ Create the PyQtGraph widget"""
        
        win = pg.GraphicsView(parent)

        return win


class PyQtGraphLayoutWidget(RawWidget):
    """A base for PyQtGraph Widgets for enaml."""
    
    def __init__(self, *params, **kwds):
        super(PyQtGraphLayoutWidget, self).__init__(*params, **kwds)
        
    def create_widget(self, parent):
        """ Create the PyQtGraph widget"""
        
        win = pg.GraphicsLayoutWidget(parent)

        return win


class PyQtImageView(PyQtGraphLayoutWidget):
    """A base for PyQtGraph Widgets for enaml."""
    
    img_array = Instance(np.ndarray)
    
    epipolar_scatter = Instance(pg.ScatterPlotItem)
    almucantar_scatter = Instance(pg.ScatterPlotItem)
    principalplane_scatter = Instance(pg.ScatterPlotItem)

    Almucantar_coords = List()
    PrincipalPlane_coords = List()
    
    roi = Instance(pg.RectROI)
    img_item = Instance(pg.ImageItem)
    
    epipolar_signal = Signal()
    epipolar_points = Int(100)
    server_id = Str()

    show_almucantar = Bool()
    show_principalplane = Bool()
    
    def __init__(self, *params, **kwds):
        super(PyQtImageView, self).__init__(*params, **kwds)

    def updateEpipolar(self, xs, ys):
        """Update the epipolar markers."""
        
        self.epipolar_scatter.setData(xs, ys)
        
    def getArrayRegion(self, data=None):
        """Get the region selected by ROI"""
        
        if data is None:
            data = self.img_array[..., 0].astype(np.float)
            
        return self.roi.getArrayRegion(data, self.img_item)
    
    def drawEpiploarPoints(self, plot_area):
        """Initialize the epipolar points on the plot."""
        
        self.epipolar_scatter = pg.ScatterPlotItem(
            size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 120)
        )
        self.epipolar_scatter.addPoints(
            pos=5*np.ones((self.epipolar_points, 2))
        )
        self.epipolar_scatter.setZValue(100)

        plot_area.addItem(self.epipolar_scatter)
        
    def drawAlmucantar(self, plot_area):
        """Initialize the Almucantar marker"""
        
        self.almucantar_scatter = pg.ScatterPlotItem(
            size=3, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120)
        )
        self.almucantar_scatter.addPoints(
            pos=np.array(self.Almucantar_coords)
        )
        self.almucantar_scatter.setZValue(99)

        plot_area.addItem(self.almucantar_scatter)
    
    def drawPrincipalPlane(self, plot_area):
        """Initialize the Principal Plane marker"""
        
        self.principalplane_scatter = pg.ScatterPlotItem(
            size=3, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120)
        )
        self.principalplane_scatter.addPoints(
            pos=np.array(self.PrincipalPlane_coords)
        )
        self.principalplane_scatter.setZValue(98)

        plot_area.addItem(self.principalplane_scatter)
    
    @observe('img_array')
    def update_img_array(self, change):
        """Update the image array."""
        
        if self.img_item is not None:
            self.img_item.setImage(change['value'].astype(np.float))

    @observe('Almucantar_coords')
    def update_Almucantar_coords(self, change):
        """Update the Almucantar coords."""
        
        if self.almucantar_scatter is not None:
            self.almucantar_scatter.setData(
                pos=np.array(change['value'])
            )
    
    @observe('show_almucantar')
    def showAlmucantar(self, change):
        """Control the visibility of the Almucantar widget."""
        
        if self.almucantar_scatter is not None:
            self.almucantar_scatter.setVisible(change['value'])
    
    @observe('PrincipalPlane_coords')
    def update_PrincipalPlane_coords(self, change):
        """Update the PrincipalPlane coords."""
        
        if self.principalplane_scatter is not None:
            self.principalplane_scatter.setData(
                pos=np.array(change['value'])
            )
    
    @observe('show_principalplane')
    def showPrincipalplane(self, change):
        """Control the visibility of the principalplane widget."""
        
        if self.principalplane_scatter is not None:
            self.principalplane_scatter.setVisible(change['value'])
    
    def create_widget(self, parent):
        """ Create the PyQtGraph widget"""
        
        win = super(PyQtImageView, self).create_widget(parent)
        
        plot_area = win.addPlot()
        plot_area.hideAxis('bottom')
        plot_area.hideAxis('left')
        
        #
        # Setup the epipolar scatters
        #
        self.drawEpiploarPoints(plot_area)
        
        #
        # Setup the Almucantar points
        #
        self.drawAlmucantar(plot_area)
        
        #
        # Setup the PrincipalPlane points
        #
        self.drawPrincipalPlane(plot_area)
        
        def mouseClicked(evt):

            pos = evt.scenePos()
            
            if plot_area.sceneBoundingRect().contains(pos):
                mp = plot_area.vb.mapSceneToView(pos)
                h, w = self.img_array.shape[:2]
                x, y = np.clip((mp.x(), mp.y()), 0, h-1).astype(np.int)
        
                self.updateEpipolar(
                    self.epipolar_points*[x],
                    self.epipolar_points*[y]
                )
                
                self.epipolar_signal.emit(
                    {'server_id': self.server_id, 'pos': (x, y)}
                )
                
        plot_area.scene().sigMouseClicked.connect(mouseClicked)
        
        #
        # Item for displaying image data
        #
        self.img_item = pg.ImageItem()
        plot_area.addItem(self.img_item)        
        
        self.img_item.setImage(self.img_array.astype(np.float))
        #plot_area.vb.invertY(True)

        #
        # Add ROI
        #
        self.roi = pg.RectROI([20, 20], [20, 20], pen=(0,9))
        self.roi.addRotateHandle([1, 0], [0.5, 0.5])
        plot_area.vb.addItem(self.roi)
        
        #
        # Create histogram
        #
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img_item)
        hist.gradient.hide()
        win.addItem(hist)
        
        win.resize(500, 400)
        
        return win

