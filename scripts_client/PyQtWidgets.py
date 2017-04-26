"""PyQtGraph widget for the enaml framework

Based on the example by blink1073:
https://gist.github.com/blink1073/7411284
"""
from atom.api import Instance, Signal, Str, Int, observe, List, Bool
from enaml.qt import QtCore, QtGui
from enaml.widgets.api import *
import math
import numpy as np
from pyqtgraph import opengl as gl
import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder='row-major')


MASK_INIT_RESOLUTION = 20


class PyQtGraphLayoutWidget(RawWidget):
    """A base for PyQtGraph Widgets for enaml."""

    def __init__(self, *params, **kwds):
        super(PyQtGraphLayoutWidget, self).__init__(*params, **kwds)

    def create_widget(self, parent):
        """ Create the PyQtGraph widget"""

        win = pg.GraphicsLayoutWidget(parent)

        return win


class PyQtImageView(PyQtGraphLayoutWidget):
    """A base for PyQtGraph Widgets for enaml.

    It implement different widgets that help in analyzing the images.

    Attributes:
        img_array (array): Displayed image.
        epipolar_scatter (pg.ScatterPlotItem): Projection of selected pixel on
            different images.
        LIDAR_grid_scatter (pg.ScatterPlotItem): Projection of LIDAR grid on
            different images.
        almucantar_scatter (pg.ScatterPlotItem): Projection of the Almucantar on
            the image.
        principalplane_scatter (pg.ScatterPlotItem): Projection of the Principal
            Plane on the image.
        ROI (pg.RectROI): Rectangle ROI that is extracted for reconstruction.
        mask_ROI (pg.PolyLineROI): Polygonal ROI that is used for masking out
            obstructing objects.
    """

    #
    # The displayed image.
    #
    img_array = Instance(np.ndarray)

    #
    # Mask array.
    # This is used for removing the sunshader pixels.
    #
    mask_array = Instance(np.ndarray)

    #
    # Different lines that can be displayed on the GUI
    #
    epipolar_scatter = Instance(pg.ScatterPlotItem)
    LIDAR_grid_scatter = Instance(pg.ScatterPlotItem)
    almucantar_scatter = Instance(pg.ScatterPlotItem)
    principalplane_scatter = Instance(pg.ScatterPlotItem)

    Almucantar_coords = List()
    PrincipalPlane_coords = List()

    #
    # ROI - Rectangle ROI that can be set by the user.
    #
    ROI = Instance(pg.RectROI)
    ROI_signal = Signal()

    #
    # mask_ROI - Polygon ROI that can be use to mask buildings and other
    # obstacles.
    #
    mask_ROI = Instance(pg.PolyLineROI)

    img_item = Instance(pg.ImageItem)

    epipolar_signal = Signal()
    epipolar_points = Int(100)
    LIDAR_grid_points = Int(1000)
    server_id = Str()

    show_almucantar = Bool()
    show_principalplane = Bool()
    show_ROI = Bool()
    show_mask = Bool()
    show_LIDAR_grid = Bool()

    export_flag = Bool()

    def __init__(self, *params, **kwds):
        super(PyQtImageView, self).__init__(*params, **kwds)

    def updateEpipolar(self, xs, ys):
        """Update the epipolar markers."""

        self.epipolar_scatter.setData(xs, ys)

    def updateLIDARgridPts(self, xs, ys):
        """Update the LIDAR grid markers."""

        self.LIDAR_grid_scatter.setData(xs, ys)

    def getArrayRegion(self, data=None):
        """Get the region selected by ROI.

        The function accepts an array in the size of the image.
        It crops a region marked by the ROI.

        Args:
            data (array): The array to crop the ROI from. If None
                the image will be croped.
        """

        if data is None:
            data = self.img_array[..., 0].astype(np.float)

        #
        # Get ROI region.
        #
        roi = self.ROI.getArrayRegion(data, self.img_item)

        return roi

    def getMask(self):
        """Get the mask as set by mask_ROI.
        """

        data = np.ones(self.img_array.shape[:2], np.uint8)

        #
        # Get ROI region.
        #
        sl, _ = self.mask_ROI.getArraySlice(data, self.img_item, axes=(0, 1))
        sl_mask = self.mask_ROI.getArrayRegion(data, self.img_item)

        #
        # The new version of pyqtgraph has some rounding problems.
        # Fix it the slices accordingly.
        #
        fixed_slices = (
            slice(sl[0].start, sl[0].start+sl_mask.shape[0]),
            slice(sl[1].start, sl[1].start+sl_mask.shape[1])
        )

        mask = np.zeros(self.img_array.shape[:2], np.uint8)
        mask[fixed_slices] = sl_mask

        return mask

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

    def drawLIDARgrid(self, plot_area):
        """Initialize the LIDAR grid (scatter points) on the plot."""

        self.LIDAR_grid_scatter = pg.ScatterPlotItem(
            size=1, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120)
        )
        self.LIDAR_grid_scatter.addPoints(
            pos=5*np.ones((self.LIDAR_grid_points, 2))
        )
        self.LIDAR_grid_scatter.setZValue(100)

        plot_area.addItem(self.LIDAR_grid_scatter)
        self.LIDAR_grid_scatter.setVisible(False)

    def drawAlmucantar(self, plot_area):
        """Initialize the Almucantar marker"""

        self.almucantar_scatter = pg.ScatterPlotItem(
            size=3, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120)
        )
        self.almucantar_scatter.addPoints(
            pos=np.array(self.Almucantar_coords)
        )
        self.almucantar_scatter.setZValue(99)
        self.almucantar_scatter.setVisible(False)

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
        self.principalplane_scatter.setVisible(False)

        plot_area.addItem(self.principalplane_scatter)

    def drawROIs(self, plot_area, img_shape):
        """Initialize the ROI markers"""

        #
        # Mask ROI
        #
        angles = np.linspace(0, 2*np.pi, MASK_INIT_RESOLUTION)
        xs = img_shape[0] * (1 + 0.9 * np.cos(angles)) / 2
        ys = img_shape[1] * (1 + 0.9 * np.sin(angles)) / 2
        mask_positions = np.vstack((xs, ys)).T
        self.mask_ROI = pg.PolyLineROI(mask_positions, closed=True)
        self.mask_ROI.setVisible(False)

        plot_area.vb.addItem(self.mask_ROI)

        #
        # Reconstruction ROI
        #
        self.ROI = pg.RectROI([20, 20], [20, 20], pen=(0,9))
        self.ROI.addRotateHandle([1, 0], [0.5, 0.5])
        self.ROI.setVisible(False)

        #
        # Callback when the user stopsmoving a ROI.
        #
        self.ROI.sigRegionChangeFinished.connect(self._ROI_updated)

        plot_area.vb.addItem(self.ROI)

    def _calcMask(self):
        """Calculate the sunshader mask.

        .. note::
            Currently not implemented.
        """

        pass

    def _ROI_updated(self):
        """Callback of ROI udpate."""

        _, tr = self.ROI.getArraySlice(
            self.img_array,
            self.img_item
        )

        size = self.ROI.state['size']

        #
        # Calculate the bounds.
        #
        pts = np.array(
            [tr.map(x, y) for x, y in \
             ((0, 0), (size.x(), 0), (0, size.y()), (size.x(), size.y()))]
        )

        self.ROI_signal.emit(
            {'server_id': self.server_id, 'pts': pts, 'shape': self.img_array.shape}
        )

    def update_ROI_resolution(self, old_shape):
        """Update the ROI_resolution.

        Used to fix the save ROI resolution to new array resolution.
        """

        s = float(self.img_array.shape[0]) / float(old_shape[0])
        c = np.array(old_shape)/2
        t = np.array(self.img_array)/2 -c
        self.ROI.scale(s, center=c)
        self.ROI.translate((t[0], t[1]))
        self.mask_ROI.scale(s, center=c)
        self.mask_ROI.translate((t[0], t[1]))

    @observe('img_array')
    def update_img_array(self, change):
        """Update the image array."""

        if self.img_item is not None:
            self.img_item.setImage(change['value'].astype(np.float))

        self._calcMask()

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

    @observe('show_ROI')
    def showROI(self, change):
        """Control the visibility of the ROI widget."""

        if self.ROI is not None:
            self.ROI.setVisible(change['value'])

    @observe('show_LIDAR_grid')
    def showLIDARgrid(self, change):
        """Control the visibility of the LIDAR grid widget."""

        self.LIDAR_grid_scatter.setVisible(change['value'])

    @observe('show_mask')
    def showMaskROI(self, change):
        """Control the visibility of the mask ROI widget."""

        if self.mask_ROI is not None:
            self.mask_ROI.setVisible(change['value'])

    def applyGamma(self, apply_flag):
        """Apply Gamma correction."""

        if apply_flag:
            lut = np.array([((i / 255.0) ** 0.4) * 255
                            for i in np.arange(0, 256)]).astype(np.uint8)
        else:
            lut = np.arange(0, 256).astype(np.uint8)

        self.img_item.setLookupTable(lut)

    def setIntensity(self, intensity):
        """Set the intensity of the image."""

        self.img_item.setLevels((0, intensity))


    def create_widget(self, parent):
        """Create the PyQtGraph widget"""

        win = super(PyQtImageView, self).create_widget(parent)

        plot_area = win.addPlot()
        plot_area.hideAxis('bottom')
        plot_area.hideAxis('left')

        #
        # Setup the epipolar scatters
        #
        self.drawEpiploarPoints(plot_area)

        #
        # Setup the LIDAR grid.
        #
        self.drawLIDARgrid(plot_area)

        #
        # Setup the Almucantar points
        #
        self.drawAlmucantar(plot_area)

        #
        # Setup the PrincipalPlane points
        #
        self.drawPrincipalPlane(plot_area)

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
                h, w = self.img_array.shape[:2]
                x, y = np.clip((mp.x(), mp.y()), 0, h-1).astype(np.int)

                #
                # Update the epiploar line of this view.
                #
                self.updateEpipolar(
                    self.epipolar_points*[x],
                    self.epipolar_points*[y]
                )

                #
                # Update the epipolar line of all other views.
                #
                self.epipolar_signal.emit(
                    {'server_id': self.server_id, 'pos': (x, y)}
                )

        #
        # Connect the click callback to the plot.
        #
        plot_area.scene().sigMouseClicked.connect(mouseClicked)

        #
        # Item for displaying image data
        #
        self.img_item = pg.ImageItem()
        plot_area.addItem(self.img_item)

        self.img_item.setImage(self.img_array.astype(np.float))
        #plot_area.vb.invertY(True)

        #
        # Setup the ROIs
        #
        self.drawROIs(plot_area, self.img_array.shape)

        win.resize(400, 400)

        return win


    #myFilter = MyEventFilter()
    #win.installEventFilter(myFilter)

#class MyEventFilter(QtCore.QObject):
    #def eventFilter(self, event):
        #if event.type() == QtCore.QEvent.Wheel:
            ## do some stuff ...
            #return False # means stop event propagation

        #return super(MyEventFilter,self).eventFilter(event)


