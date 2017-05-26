"""PyQtGraph widget for the enaml framework

Based on the example by blink1073:
https://gist.github.com/blink1073/7411284
"""
from atom.api import Instance, Signal, Str, Int, observe, List, Bool, ForwardTyped, Typed, Tuple
from enaml.core.declarative import d_
from enaml.qt import QtCore, QtGui
from enaml.qt.qt_control import QtControl
from enaml.widgets.control import Control, ProxyControl
import math
import numpy as np
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder='row-major')

import CameraNetwork.global_settings as gs

MASK_INIT_RESOLUTION = 20
DEFAULT_IMG_SHAPE = (301, 301, 3)


class ProxyImageAnalysis(ProxyControl):
    #: A reference to the ImageAnalysis declaration.
    declaration = ForwardTyped(lambda: ImageAnalysis)

    def set_server_id(self, server_id):
        raise NotImplementedError

    def set_img_array(self, img_array):
        raise NotImplementedError

    def set_Almucantar_coords(self, almucantar_coords):
        raise NotImplementedError

    def set_PrincipalPlane_coords(self, principalplane_coords):
        raise NotImplementedError

    def set_Epipolar_coords(self, epipolar_coords):
        raise NotImplementedError

    def set_GRID_coords(self, grid_coords):
        raise NotImplementedError

    def set_show_almucantar(self, show):
        raise NotImplementedError

    def set_show_principalplane(self, show):
        raise NotImplementedError

    def set_show_grid(self, show):
        raise NotImplementedError

    def set_show_mask(self, show):
        raise NotImplementedError

    def set_show_ROI(self, show):
        raise NotImplementedError

    def set_gamma(self, apply):
        raise NotImplementedError

    def set_intensity(self, intensity):
        raise NotImplementedError


class ImageAnalysis(Control):
    """A base for PyQtGraph Widgets for enaml.

    It implement different widgets that help in analyzing the images.

    Attributes:
        img_array (array): Displayed image.
        epipolar_scatter (pg.ScatterPlotItem): Projection of selected pixel on
            different images.
        grid_scatter (pg.ScatterPlotItem): Projection of reconstruction grid on
            different images.
        almucantar_scatter (pg.ScatterPlotItem): Projection of the Almucantar on
            the image.
        principalplane_scatter (pg.ScatterPlotItem): Projection of the Principal
            Plane on the image.
        ROI (pg.RectROI): Rectangle ROI that is extracted for reconstruction.
        mask_ROI (pg.PolyLineROI): Polygonal ROI that is used for masking out
            obstructing objects.
    """

    #: A reference to the ProxyImageAnalysis object.
    proxy = Typed(ProxyImageAnalysis)

    #
    # The ID of the current server.
    #
    server_id = d_(Str())

    #
    # The displayed image as numpy array.
    #
    img_array = d_(Instance(np.ndarray))

    #
    # Coordinates of the Almucantar and PrinciplePlane and Epipolar visualizations.
    #
    Almucantar_coords = d_(List())
    PrincipalPlane_coords = d_(List())
    Epipolar_coords = d_(List())
    GRID_coords = d_(List())

    #
    # Flags that control the display
    #
    gamma = d_(Bool(False))
    intensity = d_(Int(100))
    show_grid = d_(Bool(False))
    show_mask = d_(Bool(False))
    show_ROI = d_(Bool(False))
    show_almucantar = Bool(False)
    show_principalplane = Bool(False)

    #
    # Signals to notify the main model of modifications
    # that need to be broadcast to the rest of the cameras.
    #
    LOS_signal = Signal()

    #--------------------------------------------------------------------------
    # Observers
    #--------------------------------------------------------------------------
    @observe('img_array', 'server_id', 'Almucantar_coords', 'PrincipalPlane_coords',
             'show_almucantar', 'show_principalplane', 'show_ROI', 'show_mask',
             'show_grid', 'gamma', 'intensity')
    def _update_proxy(self, change):
        """ Update the proxy widget when the Widget data changes.

        This method only updates the proxy when an attribute is updated;
        not when it is created or deleted.

        """
        # The superclass implementation is sufficient.
        super(ImageAnalysis, self)._update_proxy(change)


class QtImageAnalysis(QtControl, ProxyImageAnalysis):
    #: A reference to the widget created by the proxy.
    widget = Typed(pg.GraphicsLayoutWidget)

    server_id = Str()

    #
    # Different controls that can be displayed on the GUI
    #
    almucantar_scatter = Instance(pg.ScatterPlotItem)
    epipolar_scatter = Instance(pg.ScatterPlotItem)
    grid_scatter = Instance(pg.ScatterPlotItem)
    principalplane_scatter = Instance(pg.ScatterPlotItem)
    plot_area = Instance(pg.PlotItem)

    #
    # ROI - Rectangle ROI that can be set by the user.
    #
    ROI = Instance(pg.RectROI)
    ROIs_signal = Signal()

    #
    # mask_ROI - Polygon ROI that can be use to mask buildings and other
    # obstacles.
    #
    mask_ROI = Instance(pg.PolyLineROI)

    #
    # The image itself, a PyQtGraph ImageItem.
    #
    img_item = Instance(pg.ImageItem)

    def initEpiploarPoints(self, epipolar_coords):
        """Initialize the epipolar points on the plot."""

        self.epipolar_scatter = pg.ScatterPlotItem(
            size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 120)
        )
        self.epipolar_scatter.addPoints(
            pos=np.array(epipolar_coords)
        )
        self.epipolar_scatter.setZValue(100)

        self.plot_area.addItem(self.epipolar_scatter)

    def initGrid(self, grid_coords):
        """Initialize the grid (scatter points) on the plot."""

        self.grid_scatter = pg.ScatterPlotItem(
            size=1, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120)
        )
        self.grid_scatter.addPoints(
            pos=np.array(grid_coords)
        )
        self.grid_scatter.setZValue(100)

        self.plot_area.addItem(self.grid_scatter)
        self.grid_scatter.setVisible(False)

    def initAlmucantar(self, almucantar_coords):
        """Initialize the Almucantar marker"""

        self.almucantar_scatter = pg.ScatterPlotItem(
            size=3, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120)
        )
        self.almucantar_scatter.addPoints(
            pos=np.array(almucantar_coords)
        )
        self.almucantar_scatter.setZValue(99)
        self.almucantar_scatter.setVisible(False)

        self.plot_area.addItem(self.almucantar_scatter)

    def initPrincipalPlane(self, PrincipalPlane_coords):
        """Initialize the Principal Plane marker"""

        self.principalplane_scatter = pg.ScatterPlotItem(
            size=3, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120)
        )
        self.principalplane_scatter.addPoints(
            pos=np.array(PrincipalPlane_coords)
        )
        self.principalplane_scatter.setZValue(98)
        self.principalplane_scatter.setVisible(False)

        self.plot_area.addItem(self.principalplane_scatter)

    def initROIs(self, img_shape):
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

        self.plot_area.vb.addItem(self.mask_ROI)

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

        self.plot_area.vb.addItem(self.ROI)

    def mouseClicked(self, evt):
        """Callback of mouse click (used for updating the epipolar lines)."""
        #
        # Get the click position.
        #
        pos = evt.scenePos()

        if self.plot_area.sceneBoundingRect().contains(pos):
            #
            # Map the click to the image.
            #
            mp = self.plot_area.vb.mapSceneToView(pos)
            h, w = self.img_item.image.shape[:2]
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
            self.LOS_signal.emit(
                {'server_id': self.server_id, 'pos': (x, y)}
            )

    def create_widget(self):
        """Create the PyQtGraph widget"""

        self.widget = pg.GraphicsLayoutWidget(self.parent_widget())

        self.plot_area = self.widget.addPlot()
        self.plot_area.hideAxis('bottom')
        self.plot_area.hideAxis('left')

        #
        # Connect the click callback to the plot.
        #
        self.plot_area.scene().sigMouseClicked.connect(self.mouseClicked)

        #
        # Item for displaying image data
        #
        self.img_item = pg.ImageItem()
        self.plot_area.addItem(self.img_item)

        self.img_item.setImage(np.zeros(DEFAULT_IMG_SHAPE))

        #
        # Setup the ROIs
        #
        self.initROIs(DEFAULT_IMG_SHAPE)

        self.widget.resize(400, 400)

        return self.widget

    def init_widget(self):
        """ Initialize the widget.

        """
        super(QtImageAnalysis, self).init_widget()
        d = self.declaration
        self.set_server_id(d.server_id)
        self.set_img_array(d.img_array)
        self.set_show_almucantar(d.show_almucantar)
        self.set_show_principalplane(d.show_principalplane)
        self.set_show_grid(d.show_grid)
        self.set_show_mask(d.show_mask)
        self.set_show_ROI(d.show_ROI)
        self.set_gamma(d.gamma)
        self.set_intensity(d.intensity)

    def set_server_id(self, server_id):

        self.server_id = server_id

    def set_img_array(self, img_array):
        """Update the image array."""

        self.img_item.setImage(img_array.astype(np.float))

    def set_Almucantar_coords(self, Almucantar_coords):
        """Update the Almucantar coords."""

        if self.almucantar_scatter is None:
            self.initAlmucantar(Almucantar_coords)
            return

        self.almucantar_scatter.setData(
            pos=np.array(Almucantar_coords)
        )

    def set_PrincipalPlane_coords(self, PrincipalPlane_coords):
        """Update the Almucantar coords."""

        if self.principalplane_scatter is None:
            self.initPrincipalPlane(PrincipalPlane_coords)
            return

        self.principalplane_scatter.setData(
            pos=np.array(PrincipalPlane_coords)
        )

    def set_Epipolar_coords(self, Epipolar_coords):
        """Update the Epipolar coords."""

        if self.epipolar_scatter is None:
            self.initEpiploarPoints(Epipolar_coords)
            return

        self.epipolar_scatter.setData(
            pos=np.array(Epipolar_coords)
        )

    def set_GRID_coords(self, GRID_coords):
        """Update the grid coords."""

        if self.grid_scatter is None:
            self.initGrid(GRID_coords)
            return

        self.grid_scatter.setData(
            pos=np.array(GRID_coords)
        )

    def set_show_almucantar(self, show):
        """Control the visibility of the Almucantar widget."""

        if self.almucantar_scatter is None:
            return

        self.almucantar_scatter.setVisible(show)

    def set_show_principalplane(self, show):
        """Control the visibility of the PrincipalPlane widget."""

        if self.principalplane_scatter is None:
            return

        self.principalplane_scatter.setVisible(show)

    def set_show_grid(self, show):
        """Control the visibility of the grid widget."""

        if self.grid_scatter is None:
            return

        self.grid_scatter.setVisible(show)

    def set_show_mask(self, show):
        """Control the visibility of the mask ROI widget."""

        self.mask_ROI.setVisible(show)

    def set_show_ROI(self, show):
        """Control the visibility of the ROI widget."""

        self.ROI.setVisible(show)

    def set_gamma(self, apply_flag):
        """Apply Gamma correction."""

        if apply_flag:
            lut = np.array([((i / 255.0) ** 0.4) * 255
                            for i in np.arange(0, 256)]).astype(np.uint8)
        else:
            lut = np.arange(0, 256).astype(np.uint8)

        self.img_item.setLookupTable(lut)

    def set_intensity(self, intensity):
        """Set the intensity of the image."""

        self.img_item.setLevels((0, intensity))

    def getArrayRegion(self, data):
        """Get the region selected by ROI.

        The function accepts an array in the size of the image.
        It crops a region marked by the ROI.

        Args:
            data (array): The array to crop the ROI from.

        TODO:
            Fix the bug that create artifacts.
        """

        #
        # Get ROI region.
        #
        roi = self.ROI.getArrayRegion(data, self.img_item)

        return roi

    def _ROI_updated(self):
        """Callback of ROI udpate."""

        _, tr = self.ROI.getArraySlice(
            self.img_item.image,
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

        self.ROIs_signal.emit(
            {'server_id': self.server_id, 'pts': pts, 'shape': self.img_item.image.shape}
        )

    def update_ROI_resolution(self, old_shape):
        """Update the ROI_resolution.

        Used to fix the save ROI resolution to new array resolution.
        """

        s = float(self.img_item.image.shape[0]) / float(old_shape[0])
        c = np.array(old_shape)/2
        t = np.array(self.img_item.image.shape[:2])/2 -c
        self.ROI.scale(s, center=c)
        self.ROI.translate((t[0], t[1]))
        self.mask_ROI.scale(s, center=c)
        self.mask_ROI.translate((t[0], t[1]))


def image_analysis_factory():
    return QtImageAnalysis

