"""PyQtGraph widget for the enaml framework

Based on the example by blink1073:
https://gist.github.com/blink1073/7411284
"""
from atom.api import Instance, Signal, Str, Int, observe, List, Bool
from enaml.core.declarative import d_
from enaml.qt import QtCore, QtGui
from enaml.qt.qt_control import QtControl
from enaml.widgets.control import Control, ProxyControl
import math
import numpy as np
import pyqtgraph as pg
from pyqtgraph import opengl as gl

pg.setConfigOptions(imageAxisOrder='row-major')


MASK_INIT_RESOLUTION = 20


class QtImageAnalysis(QtControl, ProxyImageView):
    #: A reference to the widget created by the proxy.
    widget = Typed(pg.GraphicsLayoutWidget)

    #
    # Different controls that can be displayed on the GUI
    #
    epipolar_scatter = Instance(pg.ScatterPlotItem)
    grid_scatter = Instance(pg.ScatterPlotItem)
    almucantar_scatter = Instance(pg.ScatterPlotItem)
    principalplane_scatter = Instance(pg.ScatterPlotItem)

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

    def drawGrid(self, plot_area):
        """Initialize the grid (scatter points) on the plot."""

        self.grid_scatter = pg.ScatterPlotItem(
            size=1, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120)
        )
        self.grid_scatter.addPoints(
            pos=5*np.ones((self.grid_points, 2))
        )
        self.grid_scatter.setZValue(100)

        plot_area.addItem(self.grid_scatter)
        self.grid_scatter.setVisible(False)

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

    def create_widget(self, parent):
        """Create the PyQtGraph widget"""

        self.widget = pg.GraphicsLayoutWidget(parent)

        plot_area = self.widget.addPlot()
        plot_area.hideAxis('bottom')
        plot_area.hideAxis('left')

        #
        # Setup the epipolar scatters
        #
        self.drawEpiploarPoints(plot_area)

        #
        # Setup the grid.
        #
        self.drawGrid(plot_area)

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
                self.LOS_signal.emit(
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

        self.widget.resize(400, 400)

        return self.widget

    def set_img_array(self, img_array):
        """Update the image array."""

        self.img_item.setImage(img_array.astype(np.float))

    def set_Almucantar_coords(self, Almucantar_coords):
        """Update the Almucantar coords."""

        self.almucantar_scatter.setData(
            pos=np.array(Almucantar_coords)
        )

    @observe('show_almucantar')
    def _show_almucantar(self, change):
        """Control the visibility of the Almucantar widget."""

        if self.almucantar_scatter is not None:
            self.almucantar_scatter.setVisible(change['value'])

    @observe('PrincipalPlane_coords')
    def _update_PrincipalPlane_coords(self, change):
        """Update the PrincipalPlane coords."""

        if self.principalplane_scatter is not None:
            self.principalplane_scatter.setData(
                pos=np.array(change['value'])
            )

    @observe('show_principalplane')
    def _show_principalplane(self, change):
        """Control the visibility of the principalplane widget."""

        if self.principalplane_scatter is not None:
            self.principalplane_scatter.setVisible(change['value'])

    @observe('show_ROI')
    def _show_ROI(self, change):
        """Control the visibility of the ROI widget."""

        print "Works :-)"
        if self.ROI is not None:
            self.ROI.setVisible(change['value'])

    @observe('show_grid')
    def show_grid(self, change):
        """Control the visibility of the grid widget."""

        self.grid_scatter.setVisible(change['value'])

    @observe('show_mask')
    def showMaskROI(self, change):
        """Control the visibility of the mask ROI widget."""

        if self.mask_ROI is not None:
            self.mask_ROI.setVisible(change['value'])

    @observe('intensity')
    def setIntensity(self, change):
        """Set the intensity of the image."""

        self.img_item.setLevels((0, change['value']))


class ProxyImageAnalysis(ProxyControl):
    #: A reference to the ImageView declaration.
    declaration = ForwardTyped(lambda: ImageView)



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

    #
    # The ID of the current server.
    #
    server_id = d_(Str())

    #
    # The displayed image as numpy array.
    #
    img_array = d_(Instance(np.ndarray))

    Almucantar_coords = d_(List())
    PrincipalPlane_coords = d_(List())

    #
    # Signals to notify the main model of modifications
    # that need to be broadcast to the rest of the cameras.
    #
    LOS_signal = Signal()
    epipolar_points = Int(100)

    #
    # Flags that control the display
    #
    show_almucantar = Bool(False)
    show_principalplane = Bool(False)
    show_ROI = d_(Bool(False))
    show_mask = d_(Bool(False))
    show_grid = d_(Bool(False))
    gamma = d_(Bool(False))
    intensity = d_(Int(40))

    #
    # Extra...
    #
    grid_points = Int(1000)

    #
    # Whether use this view in the exported (reconstruction) data.
    #
    export_flag = Bool()

    def updateEpipolar(self, xs, ys):
        """Update the epipolar markers."""

        self.epipolar_scatter.setData(xs, ys)

    def updateGridPts(self, xs, ys):
        """Update the Grid markers."""

        self.grid_scatter.setData(xs, ys)

    def getArrayRegion(self, data=None):
        """Get the region selected by ROI.

        The function accepts an array in the size of the image.
        It crops a region marked by the ROI.

        Args:
            data (array): The array to crop the ROI from. If None
                the image will be croped.

        TODO:
            Fix the bug that create artifacts.
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

        self.ROIs_signal.emit(
            {'server_id': self.server_id, 'pts': pts, 'shape': self.img_array.shape}
        )

    def update_ROI_resolution(self, old_shape):
        """Update the ROI_resolution.

        Used to fix the save ROI resolution to new array resolution.
        """

        s = float(self.img_array.shape[0]) / float(old_shape[0])
        c = np.array(old_shape)/2
        t = np.array(self.img_array.shape[:2])/2 -c
        self.ROI.scale(s, center=c)
        self.ROI.translate((t[0], t[1]))
        self.mask_ROI.scale(s, center=c)
        self.mask_ROI.translate((t[0], t[1]))

    def applyGamma(self, apply_flag):
        """Apply Gamma correction."""

        if apply_flag:
            lut = np.array([((i / 255.0) ** 0.4) * 255
                            for i in np.arange(0, 256)]).astype(np.uint8)
        else:
            lut = np.arange(0, 256).astype(np.uint8)

        self.img_item.setLookupTable(lut)


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
        super(PyQtImageView, self)._update_proxy(change)


def image_analysis_factory():
    return ImageAnalysis

