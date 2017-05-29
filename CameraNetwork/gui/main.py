"""Run a GUI Client.

A GUI client allows easy access to cameras thier settings and their
measurements.
"""
from __future__ import division

#
# We need to import enaml.qt before matplotlib to avoid some qt errors
#
import enaml
from enaml.application import deferred_call, is_main_thread
from enaml.image import Image as EImage
from enaml.layout.dock_layout import InsertDockBarItem
from enaml.qt.qt_application import QtApplication
from enaml.qt.qt_factories import QT_FACTORIES

from atom.api import Atom, Bool, Signal, Float, Int, Str, Unicode, \
     Typed, observe, Dict, Value, List, Tuple, Instance, ForwardTyped

#
# Import the enaml view.
#
with enaml.imports():
    from CameraNetwork.gui.enaml_files.main_view import MainView, ArrayView
    from CameraNetwork.gui.enaml_files.popups import ThumbPopup
    from CameraNetwork.gui.enaml_files.settings import SettingsDialog

import copy
import cPickle
import cv2
from datetime import datetime
import json
import logging
import math
from mayavi.modules.surface import Surface
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.tools.figure import clf
import os
import pkg_resources
import pymap3d
import random
import scipy.io as sio
import StringIO
import subprocess
import time
from threading import Thread
import traceback
from zmq.eventloop import ioloop

import CameraNetwork
from CameraNetwork import global_settings as gs
from CameraNetwork.export import exportToShdom
from CameraNetwork.mdp import MDP
from CameraNetwork.radiosonde import load_radiosonde
from CameraNetwork.sunphotometer import calcSunphometerCoords
from CameraNetwork.utils import buff2dict
from CameraNetwork.utils import DataObj
from CameraNetwork.utils import extractImgArray
from CameraNetwork.utils import sun_direction
from CameraNetwork.visualization import (convertMapData, loadMapData)

from .image_analysis import image_analysis_factory
QT_FACTORIES.update({"ImageAnalysis": image_analysis_factory})

import ephem
import numpy as np
import pandas as pd

#
# I added this 'Qt4Agg' to avoid the following error:
# "TypeError: 'figure' is an unknown keyword argument"
# See:
# https://github.com/matplotlib/matplotlib/issues/3623/
#
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib.figure import Figure


ROI_length = 6000
EPIPOLAR_N = 200
EPIPOLAR_length = 10000
MAP_ZSCALE = 3


################################################################################
# Callback for the controller
################################################################################
def new_thumbnail(img):
    thumb_win = ThumbPopup(
        img=img,
    )
    thumb_win.show()


def open_settings(main_view, main_model, server_model):
    """Open settings popup window."""

    hresult = SettingsDialog(
        main_view,
        main_model=main_model,
        server_model=server_model
    ).exec_()

    if hresult:
        main_model.send_message(
            server_model,
            gs.MSG_TYPE_SET_SETTINGS,
            kwds=dict(
                camera_settings=server_model.camera_settings,
                capture_settings=server_model.capture_settings
            )
        )


################################################################################
# Sub models.
################################################################################
class LoggerModel(Atom):
    """Model of the Exception logger."""

    text = Str()

    def log(self, server_id, msg):
        """Add a log message."""

        self.text = self.text + "Server {} raised an error:\n" \
            "=========================\n{}".format(server_id, msg)

    def clear(self):
        """Clear all messages."""

        self.text = ""


class Map3dModel(Atom):
    """Model of the 3D map, showing the terrain, cameras and reconstruction."""

    main_model = ForwardTyped(lambda: MainModel)

    map_scene = Typed(MlabSceneModel)
    map_coords = Tuple()

    cameras_ROIs = Dict()
    grid_cube = Typed(Surface)

    #
    # Flags for controlling the map details.
    #
    show_ROIs = Bool(False)
    show_grid = Bool(False)

    latitude = Float(32.775776)
    longitude = Float(35.024963)
    altitude = Int(229)

    def _default_map_scene(self):
        """Draw the default map scene."""

        #
        # Load the map data.
        #
        self.map_coords = loadMapData()

        #
        # Create the mayavi scene.
        #
        scene = MlabSceneModel()
        return scene

    def draw_camera(self, server_id, img_data):
        """Draw a camera on the map."""

        n, e, d = pymap3d.geodetic2ned(
            img_data.latitude, img_data.longitude, img_data.altitude,
            lat0=self.latitude, lon0=self.longitude, h0=self.altitude)
        x, y, z = e, n, -d

        #
        # Draw a red sphere at the camera center.
        #
        self.map_scene.mlab.points3d(
            [x], [y], [MAP_ZSCALE * z],
            color=(1, 0, 0), mode='sphere',
            scale_mode='scalar', scale_factor=500,
            figure=self.map_scene.mayavi_scene
        )

        #
        # Draw the camera ROI
        #
        triangles = [
            (0, 1, 2),
            (0, 2, 4),
            (0, 4, 3),
            (0, 3, 1),
        ]

        phi = [0, np.pi/2, np.pi, 0]
        psi = [np.pi/10] * 4

        x_ = np.insert(x + ROI_length * np.sin(phi), 0, x)
        y_ = np.insert(y + ROI_length * np.cos(phi), 0, y)
        z_ = np.insert(z + ROI_length * np.cos(psi), 0, z)

        roi_mesh = self.map_scene.mlab.triangular_mesh(
            x_,
            y_,
            MAP_ZSCALE * z_,
            triangles,
            color=(0.5, 0.5, 0.5),
            opacity=0.2
        )
        roi_mesh.visible = self.show_ROIs
        self.cameras_ROIs[server_id] = roi_mesh

        #
        # Write the id of the camera.
        #
        self.map_scene.mlab.text3d(x, y, z+50, server_id, color=(0, 0, 0), scale=500.)

    def draw_grid(self):
        """Draw the reconstruction grid/cube on the map."""

        if self.main_model.GRID_NED == ():
            return

        X, Y, Z = self.main_model.GRID_NED
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        z_min, z_max = Z.min(), Z.max()

        x = np.array((x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min))
        y = np.array((y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max))
        z = np.array((z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max))

        triangles = [
            (0, 1, 5),
            (1, 2, 6),
            (2, 3, 7),
            (3, 0, 4),
            (0, 5, 4),
            (1, 6, 5),
            (2, 7, 6),
            (3, 4, 7),
            (4, 5, 6),
            (6, 7, 4)
        ]

        grid_mesh = self.map_scene.mlab.triangular_mesh(
            x,
            y,
            -z,
            triangles,
            color=(0, 0, 1),
            opacity=0.2
        )
        self.grid_cube = grid_mesh
        grid_mesh.visible = self.show_grid

    def draw_map(self):
        """Clear the map view and draw elevation map."""

        mayavi_scene = self.map_scene.mayavi_scene
        self.cameras_ROIs = dict()
        clf(figure=mayavi_scene)
        X, Y, Z, Z_mask = convertMapData(
            self.map_coords[0],
            self.map_coords[1],
            self.map_coords[2],
            lat0=self.latitude,
            lon0=self.longitude,
            alt0=self.altitude,
        )

        self.map_scene.mlab.surf(Y, X, MAP_ZSCALE * Z, figure=mayavi_scene, mask=Z_mask)

    def updateROImesh(self, server_id, pts, shape):
        """Update the 3D visualization of the camera ROI."""

        center = float(shape[0]) / 2

        pts = (pts - center) / center
        X, Y = pts[:, 0], pts[:, 1]

        phi = np.arctan2(X, Y)
        psi = np.pi/2 * np.sqrt(X**2 + Y**2)

        roi_mesh = self.cameras_ROIs[server_id]

        x, y, z = roi_mesh.mlab_source.points[0]

        x_ = np.insert(x + ROI_length * np.sin(phi), 0, x)
        y_ = np.insert(y + ROI_length * np.cos(phi), 0, y)
        z_ = np.insert(z + ROI_length * np.cos(psi), 0, z)

        roi_mesh.mlab_source.set(
            x=x_, y=y_, z=MAP_ZSCALE * z_
        )

    @observe("show_grid")
    def _showGrid(self, change):
        """Show/Hide the grid cube visualization."""

        if self.grid_cube is None:
            return

        self.grid_cube.visible = change["value"]

    @observe("show_ROIs")
    def _showCamerasROIs(self, change):
        """Show/Hide the camera's ROI visualization."""

        for roi_mesh in self.cameras_ROIs.values():
            roi_mesh.visible = change["value"]


class TimesModel(Atom):
    """Model of the capture times tables."""

    days_list = List()
    images_df = Typed(pd.DataFrame)
    img_index = Tuple()

    def _default_images_df(self):
        """Initialize an empty data frame."""

        df = pd.DataFrame(columns=('Time', 'hdr')).set_index(['Time', 'hdr'])
        return df

    def updateDays(self, days_list):
        """Update the list of available days."""

        days_list = [datetime.strptime(d, "%Y_%m_%d").date() for d in days_list]
        self.days_list = sorted(set(days_list + self.days_list))

    def updateTimes(self, server_id, images_df):
        """Update the times table."""

        images_series = images_df["path"]
        images_series.name = server_id


        new_df = self.images_df.copy()
        if server_id in self.images_df.columns:
            new_df.drop(server_id, axis=1, inplace=True)
        new_df = pd.concat((new_df, images_series), axis=1)
        new_df = new_df.reindex_axis(sorted(new_df.columns), axis=1)

        self.images_df = new_df

    def clear(self):
        """Clear the times table."""

        self.images_df = self._default_images_df()


class ArrayModel(Atom):
    """Representation of an image array."""

    main_model = ForwardTyped(lambda: MainModel)
    arrays_model = ForwardTyped(lambda: ArraysModel)

    img_data = Typed(DataObj, kwargs={})
    img_array = Typed(np.ndarray)

    resolution = Int(301)
    fov = Float(math.pi/2)

    #
    # Earth coords of the camera.
    #
    longitude = Float(gs.DEFAULT_LONGITUDE)
    latitude = Float(gs.DEFAULT_LATITUDE)
    altitude = Float(gs.DEFAULT_ALTITUDE)

    #
    # The center of the camera in ECEF coords.
    #
    center = Tuple()

    #
    # The mouse click LOS projected to camera coords.
    #
    Epipolar_coords = Tuple()

    #
    # The reconstruction grid projected to camera coords.
    #
    GRID_coords = Tuple()

    #
    # Coords of the Almucantar and Principle Planes controls.
    #
    Almucantar_coords = List(default=[])
    PrincipalPlane_coords = List(default=[])

    #
    # ROIs state.
    #
    ROI_state = Dict()
    mask_ROI_state = Dict()

    def _default_Epipolar_coords(self):
        Epipolar_coords = self.projectECEF(self.arrays_model.LOS_ECEF)
        return Epipolar_coords

    def _default_GRID_coords(self):
        GRID_coords = self.projectECEF(self.main_model.GRID_ECEF)
        return GRID_coords

    def calcLOS(self, x, y):
        """Create set of points in space.

        Create a Line Of Sight (LOS) points, set by the
        x,y coords of the mouse click on this view.

        Args:
            x, y (ints): view coords of mouse click.
            N (int): Number of points (resolution) of LOS.

        Returns:
             Returns the LOS points in ECEF coords.
        """

        #
        # Center the click coords around image center.
        #
        x = (x - self.resolution/2) / (self.resolution/2)
        y = (y - self.resolution/2) / (self.resolution/2)

        #
        # Calculate angle of click.
        # Note:
        # phi is the azimuth angle, 0 at the north and increasing
        # east. The given x, y are East and North correspondingly.
        # Therefore there is a need to transpose them as the atan
        # is defined as atan2(y, x).
        #
        phi = math.atan2(x, y)
        psi = self.fov * math.sqrt(x**2 + y**2)

        #
        # Calculate a LOS in this direction.
        # The LOS is first calculated in local coords (NED) of the camera.
        #
        pts = np.linspace(100, gs.LOS_LENGTH, gs.LOS_PTS_NUM)
        Z = -math.cos(psi) * pts
        X = math.sin(psi) * math.cos(phi) * pts
        Y = math.sin(psi) * math.sin(phi) * pts

        #
        # Calculate the LOS in ECEF coords.
        #
        LOS_ECEF = pymap3d.ned2ecef(
            X, Y, Z, self.latitude, self.longitude, self.altitude)

        return LOS_ECEF

    def projectECEF(self, ECEF_pts, filter_fov=True):
        """Project set of points in ECEF coords on the view.

        Args:
            ECEF_pts (tuple of arrays): points in ECEF coords.
            fiter_fov (bool, optional): If True, points below the horizion
               will not be returned. If false, the indices of these points
               will be returned.

        Returns:
            points projected to the view of this server.
        """

        #
        # Convert ECEF points to NED centered at camera.
        #
        X, Y, Z = pymap3d.ecef2ned(
            ECEF_pts[0], ECEF_pts[1], ECEF_pts[2],
            self.latitude, self.longitude, self.altitude)

        #
        # Convert the points to NEU.
        #
        neu_pts = np.array([X.flatten(), Y.flatten(), -Z.flatten()]).T

        #
        # Normalize the points
        #
        with np.errstate(all='raise'):
            neu_pts = \
                neu_pts/np.linalg.norm(neu_pts, axis=1).reshape(-1, 1)

        #
        # Zero points below the horizon.
        #
        cosPSI = neu_pts[:,2].copy()
        cosPSI[cosPSI<0] = 0

        #
        # Calculate The x, y of the projected points.
        # Note that x and y here are according to the pyQtGraph convention
        # of right, up (East, North) respectively.
        #
        PSI = np.arccos(neu_pts[:,2])
        PHI = np.arctan2(neu_pts[:,1], neu_pts[:,0])
        R = PSI / self.fov * self.resolution/2
        xs = R * np.sin(PHI) + self.resolution/2
        ys = R * np.cos(PHI) + self.resolution/2

        if filter_fov:
            return xs[cosPSI>0], ys[cosPSI>0]
        else:
            return xs, ys, cosPSI>0

    @observe("img_array")
    def _update_img_data(self, change):

        if change["value"] is None:
            return

        self.resolution = self.img_array.shape[0]

    @observe("img_data")
    def _update_img_data(self, change):

        if change["value"] is None:
            return

        self.longitude = float(self.img_data.longitude)
        self.latitude = float(self.img_data.latitude)
        self.altitude = float(self.img_data.altitude)

        self.center = pymap3d.ned2ecef(
            0, 0, 0,
            self.latitude,
            self.longitude,
            self.altitude
        )

        alm_coords, pp_coords = \
            calcSunphometerCoords(self.img_data, resolution=self.resolution)
        self.Almucantar_coords = alm_coords
        self.PrincipalPlane_coords = pp_coords

    @observe('arrays_model.LOS_ECEF')
    def _updateEpipolar(self, change):
        """Project the LOS points (mouse click position) to camera."""

        self.Epipolar_coords = self.projectECEF(self.arrays_model.LOS_ECEF)

    @observe('main_model.GRID_ECEF')
    def _updateGRID(self, change):
        """Project the reconstruction GRID points to camera coords."""

        self.GRID_coords = self.projectECEF(self.main_model.GRID_ECEF)


class ArraysModel(Atom):
    """Model of the currently displayed arrays."""

    main_model = ForwardTyped(lambda: MainModel)

    array_items = Dict()
    array_views = Dict()

    #
    # Intensity level for displayed images.
    #
    intensity = Int(100)
    gamma = Bool(False)

    #
    # Flags for controlling visualization.
    #
    show_ROIs = Bool(True)
    show_grid = Bool(False)
    show_masks = Bool(False)

    #
    # The 'mouse click' Line Of Site points in ECEF coords.
    #
    LOS_ECEF = Tuple()

    def _default_LOS_ECEF(self):
        """Initialize the default LOS in ECEF coords."""

        X, Y, Z = np.meshgrid(
            np.zeros(1),
            np.zeros(1),
            np.linspace(-gs.LOS_LENGTH, -100, gs.LOS_PTS_NUM),
        )

        LOS_ECEF = pymap3d.ned2ecef(
            X, Y, Z, self.main_model.latitude, self.main_model.longitude, self.main_model.altitude)

        return LOS_ECEF

    def clear_arrays(self):
        """Clear all arrays."""

        self.array_items = dict()
        self.array_views = dict()

    def new_array(self, server_id, img_array, img_data):
        """This callback is called when a new array is added to the display.

        The callback creates all objects required to display the image.

        Args:
            server_id (str): ID of the server to which an array is added.
            img_array (array): New image.
            img_data (dict): Meta data of the image.
        """

        if img_array.ndim == 4:
            #
            # Multiple images are reduced to single frame
            # by averaging.
            #
            img_array = np.mean(img_array, axis=3).astype(np.uint8)

        #
        # Create the array model which handles the array view on the display.
        #
        server_keys = self.array_items.keys()
        if server_id in server_keys:
            #
            # The specific Server/Camera is already displayed. Update the array
            # model and view.
            #
            array_model = self.array_items[server_id]

            new_array_model = False
        else:
            #
            # The specific camera is not displayed. Create it.
            #
            new_array_model = True
            array_model = ArrayModel(main_model=self.main_model, arrays_model=self)

        #
        # Update the model.
        #
        array_model.img_array = img_array
        array_model.img_data = img_data

        if new_array_model:
            temp_dict = self.array_items.copy()
            temp_dict[server_id] = array_model
            self.array_items = temp_dict

    def updateLOS(self, data):
        """Handle click events on image array."""

        server_id = data['server_id']
        pos_x, pos_y = data['pos']

        clicked_model = self.array_items[server_id]
        self.LOS_ECEF = clicked_model.calcLOS(pos_x, pos_y)

    def save_rois(self, base_path=None):
        """Save the current ROIS for later use."""

        #
        # Prepare a (semi-)unique path.
        #
        if base_path is None:
            base_path = pkg_resources.resource_filename("CameraNetwork", "../data/ROIS")
            if not os.path.exists(base_path):
                os.makedirs(base_path)

        array_model = self.array_items.values()[0]
        dst_path = os.path.join(
            base_path,
            array_model.img_data.name_time.strftime("%Y_%m_%d_%H_%M_%S.pkl")
        )

        #
        # Get the states of the ROIs.
        #
        rois_dict = {}
        masks_dict = {}
        array_shapes = {}
        for server_id in self.array_items.keys():
            rois_dict[server_id] = self.array_items[server_id].ROI_state
            masks_dict[server_id] = self.array_items[server_id].mask_ROI_state
            array_shapes[server_id] = self.array_items[server_id].img_array.shape[:2]

        #
        # Save a pickle.
        #
        with open(dst_path, 'wb') as f:
            cPickle.dump((rois_dict, masks_dict, array_shapes), f)

    def load_rois(self, path='./ROIS.pkl'):
        """Apply the saved rois on the current arrays."""

        try:
            #
            # Load the saved states.
            #
            with open(path, 'rb') as f:
                rois_dict, masks_dict, array_shapes = cPickle.load(f)

            #
            # Update the ROIs states.
            #
            for server_id in self.array_items.keys():
                if server_id not in rois_dict:
                    continue

                self.array_items[server_id].ROI_state = rois_dict[server_id]
                self.array_items[server_id].mask_ROI_state = masks_dict[server_id]
                #self.array_views[server_id].image_widget.update_ROI_resolution(array_shapes[server_id])

        except Exception as e:
            logging.error(
                "Failed setting rois to Arrays view:\n{}".format(
                    traceback.format_exc()))


################################################################################
# Main model.
################################################################################
class MainModel(Atom):
    """The data model of the client."""

    #
    # Communication objects with the camera network.
    #
    thread = Typed(Thread)
    client_instance = Typed(CameraNetwork.Client)

    #
    # Submodels.
    #
    logger = Typed(LoggerModel)
    map3d = Typed(Map3dModel)
    times = Typed(TimesModel)
    arrays = Typed(ArraysModel)

    #
    # Book keeping.
    #
    servers_dict = Dict()
    tunnels_dict = Dict()

    #
    # Popup thumbnail.
    # Note:
    # This was used mainly when the calibration was done in the camera.
    # Now the calibration is done in the lab.
    #
    thumb = Typed(EImage)

    settings_signal = Signal()

    sunshader_required_angle = Int()

    #
    # Reconstruction parameters
    # The default values are the Technion lidar position
    #
    latitude = Float(32.775776)
    longitude = Float(35.024963)
    altitude = Int(229)

    #
    # Reconstruction Grid parameters.
    # Note:
    # There are two grids used:
    # - GRID_ECEF: Used for visualization on the camera array.
    # - GRID_NED: The grid exported for reconstruction.
    #
    delx = Float(100)
    dely = Float(100)
    delz = Float(100)
    TOG = Float(6000)
    GRID_ECEF = Tuple()
    GRID_NED = Tuple()
    grid_mode = Str("Manual")
    grid_width = Float(6000)
    grid_length = Float(6000)

    #
    # Sunshader mask threshold used in grabcut algorithm.
    #
    grabcut_threshold = Float(3)

    #
    # Global (broadcast) capture settings.
    #
    capture_settings = Dict(default=gs.CAPTURE_SETTINGS)

    #
    # Progress bar value for export status
    #
    export_progress = Int()

    ############################################################################
    # Default constructors
    ############################################################################
    def _default_logger(self):
        """Initialize the logger object."""

        return LoggerModel()

    def _default_map3d(self):
        """Initialize the map 3D."""

        return Map3dModel(
            main_model=self
        )

    def _default_times(self):
        """Initialize the times model."""

        return TimesModel()

    def _default_arrays(self):
        """Initialize the reconstruction grid."""

        return ArraysModel(main_model=self)

    def _default_GRID_NED(self):
        """Initialize the reconstruction grid."""

        self.updateGRID()

        return self.GRID_NED

    def _default_GRID_ECEF(self):
        """Initialize the reconstruction grid."""

        self.updateGRID()
        return self.GRID_ECEF

    ############################################################################
    # GUI communication.
    ############################################################################
    def start_camera_thread(self, local_mode):
        """Start a camera client on a separate thread."""

        #
        # Create the camera client instance
        #
        proxy_params = CameraNetwork.retrieve_proxy_parameters(local_mode)
        client_instance = CameraNetwork.Client(proxy_params)

        #
        # Bind callbacks
        #
        client_instance.handle_new_server = self.handle_new_server_cb
        client_instance.handle_server_failure = self.handle_server_failure_cb
        client_instance.handle_receive = self.handle_receive_cb
        client_instance.tunnels_cb = self.tunnels_cb

        self.client_instance = client_instance

        #
        # Start the camera thread
        #
        self.thread = Thread(target=self.client_instance.start, args=(gs.GUI_STARTUP_DELAY,))
        self.thread.daemon = True
        self.thread.start()

    def send_message(self, server_model, cmd=None, args=(), kwds={}):
        """Send message to a specific server."""

        logging.debug('sending message from:' + str(server_model))
        loop = ioloop.IOLoop.instance()

        if cmd is not None:
            loop.add_callback(
                self.client_instance.send,
                server_address=server_model.server_id,
                cmd=cmd, args=args, kwds=kwds
            )
        else:
            loop.add_callback(
                self.client_instance.send,
                **server_model.get_msg_out()
            )

    def send_mmi(self, service, msg=[]):
        """Send message to proxy."""

        logging.debug('sending mmi: {}'.format(service))
        loop = ioloop.IOLoop.instance()

        loop.add_callback(
            self.client_instance.send_mmi,
            service=service,
            msg=msg
        )

    def broadcast_message(self, cmd, args=(), kwds={}):
        """Send message to to all online servers."""

        logging.debug("Broadcasting message: {}".format(cmd))
        loop = ioloop.IOLoop.instance()

        for server_id, server_model in self.servers_dict.items():
            if cmd not in gs.LOCAL_MESSAGES and server_id.endswith("L"):
                #
                # some messages are not sent to local servers.
                #
                continue

            loop.add_callback(
                self.client_instance.send,
                server_address=server_model.server_id, cmd=cmd,
                msg_extra=MDP.BROADCAST, args=args, kwds=kwds
            )

    ############################################################################
    # GUI Actions.
    ############################################################################
    def open_tunnel(self, server_id, tunnel_details):
        """Open the putty client"""

        tunnel_user = tunnel_details['password']
        tunnel_port = tunnel_details['tunnel_port']
        tunnel_pw = tunnel_details['password']
        tunnel_ip = self.client_instance.proxy_params['ip']

        putty_cmd = 'kitty_portable -P {port} -pw {password} {user}@{proxy_ip} -title "GLOBAL Camera {title}"'.format(
            user=tunnel_user,
            password=tunnel_pw,
            port=tunnel_port,
            proxy_ip=tunnel_ip,
            title=server_id
        )
        subprocess.Popen(putty_cmd)

    def get_revisions_list(self):
        """Get the revision list of the local repository."""

        local_repo = CameraNetwork.Repository(os.getcwd())

        #
        # Check if the local copy is synced with the remote.
        #
        incoming_log = local_repo.incoming()
        outgoing_log = local_repo.outgoing()

        if incoming_log.find('no changes found') == -1 or outgoing_log.find('no changes found') == -1:
            return []

        #
        # Get list of revisions
        #
        repo_log = local_repo.log('-l', '10', '--template', '{rev}***{desc}***{node}||||')
        temp = [rev.strip().split('***') for rev in repo_log.strip().split('||||')[:-1]]
        revisions = [(rev_num+': '+desc, node) for rev_num, desc, node in temp]

        return revisions

    def updateExportProgress(self, progress_ratio):
        """Update the status of the progress bar.

        Args:
            progress_ratio (float): Progress ratio (0 to 1).

        """

        self.export_progress = int(100*progress_ratio)

    ############################################################################
    # MDP Callbacks
    ############################################################################
    def tunnels_cb(self, tunnels):
        """Update the tunnels dict."""

        deferred_call(self.tunnels_update, tunnels)

    def settings(self, server_model):
        pass

    def handle_new_server_cb(self, server_id):
        """Deffer the callback to the gui loop."""

        deferred_call(self.add_server, server_id)

    def handle_server_failure_cb(self, server_id):
        """Deffer the callback to the gui loop."""

        deferred_call(self.remove_server, server_id)

    def handle_receive_cb(self, msg_extra, service, status, cmd, args, kwds):
        """Deffer the callback to the gui loop."""

        deferred_call(self.receive_message, msg_extra, service, status, cmd, args, kwds)

    ############################################################################
    # Cross thread callback "mirrors"
    ############################################################################
    def tunnels_update(self, tunnels):
        self.tunnels_dict = tunnels

    def add_server(self, server_id):
        logging.info('Adding the new server: {}'.format(server_id))

        temp_dict = self.servers_dict.copy()
        new_server = ServerModel(server_id=server_id, main_model=self)
        new_server.init_server()
        temp_dict[server_id] = new_server
        self.servers_dict = temp_dict

    def remove_server(self, server_id):
        logging.info('Removing the server: {}'.format(server_id))

        temp_dict = self.servers_dict.copy()
        temp_dict.pop(server_id, None)
        self.servers_dict = temp_dict

    def receive_message(self, msg_extra, service, status, cmd, args, kwds):

        server_id = service

        if status == gs.MSG_STATUS_ERROR:
            #
            # Log the error message.
            #
            self.logger.log(server_id, args[0])

            return

        #
        # Update the msg box.
        #
        if server_id in self.servers_dict:
            self.servers_dict[server_id].put_msg_in(status, cmd, args, kwds)

        if msg_extra & MDP.BROADCAST and hasattr(self, 'reply_broadcast_{}'.format(cmd)):
            #
            # If reply to broadcast command with special reply handler, use it.
            #
            self.handle_broadcast_reply(status, server_id, cmd, args, kwds)
        elif server_id in self.servers_dict:
            #
            # Use the standard reply handler.
            #
            self.servers_dict[server_id].handle_reply(status, cmd, args, kwds)

    def handle_broadcast_reply(self, status, server_id, cmd, args, kwds):
        """Handle broadcast reply"""

        cb = getattr(self, 'reply_broadcast_{}'.format(cmd), None)

        if cb is not None:
            cb(server_id, *args, **kwds)

    ############################################################################
    # Handle reply to broadcast messages.
    ############################################################################
    def reply_broadcast_days(self, server_id, days_list):
        """Handle the broadcast reply of the days command."""

        self.times.updateDays(days_list)

    def reply_broadcast_query(self, server_id, images_df):
        """Handle the broadcast reply of the query command."""

        logging.debug("Got reply query {}.".format(server_id))

        self.times.updateTimes(server_id, images_df)

    def reply_broadcast_seek(self, server_id, matfile, img_data):
        """Handle the broadcast reply of the seek command."""

        self.new_array(server_id, matfile, img_data)

    ############################################################################
    # Misc.
    ############################################################################
    def updateGRID(self):
        """Update the reconstruction grid.

        The grid is calculate in ECEF coords.
        """

        #
        # Calculate the bounding box of the grid.
        #
        s_pts = np.array((-self.grid_length/2, -self.grid_width/2, -self.TOG))
        e_pts = np.array((self.grid_length/2, self.grid_width/2, 0))

        #
        # Create the grid.
        # Note GRID_NED is an open grid storing the requested
        # grid resolution. It is used for reconstruction and also
        # for visualization in the 3D map.
        #
        self.GRID_NED = (
            np.arange(s_pts[0], e_pts[0]+self.delx, self.delx),
            np.arange(s_pts[1], e_pts[1]+self.dely, self.dely),
            np.arange(s_pts[2], e_pts[2]+self.delz, self.delz)
        )

        #
        # GRID_ECEF is just for visualization of the grid on the
        # image arrays.
        #
        X, Y, Z = np.meshgrid(
            np.linspace(s_pts[0], e_pts[0], 10),
            np.linspace(s_pts[1], e_pts[1], 10),
            np.linspace(s_pts[2], e_pts[2], 10),
        )

        self.GRID_ECEF = pymap3d.ned2ecef(
            X, Y, Z, self.latitude, self.longitude, self.altitude)

    def updateROIs(self, data):
        """Handle update of a server ROI."""

        server_id = data['server_id']
        pts = data['pts']
        shape = data['shape']

        self.map3d.updateROImesh(server_id, pts, shape)

    def clear_map(self):
        #
        # TODO:
        # Just remove cameras and not the map/grid.
        #
        self.draw_map()
        self.draw_grid()

    def draw_map(self):
        self.map3d.draw_map()

    def draw_grid(self):
        self.map3d.draw_grid()

    def new_array(self, server_id, matfile, img_data):
        """Handle a new array."""

        img_array = extractImgArray(matfile)

        #
        # Draw the camera on the map.
        #
        self.map3d.draw_camera(server_id, img_data)

        #
        # Add new array.
        #
        self.arrays.new_array(server_id, img_array, img_data)

    def exportData(self):
        """Export data for reconstruction."""

        if len(self.arrays.array_items.keys()) == 0:
            return

        #
        # Unique base path
        #
        array_model = self.arrays.array_items.values()[0]
        base_path = os.path.join(
            'reconstruction',
            array_model.img_data.name_time.strftime("%Y_%m_%d_%H_%M_%S")
        )
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        #
        # Get the radiosonde
        #
        date = array_model.img_data.name_time
        rs_df = load_radiosonde(date)
        rs_df[['HGHT', 'TEMP']].to_csv(
            os.path.join(base_path, 'radiosonde.csv'))

        #
        # Save the ROIs
        #
        self.arrays.save_rois(base_path=base_path)

        #
        # Save the GRIDs
        #
        sio.savemat(
            os.path.join(base_path, 'grid.mat'),
            dict(
                X=self.GRID_NED[0],
                Y=self.GRID_NED[1],
                Z=self.GRID_NED[2]
                ),
            do_compression=True
        )

        #
        # Match array_models and array_views.
        #
        array_items = {}
        for array_view in self.arrays.array_views.values():
            if not array_view.export_flag.checked:
                logging.info(
                    "Reconstruction: Camera {} ignored.".format(array_view.server_id)
                )
                continue

            server_id = array_view.server_id
            array_items[server_id] = (self.arrays.array_items[server_id], array_view)

        #
        # Start export on separate thread.
        #
        thread = Thread(
            target=exportToShdom,
            kwargs=dict(
                base_path=base_path,
                array_items=array_items,
                grid=self.GRID_NED,
                lat=self.latitude,
                lon=self.longitude,
                alt=self.altitude,
                grabcut_threshold=self.grabcut_threshold,
                progress_callback=self.updateExportProgress
            )
        )
        thread.daemon = True
        thread.start()


class ServerModel(Atom):
    """The data model of the server (camera)."""

    server_id = Str()
    cmd = Str()
    reply = Str()
    reply_data = Str()
    tunnel_port = Int()
    tunnel_ip = Str()
    sunshader_figure = Value()
    radiometric_figure = Value()
    extrinsic_scene = Typed(MlabSceneModel)
    sunshader_required_angle = Int()
    camera_settings = Dict(default=gs.CAMERA_SETTINGS)
    capture_settings = Dict(default=gs.CAPTURE_SETTINGS)
    status_text = Str()

    main_model = Typed(MainModel)

    days_list = List()

    images_df = Typed(pd.DataFrame)
    img_index = Tuple()

    def init_server(self):
        pass

    def _default_sunshader_figure(self):
        """Draw the default plot figure."""

        figure = Figure(figsize=(2, 1))
        ax = figure.add_subplot(111)
        x = np.arange(20, 160)
        ax.plot(x, np.zeros_like(x))

        return figure

    def _default_radiometric_figure(self):
        """Draw the default plot figure."""

        figure = Figure(figsize=(2, 1))
        ax = figure.add_subplot(111)
        x = np.arange(20, 160)
        ax.plot(x, np.zeros_like(x))

        return figure

    def _default_extrinsic_scene(self):
        """Draw the default extrinsic plot scene."""

        scene = MlabSceneModel()
        return scene

    def _default_images_df(self):
        """Initialize an empty data frame."""

        df = pd.DataFrame(columns=('Time', 'hdr')).set_index(['Time', 'hdr'])
        return df

    ############################################################################
    # Low level communication.
    ############################################################################
    def get_msg_out(self):
        return {
            'server_address': self.server_id,
            'cmd': self.cmd,
        }

    def put_msg_in(self, status, cmd, args, kwds):

        self.cmd = cmd
        self.reply = status
        try:
            self.reply_data = json.dumps((args, kwds), indent=4)
        except Exception:
            self.reply_data = "Non jsonable data"

    def handle_reply(self, status, cmd, args, kwds):
        """Handle standard reply."""

        cb = getattr(self, 'reply_{}'.format(cmd), None)

        if cb is not None:
            cb(*args, **kwds)

    ############################################################################
    # Reply handler (replies from servers).
    ############################################################################
    def reply_tunnel_details(self, tunnel_port, tunnel_ip, tunnel_user, tunnel_pw):
        """Open the putty client"""

        putty_cmd = 'kitty_portable -P {port} -pw {password} {user}@{proxy_ip} -title "Camera {title}"'.format(
            user=tunnel_user,
            password=tunnel_pw,
            port=tunnel_port,
            proxy_ip=tunnel_ip,
            title=self.server_id
        )

        subprocess.Popen(putty_cmd)

    def reply_local_ip(self, port, ip, user, pw):
        """Open the putty client"""

        putty_cmd = 'kitty_portable -P {port} -pw {password} {user}@{proxy_ip} -title "Camera {title}"'.format(
            user=tunnel_user,
            password=tunnel_pw,
            port=tunnel_port,
            proxy_ip=tunnel_ip,
            title=self.server_id
        )

        subprocess.Popen(putty_cmd)

    def reply_status(self, git_result, memory_result):
        """Open the putty client"""

        self.status_text = \
            "Memory Status:\n--------------\n{}\n\nGit HEAD:\n---------\n{}".format(
            memory_result[0], git_result[0]
        )

    def reply_get_settings(self, camera_settings, capture_settings):
        """Handle reply of settings."""

        #
        # Start with temp settings incase the camera is not updated with new
        # settings.
        #
        temp_settings = copy.copy(gs.CAMERA_SETTINGS)
        temp_settings.update(camera_settings)
        self.camera_settings = temp_settings

        temp_settings = copy.copy(gs.CAPTURE_SETTINGS)
        temp_settings.update(capture_settings)
        self.capture_settings = temp_settings

        #
        # Open the settings popup.
        #
        self.main_model.settings_signal.emit(self)

    def reply_thumbnail(self, thumbnail):
        #
        # The thumbnails are sent as jpeg. But the pyqt (the backend of enaml) is compiled
        # without support for jpeg, therefore I have to convert it manually to raw rgba.
        #
        buff = StringIO.StringIO(thumbnail)
        img = Image.open(buff)
        width, height = img.size
        array = np.array(img.getdata(), np.uint8)

        #
        # Handle gray scale image
        #
        if array.ndim == 1:
            array.shape = (-1, 1)
            array = np.hstack((array, array, array))

        array = np.hstack((array[:, ::-1], np.ones((width*height, 1), dtype=np.uint8)*255))
        self.main_model.thumb = EImage(data=array.tostring(), format='argb32', raw_size=(width, height))

    def reply_radiometric(self, angles, measurements, estimations, ratios):
        """Handle to reply for the radiometric calibration."""

        f = Figure(figsize=(2, 1))

        for i, wl in enumerate(("Red", "Green", "Blue")):
            ax = f.add_subplot(131+i)
            ax.plot(angles, measurements[i], label="spm")
            ax.plot(angles, estimations[i], label="cam")
            ax.set_title(wl)
            ax.legend()

        self.radiometric_figure = f

    def reply_sunshader_scan(self, angles, saturated_array, sun_signal, required_angle):
        f = Figure(figsize=(2, 1))
        ax = f.add_subplot(111)
        ax.plot(angles, saturated_array)
        self.sunshader_figure = f
        self.sunshader_required_angle = int(required_angle)

    def reply_extrinsic(self, rotated_directions, calculated_directions, R):
        """Update the extrinsic calibration view."""

        scene = self.extrinsic_scene.mayavi_scene
        clf(figure=scene)
        self.extrinsic_scene.mlab.points3d(
            rotated_directions[:, 0], rotated_directions[:, 1], rotated_directions[:, 2],
            color=(1, 0, 0), mode='sphere', scale_mode='scalar', scale_factor=0.02,
            figure=scene
        )
        self.extrinsic_scene.mlab.points3d(
            calculated_directions[:, 0], calculated_directions[:, 1], calculated_directions[:, 2],
            color=(0, 0, 1), mode='sphere', scale_mode='scalar', scale_factor=0.02,
            figure=scene
        )

    def reply_array(self, matfile, img_data):

        #
        # Add new array.
        #
        self.main_model.new_array(self.server_id, matfile, img_data)

    def reply_days(self, days_list):
        """Handle the reply for days command."""

        self.days_list = [datetime.strptime(d, "%Y_%m_%d").date() for d in days_list]

    def reply_query(self, images_df):
        """Handle the reply for query command."""

        self.images_df = images_df

    def reply_seek(self, matfile, img_data):

        #
        # Add new array.
        #
        self.main_model.new_array(self.server_id, matfile, img_data)

    def reply_calibration(self, img_array, K, D, rms, rvecs, tvecs):
        #
        # The imgs are sent as jpeg. But the pyqt (the backend of enaml) is compiled
        # without support for jpeg, therefore I have to convert it manually to raw rgba.
        #
        buff = StringIO.StringIO(img_array)
        img = Image.open(buff)
        width, height = img.size
        array = np.array(img.getdata(), np.uint8)

        #
        # Handle gray scale image
        #
        if array.ndim == 1:
            array.shape = (-1, 1)
            array = np.hstack((array, array, array))

        array = np.hstack((array[:, ::-1], np.ones((width*height, 1), dtype=np.uint8)*255))
        self.main_model.thumb = EImage(data=array.tostring(), format='argb32', raw_size=(width, height))


class Controller(Atom):

    model = Typed(MainModel)
    arrays = Typed(ArraysModel)

    view = Typed(MainView)

    @observe('model.settings_signal')
    def settings_signal(self, server_model):
        """Open a settings popup window."""

        open_settings(self.view, self.model, server_model)

    @observe('model.thumb')
    def new_thumb_popup(self, change):
        new_thumbnail(self.model.thumb)

    @observe('arrays.array_items')
    def update_arrays(self, change):
        if change["type"]  != 'update' or change["value"] == {}:
            return

        server_keys = change["value"].keys()
        new_server_id = list(
            set(server_keys) - set(change["oldvalue"].keys()))[0]

        view_index = sorted(server_keys).index(new_server_id)

        array_view = ArrayView(
            title=new_server_id,
            server_id=new_server_id,
            array_model=change["value"][new_server_id],
            arrays_model=self.arrays,
        )

        self.view.array_views.objects.insert(view_index, array_view)
        self.arrays.array_views[new_server_id] = array_view


################################################################################
################################################################################
#
# Entry point to start the GUI.
#
def startGUI(local_mode):
    """Start the GUI of the camera network."""

    #
    # Instansiate the data model
    #
    main_model = MainModel()
    main_model.start_camera_thread(local_mode)

    #
    # Start the Qt application
    #
    app = QtApplication()

    view = MainView(main_model=main_model)

    controller = Controller(model=main_model, arrays=main_model.arrays, view=view)

    view.show()

    app.start()
