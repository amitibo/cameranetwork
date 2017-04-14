"""Run a GUI Client.

A GUI client allows easy access to cameras thier settings and their
measurements.
"""
from __future__ import division

#
# We need to import enaml.qt before matplotlib to avoid some qt errors
#
import enaml.qt

import argparse
from atom.api import Atom, Bool, Enum, Signal, Float, Int, Str, Unicode, \
     Typed, observe, Dict, Value, List, Tuple, Instance
import copy
import cPickle
import cv2
from datetime import datetime
from enaml.application import deferred_call, is_main_thread
import json
import logging
import math
import matplotlib.mlab as ml
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.tools.figure import clf
import os
import pymap3d
import random
import scipy.io as sio
import StringIO
import subprocess
from threading import Thread
import traceback
from zmq.eventloop import ioloop

import CameraNetwork
from CameraNetwork import global_settings as gs
from CameraNetwork.export import exportToShdom
from CameraNetwork.mdp import MDP
from CameraNetwork.radiosonde import load_radiosonde
from CameraNetwork.utils import buff2dict
from CameraNetwork.utils import DataObj
from CameraNetwork.utils import extractImgArray
from CameraNetwork.utils import sun_direction

#
# We need to import enaml.qt before matplotlib to avoid some qt errors
#
import enaml.qt
import ephem
import numpy as np
import pandas as pd

from enaml.image import Image as EImage

#
# I added this 'Qt4Agg' to avoid the following error:
# "TypeError: 'figure' is an unknown keyword argument"
# See:
# https://github.com/matplotlib/matplotlib/issues/3623/
#
import matplotlib
matplotlib.use('Qt4Agg')

from CameraNetwork.sunphotometer import calcSunphometerCoords
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


ROI_length = 6000


def loadMapData():
    """Load height data for map visualization."""

    path1 = r"..\data\reconstructions\N32E034.hgt"
    path2 = r"..\data\reconstructions\N32E035.hgt"
    with open(path1) as hgt_data:
        hgt1 = np.fromfile(hgt_data, np.dtype('>i2')).reshape((1201, 1201))[:1200, :1200]
    with open(path2) as hgt_data:
        hgt2 = np.fromfile(hgt_data, np.dtype('>i2')).reshape((1201, 1201))[:1200, :1200]
    hgt = np.hstack((hgt1, hgt2)).astype(np.float32)
    lon, lat = np.meshgrid(np.linspace(34, 36, 2400, endpoint=False), np.linspace(32, 33, 1200, endpoint=False)[::-1])
    return lat[100:400, 1100:1400], lon[100:400, 1100:1400], hgt[100:400, 1100:1400]


def convertMapData(lat, lon, hgt, lat0=32.775776, lon0=35.024963, alt0=229):
    """Convert lat/lon/height data to grid data."""

    n, e, d = pymap3d.geodetic2ned(
        lat, lon, hgt,
        lat0=lat0, lon0=lon0, h0=alt0)

    x, y, z = e, n, -d

    xi = np.linspace(-10000, 10000, 100)
    yi = np.linspace(-10000, 10000, 100)
    X, Y = np.meshgrid(xi, yi)

    Z = ml.griddata(y.flatten(), x.flatten(), z.flatten(), yi, xi, interp='linear')

    return X, Y, Z


class ClientModel(Atom):
    """The data model of the client."""

    servers_dict = Dict()
    tunnels_dict = Dict()
    thread = Typed(Thread)
    client_instance = Typed(CameraNetwork.Client)
    thumb = Typed(EImage)
    logger_text = Str()

    new_array_signal = Signal()
    clear_arrays_signal = Signal()
    settings_signal = Signal()

    array_items = Dict()

    days_list = List()

    images_df = Typed(pd.DataFrame)
    img_index = Tuple()

    map_coords = Tuple()
    map_scene = Typed(MlabSceneModel)
    cameras_ROIs = Dict()

    sunshader_required_angle = Int()

    #
    # Reconstruction parameters
    # The default values are the Technion lidar position
    #
    latitude = Float(32.775776)
    longitude = Float(35.024963)
    altitude = Int(229)

    #
    # LIDAR Grid parameters.
    # The LIDAR grid size is the cube that includes
    # all cameras, and from 0 to Top Of Grid (TOG).
    #
    delx = Float(100)
    dely = Float(100)
    delz = Float(100)
    TOG = Float(3000)
    GRID_ECEF = Tuple()
    GRID_NED = Tuple()
    grid_mode = Str()
    grid_width = Float(3000)
    grid_length = Float(5000)

    #
    # Sunshader mask threshold used in grabcut algorithm.
    #
    grabcut_threshold = Float(3)

    #
    # Global (broadcast) capture settings.
    #
    capture_settings = Dict(default=gs.CAPTURE_SETTINGS)

    #
    # Intensity level for displayed images.
    #
    intensity_value = Int(40)

    #
    # Progress bar value for export status
    #
    export_progress = Int()

    def _default_images_df(self):
        """Initialize an empty data frame."""

        df = pd.DataFrame(columns=('Time', 'hdr')).set_index(['Time', 'hdr'])
        return df

    def _default_map_scene(self):
        """Draw the default map scene."""

        #
        # Load the map data.
        #
        lat, lon, hgt = loadMapData()
        self.map_coords = convertMapData(
            lat,
            lon,
            hgt,
            lat0=self.latitude,
            lon0=self.longitude,
            alt0=self.altitude,
        )

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
        # Draw a point at the camera center.
        #
        self.map_scene.mlab.points3d(
            [x], [y], [z],
            color=(1, 0, 0), mode='sphere', scale_mode='scalar', scale_factor=500,
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
            z_,
            triangles,
            color=(0.5, 0.5, 0.5),
            opacity=0.2
        )
        self.cameras_ROIs[server_id] = roi_mesh

        #
        # Write the id of the camera.
        #
        self.map_scene.mlab.text3d(x, y, z+50, server_id, color=(0, 0, 0), scale=500.)

    def draw_grid(self):
        """Draw the reconstruction grid on the map."""

        if self.GRID_NED == ():
            return

        X, Y, Z = self.GRID_NED
        X, Y, Z = np.meshgrid(X, Y, -Z)

        #
        # Draw a point at the camera center.
        #
        self.map_scene.mlab.points3d(
            X, Y, Z,
            color=(1, 1, 1), mode='point',
            figure=self.map_scene.mayavi_scene
        )

    def updateROImesh(self, server_id, pts, shape):
        """Update the 3D visualization of the ROI."""

        center = float(shape[0])/2

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
            x=x_, y=y_, z=z_
        )

    def showCamerasROIs(self, checked):
        """Show/Hide the camera's ROI visualization."""

        for roi_mesh in self.cameras_ROIs.values():
            roi_mesh.visible = checked

    def draw_map(self):
        """Clear the map view and draw elevation map."""

        mayavi_scene = self.map_scene.mayavi_scene
        self.cameras_ROIs = dict()
        clf(figure=mayavi_scene)
        X, Y, Z = self.map_coords
        self.map_scene.mlab.surf(Y, X, Z, figure=mayavi_scene)

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

    ############################################################################
    # GUI to MDP communication.
    ############################################################################
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

        for server_model in self.servers_dict.values():
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

    def exportData(self):
        """Export data for reconstruction."""

        if len(self.array_items.items()) == 0:
            return

        #
        # Unique base path
        #
        array_model = self.array_items.values()[0][0]
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
        self.save_rois(base_path=base_path)

        #
        # Save the GRIDs
        #
        sio.savemat(
            os.path.join(base_path, 'grid.mat'),
            dict(X=self.GRID_NED[0], Y=self.GRID_NED[1], Z=self.GRID_NED[2]),
            do_compression=True
        )

        #
        # Start export on separate thread.
        #
        thread = Thread(
            target=exportToShdom,
            kwargs=dict(
                base_path=base_path,
                array_items=self.array_items.items(),
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

    def updateExportProgress(self, progress_ratio):
        """Update the status of the progress bar.

        Args:
            progress_ratio (float): Progress ratio (0 to 1).

        """

        self.export_progress = int(100*progress_ratio)

    def save_rois(self, base_path='.'):
        """Save the current ROIS for later use."""

        dst_path = os.path.join(base_path, 'ROIS.pkl')

        rois_dict = {}
        masks_dict = {}
        for server_id, (_, array_view) in self.array_items.items():
            rois_dict[server_id] = array_view.ROI.saveState()
            masks_dict[server_id] = array_view.mask_ROI.saveState()

        with open(dst_path, 'wb') as f:
            cPickle.dump((rois_dict, masks_dict), f)

    def load_rois(self, path='./ROIS.pkl'):
        """Apply the saved rois on the current arrays."""

        try:
            with open(path, 'rb') as f:
                tmp = cPickle.load(f)

            if type(tmp) is tuple:
                rois_dict, masks_dict = tmp
            else:
                #
                # Support older type ROI pickle that did not
                # include the mask ROI.
                #
                rois_dict = tmp
                masks_dict = None

            for server_id, roi in rois_dict.items():
                if server_id not in self.array_items:
                    continue

                _, array_view = self.array_items[server_id]
                array_view.ROI.setState(roi)

                if masks_dict is not None:
                    array_view.mask_ROI.setState(masks_dict[server_id])

        except Exception as e:
            logging.error(
                "Failed setting rois to Arrays view:\n{}".format(
                    traceback.format_exc()))

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

        #
        # Ignore local servers.
        # TODO:
        # Add a flag that controls this behaviour.
        #
        if server_id.endswith("L"):
            logging.debug("Local server: {} ignored.".format(server_id))
            return

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
        new_server = ServerModel(server_id=server_id, client_model=self)
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
            # Display the error message.
            #
            self.logger_text = self.logger_text + \
                'Server {} raised an error:\n=========================\n{}'.format(
                    server_id, args[0])

            return

        #
        # Check if cmd failed
        #
        if status == gs.MSG_STATUS_ERROR:
            return

        #
        # Show reply in msg box.
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

        days_list = [datetime.strptime(d, "%Y_%m_%d").date() for d in days_list]
        self.days_list = sorted(set(days_list + self.days_list))

    def reply_broadcast_query(self, server_id, images_df):
        """Handle the broadcast reply of the query command."""

        logging.debug("Got reply query {}.".format(server_id))
        images_series = images_df["path"]
        images_series.name = server_id

        new_df = self.images_df.copy()
        if server_id in self.images_df.columns:
            new_df.drop(server_id, axis=1, inplace=True)
        new_df = pd.concat((new_df, images_series), axis=1)
        new_df = new_df.reindex_axis(sorted(new_df.columns), axis=1)

        self.images_df = new_df

    def reply_broadcast_seek(self, server_id, matfile, img_data):
        """Handle the broadcast reply of the seek command."""

        img_array = extractImgArray(matfile)

        #
        # Draw the camera on the map.
        #
        self.draw_camera(server_id, img_data)

        #
        # Add new array.
        #
        self.new_array_signal.emit(server_id, img_array, img_data)

    ############################################################################
    # General.
    ############################################################################
    def clear_image_df(self):
        """Reset the images dataframe."""

        self.images_df = self._default_images_df()

    def clear_arrays(self):
        """Clear the arrays panel."""

        self.clear_arrays_signal.emit()

    def updateLIDARgrid(self):
        """Update the LIDAR grid.

        The LIDAR grid is calculate in ECEF coords.
        """

        #
        # Calculate the bounding box of the cameras.
        #
        if self.grid_mode == "Auto":
            s_pts = np.array((-1000, -1000, -self.TOG))
            e_pts = np.array((1000, 1000, 0))
            for server_id, (array_model, array_view) in self.array_items.items():
                if not array_view.export_flag.checked:
                    logging.info(
                        "LIDAR Grid: Camera {} ignored.".format(server_id)
                    )
                    continue

                #
                # Convert the ECEF center of the camera to the grid center ccords.
                #
                cam_center = pymap3d.ecef2ned(
                    array_model.center[0], array_model.center[1], array_model.center[2],
                    self.latitude, self.longitude, 0)

                #
                # Accomulate tight bounding.
                #
                s_pts = np.array((s_pts, cam_center)).min(axis=0)
                e_pts = np.array((e_pts, cam_center)).max(axis=0)
        else:
            s_pts = np.array((-self.grid_length/2, -self.grid_width/2, -self.TOG))
            e_pts = np.array((self.grid_length/2, self.grid_width/2, 0))

        #
        # Create the LIDAR grid.
        # Note GRID_NED is an open grid storing the requested
        # grid resolution.
        # GRID_ECEF is just for visualization.
        #
        self.GRID_NED = (
            np.arange(s_pts[0], e_pts[0]+self.delx, self.delx),
            np.arange(s_pts[1], e_pts[1]+self.dely, self.dely),
            np.arange(s_pts[2], e_pts[2]+self.delz, self.delz)
        )

        X, Y, Z = np.meshgrid(
            np.linspace(s_pts[0], e_pts[0], 10),
            np.linspace(s_pts[1], e_pts[1], 10),
            np.linspace(s_pts[2], e_pts[2], 10),
        )

        self.GRID_ECEF = pymap3d.ned2ecef(
            X, Y, Z, self.latitude, self.longitude, self.altitude)


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

    client_model = Typed(ClientModel)

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
        print(self.status_text)

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
        self.client_model.settings_signal.emit(self)

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
        self.client_model.thumb = EImage(data=array.tostring(), format='argb32', raw_size=(width, height))

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

        img_array = extractImgArray(matfile)

        #
        # Draw the camera on the map.
        #
        self.client_model.draw_camera(self.server_id, img_data)

        #
        # Add new array.
        #
        self.client_model.new_array_signal.emit(self.server_id, img_array, img_data)

    def reply_days(self, days_list):
        """Handle the reply for days command."""

        self.days_list = [datetime.strptime(d, "%Y_%m_%d").date() for d in days_list]

    def reply_query(self, images_df):
        """Handle the reply for query command."""

        self.images_df = images_df

    def reply_seek(self, matfile, img_data):

        img_array = extractImgArray(matfile)

        #
        # Draw the camera on the map.
        #
        self.client_model.draw_camera(self.server_id, img_data)

        #
        # Add new array.
        #
        self.client_model.new_array_signal.emit(self.server_id, img_array, img_data)

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
        self.client_model.thumb = EImage(data=array.tostring(), format='argb32', raw_size=(width, height))


class ArrayModel(Atom):
    """Representation of an image."""

    resolution = Int()
    img_data = Instance(DataObj)
    fov = Float(math.pi/2)

    #
    # Epipolar line length.
    #
    line_length = Float(10000)

    #
    # Earth coords of the camera.
    #
    longitude = Float()
    latitude = Float()
    altitude = Float()

    #
    # The center of the camera in ECEF coords.
    #
    center = Tuple()

    def setEpipolar(self, x, y, N):
        """Create set of points in space.

        This points creates line of sight (LOS) points set by the
        x,y coords of the mouse click on some (this) view.

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

        print x, y

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
        pts = np.linspace(0, self.line_length, N)
        Z = -math.cos(psi) * pts
        X = math.sin(psi) * math.cos(phi) * pts
        Y = math.sin(psi) * math.sin(phi) * pts

        #
        # Calculate the LOS in ECEF coords.
        #
        LOS_pts = pymap3d.ned2ecef(
            X, Y, Z, self.latitude, self.longitude, self.altitude)

        return LOS_pts

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
        #normXY = np.linalg.norm(neu_pts[:, :2], axis=1)
        #normXYZ = np.linalg.norm(neu_pts, axis=1)
        PSI = np.arccos(neu_pts[:,2])
        PHI = np.arctan2(neu_pts[:,1], neu_pts[:,0])
        R = PSI / self.fov * self.resolution/2
        xs = R * np.sin(PHI) + self.resolution/2
        ys = R * np.cos(PHI) + self.resolution/2

        if filter_fov:
            return xs[cosPSI>0], ys[cosPSI>0]
        else:
            return xs, ys, cosPSI>0


#
# Import the enaml view.
#
with enaml.imports():
    from camera_view import Main, update_dockarea_servers, \
         new_thumbnail, critical, new_array, clear_arrays, \
         open_settings


class Controller(Atom):

    model = Typed(ClientModel)
    view = Typed(Main)

    @observe('model.servers_dict')
    def update_servers(self, change):
        if change['type'] == 'update':
            update_dockarea_servers(self.view.dock_area, self.model)

    @observe('model.new_array_signal')
    def array_signal(self, server_id, img_array, img_data):
        """This callback is called when a new array is added to the display.

        The callback creates all objects required to display the image.

        Args:
            server_id (str): ID of the server to which an array is added.
            img_array (array): New image.
            img_data (dict): Meta data of the image.
        """

        #
        # Calculate the Almucantar and PrinciplePlanes
        #
        Almucantar_coords, PrincipalPlane_coords = \
            calcSunphometerCoords(img_data, resolution=img_array.shape[0])

        #
        # Create the array model which handles the array view on the display.
        #
        server_keys = self.model.array_items.keys()
        if server_id in server_keys:
            #
            # The specific Server/Camera is already displayed. Update the array
            # model and view.
            #
            array_model, array_view = self.model.array_items[server_id]

            #
            # Update the view.
            #
            array_view.img_array = img_array
            array_view.img_data = img_data

            array_view.Almucantar_coords = Almucantar_coords
            array_view.PrincipalPlane_coords = PrincipalPlane_coords

            #
            # Update the model.
            #
            array_model.resolution = int(img_array.shape[0])
            array_model.longitude = float(img_data.longitude)
            array_model.latitude = float(img_data.latitude)
            array_model.altitude = float(img_data.altitude)
            array_model.img_data = img_data

        else:
            #
            # The specific camera is not displayed. Create it.
            #
            view_index = sorted(server_keys+[server_id]).index(server_id)

            #
            # Create the view.
            #
            array_view = new_array(
                self.view.array_views,
                server_id, img_array,
                img_data, view_index,
                Almucantar_coords,
                PrincipalPlane_coords
            )
            array_view.image_widget.observe('epipolar_signal', self.updateEpipolar)
            array_view.image_widget.observe('export_flag', self.updateExport)
            array_view.image_widget.observe('ROI_signal', self.updateROI)

            #
            # Create the model.
            #
            array_model = ArrayModel(
                resolution=int(img_array.shape[0]),
                longitude=float(img_data.longitude),
                latitude=float(img_data.latitude),
                altitude=float(img_data.altitude),
                img_data=img_data
            )

            self.model.array_items[server_id] = array_model, array_view

        #
        # Calculate the center of the camera in ECEF coords.
        #
        array_model.center = pymap3d.ned2ecef(
            0, 0, 0, array_model.latitude, array_model.longitude, array_model.altitude)

        #
        # Create the projection of the LIDAR grid on the view.
        #
        if self.model.GRID_ECEF == ():
            self.model.updateLIDARgrid()

        xs, ys = array_model.projectECEF(self.model.GRID_ECEF)
        array_view.image_widget.updateLIDARgridPts(xs=xs, ys=ys)

        #
        # Update the view of the ROI.
        # This is necessary for displaying the ROI in the map view.
        #
        array_view.image_widget._ROI_updated()

    @observe('model.settings_signal')
    def settings_signal(self, server_model):
        """Open a settings popup window."""

        open_settings(self.view, self.model, server_model)

    @observe('model.clear_arrays_signal')
    def clear_arrays_signal(self):
        clear_arrays(self.view.array_views)
        self.model.array_items = {}

    @observe('model.thumb')
    def new_thumb_popup(self, change):
        new_thumbnail(self.model.thumb)

    def updateEpipolar(self, data):
        """Handle click events on image array."""

        server_id = data['server_id']
        pos_x, pos_y = data['pos']

        clicked_model, clicked_view = self.model.array_items[server_id]

        LOS_pts = clicked_model.setEpipolar(
            pos_x, pos_y, clicked_view.epipolar_points
        )

        for k, (array_model, array_view) in self.model.array_items.items():
            if k == server_id:
                continue

            xs, ys = array_model.projectECEF(LOS_pts)

            array_view.image_widget.updateEpipolar(xs=xs, ys=ys)

    def updateROI(self, data):
        """Handle update of a server ROI."""

        server_id = data['server_id']
        pts = data['pts']
        shape = data['shape']

        self.model.updateROImesh(server_id, pts, shape)

    @observe('model.intensity_value')
    def updateIntensity(self, change):
        for _, (_, array_view) in self.model.array_items.items():
            array_view.image_widget.setIntensity(change['value'])

    def updateExport(self, *args, **kwds):
        """This function is used only for bridging."""

        #
        # Update the LIDAR grid according to new cameras setup.
        #
        self.model.updateLIDARgrid()

        #
        # Update the view of the LIDAR grid on all images.
        #
        for _, (array_model, array_view) in self.model.array_items.items():
            xs, ys = array_model.projectECEF(self.model.GRID_ECEF)
            array_view.image_widget.updateLIDARgridPts(xs=xs, ys=ys)


def main(local_mode):
    """Main doc"""

    gs.initPaths()

    import enaml
    from enaml.qt.qt_application import QtApplication

    #
    # Setup logging
    #
    CameraNetwork.initialize_logger(
        log_path='client_logs',
    )

    #
    # Instansiate the data model
    #
    client_model = ClientModel()
    client_model.start_camera_thread(local_mode)

    #
    # Start the Qt application
    #
    app = QtApplication()

    view = Main(client_model=client_model)

    controller = Controller(model=client_model, view=view)

    view.show()

    app.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the camera client application')
    parser.add_argument('--local', action='store_true', help='Run in local mode.')
    args = parser.parse_args()

    main(args.local)