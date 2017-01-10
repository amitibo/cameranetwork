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
import cPickle
from datetime import datetime
from enaml.application import deferred_call, is_main_thread
import json
import logging
import math
import os
import pymap3d
import scipy.io as sio
import StringIO
import subprocess
from threading import Thread
import traceback
from zmq.eventloop import ioloop

import CameraNetwork
from CameraNetwork import global_settings as gs
from CameraNetwork.mdp import MDP
from CameraNetwork.radiosonde import load_radiosonde
from CameraNetwork.utils import buff2dict
from CameraNetwork.utils import DataObj
from CameraNetwork.utils import sun_direction

#
# We need to import enaml.qt before matplotlib to avoid some qt errors
#
import enaml.qt
import ephem
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from enaml.image import Image as EImage

#
# I added this 'Qt4Agg' to avoid the following error:
# "TypeError: 'figure' is an unknown keyword argument"
# See:
# https://github.com/matplotlib/matplotlib/issues/3623/
#
import matplotlib
matplotlib.use('Qt4Agg')

from CameraNetwork.sunphotometer import calcAlmucantarPrinciplePlanes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def calcSunphometerCoords(img_data, resolution):
    """Calculate the Almucantar and PrinciplePlanes for a specifica datetime."""

    Almucantar_coords, PrincipalPlane_coords, _, _ = \
        calcAlmucantarPrinciplePlanes(
            latitude=img_data.latitude,
            longitude=img_data.longitude,
            capture_time=img_data.capture_time,
            img_resolution=resolution)

    #
    # Note:
    # The X, Y coords are switched as the pyqt display is Transposed to the matplotlib coords.
    #
    return Almucantar_coords[::-1, ...].T.tolist(), PrincipalPlane_coords[::-1, ...].T.tolist()


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
    
    images_df = Typed(pd.DataFrame)
    img_index = Tuple()

    def _default_images_df(self):
        """Initialize an empty data frame."""

        df = pd.DataFrame(columns=('Time', 'hdr')).set_index(['Time', 'hdr'])
        return df

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
    def open_tunnel(self, tunnel_details):
        """Open the putty client"""

        tunnel_user = tunnel_details['password']
        tunnel_port = tunnel_details['tunnel_port']
        tunnel_pw = tunnel_details['password']
        tunnel_ip = self.client_instance.proxy_params['ip']

        putty_cmd = 'putty -P {port} -pw {password} {user}@{proxy_ip}'.format(
            user=tunnel_user,
            password=tunnel_pw,
            port=tunnel_port,
            proxy_ip=tunnel_ip
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
    
    def _calcROIbounds(self, array_model, array_view):
        """Calculate bounds of ROI in array_view
        
        Useful for debug visualization.
        """
        
        #
        # Get the ROI size
        #
        roi = array_view.roi
        size = roi.state['size']

        #
        # Get the transform from the ROI to the data.
        #
        _, tr = roi.getArraySlice(array_view.img_array, array_view.image_widget.img_item)
        
        #
        # Calculate the bounds.
        #
        center = float(array_view.img_array.shape[0])/2
        pts = np.array(
            [tr.map(x, y) for x, y in \
             ((0, 0), (size.x(), 0), (0, size.y()), (size.x(), size.y()))]
        )
        pts = (pts - center) / center
        X, Y = pts[:, 1], pts[:, 0]
        bounding_phi = np.arctan2(X, Y)
        bounding_psi = array_model.fov * np.sqrt(X**2 + Y**2)        
        
        return bounding_phi, bounding_psi
        
    def reconstruct(self, lat, lon, alt):
        """Reconstruct selected regions."""

        #
        # Set the center of the axes
        #
        Xs, Ys, Zs, PHIs, PSIs, Rs, Gs, Bs, Datas = \
            {}, {}, {}, {}, {}, {}, {}, {}, {}
        for server_id, (array_model, array_view) in self.array_items.items():
            if not array_view.reconstruct_flag.checked:
                logging.info(
                    "Reconstruction: Camera {} ignored.".format(server_id)
                )
                continue

            #
            # Extract the image values at the ROI
            #
            img_array = array_view.img_array
            Rs[server_id] = array_view.image_widget.getArrayRegion(img_array[..., 0])
            Gs[server_id] = array_view.image_widget.getArrayRegion(img_array[..., 1])
            Bs[server_id] = array_view.image_widget.getArrayRegion(img_array[..., 2])

            #
            # Calculate the center of the camera.
            # Note that the coords are stored as ENU (in contrast to NED)
            #
            n, e, d = pymap3d.geodetic2ned(
                array_model.latitude, array_model.longitude, array_model.altitude,
                lat0=lat, lon0=lon, h0=alt)
            
            logging.info(
                "Saved reconstruction data of camera: {}.".format(server_id)
                )
            
            x, y, z = e, n, -d
            Xs[server_id] = np.ones_like(Rs[server_id]) * x
            Ys[server_id] = np.ones_like(Rs[server_id]) * y
            Zs[server_id] = np.ones_like(Rs[server_id]) * z

            #
            # Calculate azimuth and elevation.
            # TODO:
            # The azimuth and elevation here are calculated assuming
            # that the cameras are near. If they are far, the azimuth
            # and elevation should take into account the earth carvature.
            # I.e. relative to the center of axis the angles are rotated.
            #
            X_, Y_ = np.meshgrid(
                np.linspace(-1, 1, img_array.shape[1]),
                np.linspace(-1, 1, img_array.shape[0])
            )
    
            PHI = np.arctan2(X_, Y_)
            PSI = array_model.fov * np.sqrt(X_**2 + Y_**2)
            
            PHIs[server_id] = array_view.image_widget.getArrayRegion(PHI)
            PSIs[server_id] = array_view.image_widget.getArrayRegion(PSI)
        
            #
            # Calculate bounding coords (useful for debug visualization)
            #
            bounding_phi, bounding_psi = self._calcROIbounds(
                array_model, array_view)
            
            #
            # Extra data
            #
            sun_alt, sun_az = sun_direction(
                latitude=str(array_model.latitude),
                longitude=str(array_model.longitude),
                altitude=array_model.altitude,
                at_time=array_model.img_data.name_time)

            Datas[server_id] = \
                dict(
                    at_time=array_model.img_data.name_time,
                    sun_alt=float(sun_alt),
                    sun_az=float(sun_az),
                    x=x,
                    y=y,
                    z=z,
                    bounding_phi=bounding_phi,
                    bounding_psi=bounding_psi
                )

        #
        # Unique base path
        #
        base_path = os.path.join(
            'reconstruction',
            array_model.img_data.name_time.strftime("%Y_%m_%d_%H_%M_%S")
        )
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        for f_name, obj in zip(
            ('Rs', 'Gs', 'Bs', 'Xs', 'Ys', 'Zs', 'PHIs', 'PSIs', 'Datas'),
            (Rs, Gs, Bs, Xs, Ys, Zs, PHIs, PSIs, Datas)):
            with open(os.path.join(base_path, '{}.pkl'.format(f_name)), 'wb') as f:
                cPickle.dump(obj, f)

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
        
    def save_rois(self, base_path='.'):
        """Save the current ROIS for later use."""
        
        dst_path = os.path.join(base_path, 'ROIS.pkl')
        
        rois_dict = {}
        for server_id, (_, array_view) in self.array_items.items():
            rois_dict[server_id] = array_view.roi.saveState()

        with open(dst_path, 'wb') as f:
            cPickle.dump(rois_dict, f)
            
    def load_rois(self, path='./ROIS.pkl'):
        """Apply the saved rois on the current arrays."""
        
        try:
            with open(path, 'rb') as f:
                rois_dict = cPickle.load(f)
            
            for server_id, roi in rois_dict.items():
                if server_id not in self.array_items:
                    continue
                
                _, array_view = self.array_items[server_id]
                array_view.roi.setState(roi)
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
    def reply_broadcast_query(self, server_id, images_df):
        """Handle the broadcast reply of the query command."""

        logging.debug("Got reply query {}.".format(server_id))
        images_df = images_df.rename(index=str, columns={images_df.columns[0]: server_id})
        if server_id in self.images_df.columns:
            self.images_df.drop(server_id, axis=1, inplace=True)
        new_df = pd.concat((self.images_df, images_df), axis=1)
        new_df = new_df.reindex_axis(sorted(new_df.columns), axis=1)

        self.images_df = new_df

    def reply_broadcast_seek(self, server_id, matfile, img_data):

        img_array = np.ascontiguousarray(buff2dict(matfile)['img_array'])

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


class ServerModel(Atom):
    """The data model of the server."""

    server_id = Str()
    cmd = Str()
    reply = Str()
    reply_data = Str()
    tunnel_port = Int()
    tunnel_ip = Str()
    sunshader_figure = Value()
    extrinsic_figure = Value()
    sunshader_required_angle = Int()
    camera_settings = Dict(default=gs.CAPTURE_SETTINGS)

    client_model = Typed(ClientModel)

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

    def _default_extrinsic_figure(self):
        """Draw the default plot figure."""

        figure = Figure(figsize=(2, 1))
        ax = figure.add_subplot(111)

        return figure

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

        putty_cmd = 'putty -P {port} -pw {password} {user}@{proxy_ip}'.format(
            user=tunnel_user,
            password=tunnel_pw,
            port=tunnel_port,
            proxy_ip=tunnel_ip
        )

        subprocess.Popen(putty_cmd)

    def reply_local_ip(self, port, ip, user, pw):
        """Open the putty client"""

        putty_cmd = 'putty -P {port} -pw {password} {user}@{proxy_ip}'.format(
            user=user,
            password=pw,
            port=port,
            proxy_ip=ip
        )

        subprocess.Popen(putty_cmd)

    def reply_get_settings(self, settings):
        """Handle reply of settings."""

        #
        # Start with temp settings incase the camera is not updated with new
        # settings.
        #
        temp_settings = gs.CAPTURE_SETTINGS
        temp_settings.update(settings)
        self.camera_settings = temp_settings
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

    def reply_sunshader_scan(self, angles, saturated_array, sun_signal, required_angle):
        f = Figure(figsize=(2, 1))
        ax = f.add_subplot(111)
        ax.plot(angles, saturated_array)
        self.sunshader_figure = f
        self.sunshader_required_angle = int(required_angle)

    def reply_extrinsic(self, rotated_directions, calculated_directions, R):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            rotated_directions[:, 0], rotated_directions[:, 1], rotated_directions[:, 2],
            c='r', marker='o')
        ax.scatter(
            calculated_directions[:, 0], calculated_directions[:, 1], calculated_directions[:, 2],
            c='b', marker='^')

        #plt.ion()

        self.extrinsic_figure = fig

    def reply_array(self, matfile, img_data):

        img_array = np.ascontiguousarray(buff2dict(matfile)['img_array'])

        #
        # Add new array.
        #
        self.client_model.new_array_signal.emit(self.server_id, img_array, img_data)

    def reply_query(self, images_df):
        """Handle the reply for query command."""

        self.images_df = images_df

    def reply_seek(self, matfile, img_data):

        img_array = np.ascontiguousarray(buff2dict(matfile)['img_array'])

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
    resolution = Int()
    longitude = Float()
    latitude = Float()
    altitude = Float()
    line_length = Float(10000)
    img_data = Instance(DataObj)
    fov = Float(math.pi/2)

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

        pts = np.linspace(0, self.line_length, N)

        x = (x - self.resolution/2) / (self.resolution/2)
        y = (y - self.resolution/2) / (self.resolution/2)

        phi = math.atan2(y, x)
        psi = self.fov * math.sqrt(x**2 + y**2)

        Z = -math.cos(psi) * pts
        X = math.sin(psi) * math.cos(phi) * pts
        Y = math.sin(psi) * math.sin(phi) * pts

        LOS_pts = pymap3d.ned2ecef(
            X, Y, Z, self.latitude, self.longitude, self.altitude)

        return LOS_pts

    def queryEpipolar(self, LOS_pts):
        """Project set of LOS points to view.

        Args:
            LOS_pts (tuple of arrays): LOS points (from another view)
                in ecef coords.

        Returns:
            LOS points project to the view of this server.
        """

        X, Y, Z = pymap3d.ecef2ned(
            LOS_pts[0], LOS_pts[1], LOS_pts[2],
            self.latitude, self.longitude, self.altitude)

        epipolar_pts = np.array([X, Y, -Z]).T

        #
        # Normalize the points
        #
        epipolar_pts = \
            epipolar_pts/np.linalg.norm(epipolar_pts, axis=1).reshape(-1, 1)

        #
        # Zero points below the horizon.
        #
        cosPSI = epipolar_pts[:,2].copy()
        cosPSI[cosPSI<0] = 0

        normXY = np.linalg.norm(epipolar_pts[:, :2], axis=1)
        PSI = np.arccos(epipolar_pts[:,2])
        R = PSI / self.fov * self.resolution/2
        xs = R * epipolar_pts[:,0]/(normXY+0.00000001) + self.resolution/2
        ys = R * epipolar_pts[:,1]/(normXY+0.00000001) + self.resolution/2

        return xs[cosPSI>0], ys[cosPSI>0]


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

        #
        # Calculate the Almucantar and PrinciplePlanes
        #
        Almucantar_coords, PrincipalPlane_coords = \
            calcSunphometerCoords(img_data, resolution=img_array.shape[0])

        server_keys = self.model.array_items.keys()
        if server_id in server_keys:
            #
            # Update the array model and view
            #
            array_model, array_view = self.model.array_items[server_id]

            array_view.img_array = img_array
            array_view.img_data = img_data

            array_view.Almucantar_coords = Almucantar_coords
            array_view.PrincipalPlane_coords = PrincipalPlane_coords

            array_model.resolution = int(img_array.shape[0])
            array_model.longitude = float(img_data.longitude)
            array_model.latitude = float(img_data.latitude)
            array_model.altitude = float(img_data.altitude)
            array_model.img_data = img_data

        else:
            view_index = sorted(server_keys+[server_id]).index(server_id)

            array_view = new_array(
                self.view.array_views,
                server_id, img_array,
                img_data, view_index,
                Almucantar_coords, PrincipalPlane_coords
            )
            array_view.image_widget.observe('epipolar_signal', self.updateEpipolar)

            array_model = ArrayModel(
                resolution=int(img_array.shape[0]),
                longitude=float(img_data.longitude),
                latitude=float(img_data.latitude),
                altitude=float(img_data.altitude),
                img_data=img_data
            )

            self.model.array_items[server_id] = array_model, array_view

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

            xs, ys = array_model.queryEpipolar(LOS_pts)

            array_view.image_widget.updateEpipolar(xs=xs, ys=ys)


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

    view = Main(model=client_model)

    controller = Controller(model=client_model, view=view)

    view.show()

    app.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the camera client application')
    parser.add_argument('--local', action='store_true', help='Run in local mode.')
    args = parser.parse_args()

    main(args.local)