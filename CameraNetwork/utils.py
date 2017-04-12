#!/usr/bin/env python

from __future__ import division
import CameraNetwork.global_settings as gs
from CameraNetwork.transformation_matrices import euler_matrix
import copy
import cPickle
from datetime import datetime
from datetime import timedelta
import ephem
import glob
import json
import random
import logging
import logging.handlers
import math
import numpy as np
import os
import pandas as pd
import platform
import scipy.io as sio
from sklearn.base import BaseEstimator
from sklearn import linear_model
import StringIO
import subprocess
from tornado import gen
import traceback
from zmq.eventloop.ioloop import ZMQIOLoop


__all__ = [
    'DataObj',
    'sync_time',
    'save_camera_data',
    'load_camera_data',
    'initialize_logger',
    'logger_configurer',
    'setup_logging',
    'setup_reverse_ssh_tunnel',
    'upload_file_to_proxy',
    'identify_server',
    'sbp_run',
    'cmd_callback',
    'handler',
    'handler_no_answer',
    'find_centroid',
    'mean_with_outliers',
    'find_camera_orientation_ransac',
    'dict2buff',
    'buff2dict',
    'name_time',
    'object_direction',
    'sun_direction',
    'obj'
]

THRESHOLD_MINUS = 2


class RestartException(Exception):
    pass


class CameraException(Exception):
    pass


class obj(object):
    pass


class DataObj(object):

    def __init__(self, **kwds):
        self.update(**kwds)

    def update(self, **kwds):
        self.__dict__.update(kwds)


def name_time(time_object=None):
    """Create path names form datetime object."""

    if time_object is None:
        time_object = datetime.utcnow()

    #
    # Prepare the capture base name.
    #
    base_path = os.path.join(
        gs.CAPTURE_PATH,
        time_object.strftime("%Y_%m_%d")
    )
    base_name = '{time}_{formated_time}'.format(
        time=(time_object - datetime(1970, 1, 1)).total_seconds(),
        formated_time=time_object.strftime("%Y_%m_%d_%H_%M_%S")
    )

    return time_object, base_path, base_name


def identify_server():
    """Identify the server/camera the code is running on"""

    try:
        if os.uname()[1] == 'raspberrypi':
            return gs.PI_SERVER
        elif os.uname()[1] == 'odroid':
            return gs.ODROID_SERVER
        else:
            raise
    except:
        raise Exception('The system is either windows (not a camera server) or an unkown os')


def ispi():
    """Check whether the setup is running on raspberrypi"""

    return hasattr(os, 'uname') and os.uname()[1]=='raspberrypi'


#----------------------------------------------------------------------
def setup_reverse_ssh_tunnel(
    ip,
    user,
    local_port=22,
    tunnel_port=22220,
    autossh_monitor_port=20000,
    ssh_cmd=gs.REVERSE_AUTOSSH_CMD,
    **kwds
    ):
    """Create the (reverse) ssh tunnel from camera to proxy server.

    Args:
        ip (str) : SERVER_IP of the proxy server.
        user (str) : User name to log on the proxy server.
        local_port (int) : Local port on which to connect to the proxy server.
        tunnel_port (int) : Tunnel port (on the remote server).
    """

    autossh_monitor_port += random.randrange(1000)
    tunnel_port += random.randrange(1000)

    _tunnel_cmd = ssh_cmd.format(
        autossh_monitor_port=autossh_monitor_port,
        server_ip=ip,
        server_user=user,
        local_port=local_port,
        tunnel_port=tunnel_port,
        identity_file=gs.IDENTITY_FILE
    )
    _tunnel_msg = gs.TUNNEL_DESCRIPTION.format(
        server_ip=ip,
        server_user=user,
        local_port=local_port,
        tunnel_port=tunnel_port,
    )

    logging.debug('Starting the ssh tunnel with the cmd: %s' % _tunnel_cmd)
    logging.info('Starting the ssh tunnel: %s' % _tunnel_msg)
    tunnel_proc = subprocess.Popen(
        _tunnel_cmd,
        universal_newlines=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return tunnel_proc, tunnel_port


#----------------------------------------------------------------------
def upload_file_to_proxy(
    ip,
    user,
    src_path,
    dst_path,
    scp_cmd=gs.SCP_CMD,
    **kwds
    ):
    """
    Upload a file from the camera to proxy server.

    Args:
        ip (str) : SERVER_IP of the proxy server.
        user (str) : User name to log on the proxy server.
        src_path (str) : Path to uploaded file.
        dst_path (str) : Path to copy to on the remote server.
    """

    _scp_cmd = scp_cmd.format(
        server_ip=ip,
        server_user=user,
        src_path=src_path,
        dst_path=dst_path,
        identity_file=gs.IDENTITY_FILE
    )

    logging.debug('Uploading file to server: %s' % _scp_cmd)
    scp_proc = subprocess.Popen(_scp_cmd, universal_newlines=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return scp_proc


def save_camera_data(
    general_settings_path,
    capture_settings_path,
    camera_settings=None,
    capture_settings=None):

    if camera_settings is not None:
        with open(general_settings_path, 'wb') as f:
            json.dump(camera_settings, f, sort_keys=True, indent=4, separators=(',', ': '))
    if capture_settings is not None:
        with open(capture_settings_path, 'wb') as f:
            json.dump(capture_settings, f, sort_keys=True, indent=4, separators=(',', ': '))


def load_camera_data(general_settings_path, capture_settings_path):

    with open(general_settings_path, 'rb') as f:
        camera_settings = json.load(f)

    capture_settings = copy.copy(gs.CAPTURE_SETTINGS)
    try:
        with open(capture_settings_path, 'rb') as f:
            capture_settings.update(json.load(f))
    except Exception as e:
        logging.error("Failed loading capture settings: {}\n{}".format(
            repr(e), traceback.format_exc())
        )
    return camera_settings, capture_settings


def sync_time():
    os.system('sudo service ntp stop')
    os.system('sudo ntpdate pool.ntp.org')
    os.system('sudo service ntp start')


def initialize_logger(log_path=None, log_level=logging.INFO, postfix=''):
    """Initialize the logger. Single process version. Logs both to file and stdout."""

    #
    # Get the log level
    #
    if type(log_level) == str:
        log_level = getattr(logging, log_level.upper(), None)
        if not isinstance(log_level, int):
            raise ValueError('Invalid log level: %s' % log_level)

    #
    # Setup the logger
    #
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    logger = logging.getLogger()
    logger.setLevel(log_level)

    #
    # create console handler.
    #
    handler = logging.StreamHandler()
    handler.setFormatter(logFormatter)
    logger.addHandler(handler)

    if log_path is None:
        return

    #
    # Create a unique name for the log file.
    #
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    dt = datetime.now()
    if dt.year < 2014:
        #
        # There is some problem with the clock
        #
        prev_logs = sorted(glob.glob(os.path.join(log_path, '*{postfix}.txt'.format(postfix=postfix))))
        if len(prev_logs) == 0:
            filename = datetime.now().strftime("cameralog_%y%m%d_%H%M%S{postfix}.txt".format(postfix=postfix))
            log_path = os.path.join(log_path, filename)
        else:
            log_path = prev_logs[-1][:-4]+'p.txt'
    else:
        filename = datetime.now().strftime("cameralog_%y%m%d_%H%M%S{postfix}.txt".format(postfix=postfix))
        log_path = os.path.join(log_path, filename)

    #
    # create error file handler and set level to error
    #
    handler = logging.FileHandler(log_path, "w", encoding=None, delay="true")
    handler.setFormatter(logFormatter)
    logger.addHandler(handler)


###################################################################################
# Logging to single file from multiple process. Example taken from:
# https://docs.python.org/dev/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
# and
# http://plumberjack.blogspot.co.il/2010/09/using-logging-with-multiprocessing.html
#
class QueueHandler(logging.Handler):
    """
    This is a logging handler which sends events to a multiprocessing queue.

    The plan is to add it to Python 3.2, but this can be copy pasted into
    user code for use with earlier Python versions.
    """

    def __init__(self, queue):
        """Initialise an instance, using the passed queue.
        """
        logging.Handler.__init__(self)
        self.queue = queue

    def emit(self, record):
        """
        Emit a record.

        Writes the LogRecord to the queue.
        """
        try:
            ei = record.exc_info
            if ei:
                dummy = self.format(record) # just to get traceback text into record.exc_text
                record.exc_info = None  # not needed any more
            self.queue.put_nowait(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def process_logger(log_queue, log_path, log_level):

    #
    # Create a unique name for the log file.
    #
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    filename = datetime.now().strftime("cameralog_%y%m%d_%H%M%S.txt")
    log_path = os.path.join(log_path, filename)

    #
    # Setup the logger
    #
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    multiprocess_logger = logging.getLogger()
    multiprocess_logger.setLevel(log_level)

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    multiprocess_logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    multiprocess_logger.addHandler(consoleHandler)

    while True:
        try:
            record = log_queue.get()
            if record is None: # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record) # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            traceback.print_exc(file=sys.stderr)


def logger_configurer(log_queue, log_level=logging.INFO):
    """The worker configuration is done at the start of the worker process run.
    Note that on Windows you can't rely on fork semantics, so each process
    will run the logging configuration code when it starts.
    """

    h = QueueHandler(log_queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(log_level)


def setup_logging(log_path, log_level):
    """Setup logging. The logging is done both to a file and to console.
    """
    import multiprocessing

    log_queue = multiprocessing.Queue(-1)

    listener = multiprocessing.Process(
        target=process_logger,
        args=(log_queue, log_path, log_level)
    )
    listener.start()

    return listener, log_queue


def sbp_run(command, shell=False, working_directory=None):
    """Shortcut for running a command on the shell.
    """

    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        cwd=working_directory
    )

    output_stream, error_stream = p.communicate()
    return output_stream, error_stream


def cmd_callback(f):
    """Decorator that creates controller callbacks."""

    @gen.coroutine
    def callback_wrapper(self, future, *args, **kwds):
        try:
            results = yield f(self, *args, **kwds)
            future.set_result(results)
        except Exception as e:
            future.set_exception(e)

    return callback_wrapper


def handler(f):
    """Decorator that creates message handlers."""

    def handle_wrapper(*args, **kwds):
        try:
            result = f(*args, **kwds)

            #
            # Contruct the result
            #
            if result is None:
                result = (), {}
            elif type(result) == dict:
                result = (), result
            elif type(result) in (tuple, list) and \
                 len(result) > 1 and \
                 type(result[1]) is not dict:
                result = result, {}

            answer = MSG_STATUS_OK, result[0], result[1]
        except Exception:
            answer = MSG_STATUS_ERROR, [
                'Calling the cmd handler caused an error:\n{}'.format(traceback.format_exc())
                ], {}

        return answer

    return handle_wrapper


def handler_no_answer(f):
    """Decorator that creates message handlers that don't reply."""

    def handle_wrapper(*args, **kwds):
        answer = None
        try:
            f(*args, **kwds)
        except Exception:
            return MSG_STATUS_ERROR, [
                'Calling the cmd handler caused an error:\n{}'.format(traceback.format_exc())
                ], {}

    return handle_wrapper


def sun_direction(
        latitude='32.8167',
        longitude='34.9833',
        altitude=230,
        at_time=None):
    """Calculate the current altitude of the sun.

    Default latitude and longitude given for haifa:
    Haifa. 32.8167 N, 34.9833 E
    """

    if at_time is None:
        at_time = datetime.utcnow()

    observer = ephem.Observer()
    observer.lat, observer.long, observer.elevation, observer.date = \
        str(latitude), str(longitude), altitude, at_time

    logging.debug("{} {} {} {}".format(latitude, longitude, altitude, at_time))
    sun = ephem.Sun(observer)
    logging.debug("Sun altitude {}, azimuth {} at time {}".format(
        sun.alt, sun.az, at_time))
    return sun.alt, sun.az


def object_direction(
    celestial_class,
    date,
    latitude,
    longitude,
    altitude,
    UTC_plus=0
    ):
    """
    Calculate a direction to a celestial object.
    Default latitude and longitude given for haifa:
    Haifa. 32.8167 N, 34.9833 E
    """

    delta_time = timedelta(seconds=3600*UTC_plus)

    observer = ephem.Observer()
    observer.lat, observer.long, observer.elevation, observer.date = \
        str(latitude), str(longitude), altitude, date - delta_time

    cel_obj = celestial_class(observer)

    direction = (
        math.cos(cel_obj.alt)*math.cos(cel_obj.az),
        -math.cos(cel_obj.alt)*math.sin(cel_obj.az),
        math.sin(cel_obj.alt)
    )

    return np.array(direction)


def find_centroid(img, minus_level=THRESHOLD_MINUS):
    """Find the centroid of the strongest pixels in an image.

    Useful for finding the sun.
    """

    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, np.max(gray)-minus_level, 255, cv2.THRESH_BINARY)[1]

    moments = cv2.moments(thresh)
    centroid = (moments['m10']/np.sum(thresh), moments['m01']/np.sum(thresh))

    return centroid


def f(angles, calculated_directions, measured_directions):

    #
    # Calculate rotation matrix
    #
    M = euler_matrix(angles[0], angles[1], angles[2])[:3, :3]

    #
    # Rotate points
    #
    rotated_directions = np.dot(measured_directions, M.T)

    #
    # Calculate distance
    #
    ret = np.linalg.norm(calculated_directions - rotated_directions)

    return ret


def find_camera_orientation(calculated_directions, measured_directions):
    """
    Find the rotation of the camera based on the coordinates of a celestail object
    The input is two sets. The first is x,y image coordinates of the object (taken
    from several images). The second set is the azimuth/altitude coordinates of the
    celestial object (in Horizontal coordinate system).
    The output is the rotation matrix of the camera. The rotation matrix is converts
    between world coordinates to camera coordinates, where the world coordinates
    are centered in camera, z is in the zenith and the x-axis points to the North.
    The inner calibration of the camera is given as a function that converts
    between the image coordinates and the camera coordinates.

    Args:
        calculated_directions (array like): The reference direction of the celestial
            object. Given as an nx3 matrix of [x, y, z] on the unit hemisphere.
        measured_directions (array like): The measured directions of the celestial
            objects. Given as an nx3 matrix of [x, y, z] on the unit hemisphere.

    Returns:
        Euler angles for rotating the measured directions to match the calculated directions.
    """
    from scipy import optimize

    angles = optimize.fmin(f, x0=(0, 0, 0), args=(calculated_directions, measured_directions), xtol=1e-6, ftol=1e-6)

    return angles


class ExtrinsicModel(BaseEstimator):
    """Model for extrinsic rotation"""

    def __init__(self, ai=0, aj=0, ak=0):
        self.ai = ai
        self.aj = aj
        self.ak = ak

    def fit(self, X, y):
        self.ai, self.aj, self.ak = find_camera_orientation(y, X)
        return self

    def score(self, X, y):
        return f((self.ai, self.aj, self.ak), y, X)

    def predict(self, X):
        #
        # Calculate rotation matrix
        #
        M = euler_matrix(self.ai, self.aj, self.ak)[:3, :3]

        #
        # Rotate points
        #
        rotated_directions = np.dot(X, M.T)

        return rotated_directions


def find_camera_orientation_ransac(
    calculated_directions,
    measured_directions,
    residual_threshold):
    """
    Find the rotation of the camera based on the coordinates of a celestail object
    The input is two sets. The first is x,y image coordinates of the object (taken
    from several images). The second set is the azimuth/altitude coordinates of the
    celestial object (in Horizontal coordinate system).
    The output is the rotation matrix of the camera. The rotation matrix is converts
    between world coordinates to camera coordinates, where the world coordinates
    are centered in camera, z is in the zenith and the x-axis points to the North.
    The inner calibration of the camera is given as a function that converts
    between the image coordinates and the camera coordinates.
    Uses Ransac to filter outliers.

    Args:
        calculated_directions (array like): The reference direction of the celestial
            object. Given as an nx3 matrix of [x, y, z] on the unit hemisphere.
        measured_directions (array like): The measured directions of the celestial
            objects. Given as an nx3 matrix of [x, y, z] on the unit hemisphere.
        residual_threshold (float): Residual threshold used by the RANSAC regressor.

    Returns:
        Euler angles for rotating the measured directions to match the calculated directions.

    """

    model_ransac = linear_model.RANSACRegressor(
        ExtrinsicModel(), random_state=0, residual_threshold=residual_threshold)
    model_ransac.fit(measured_directions, calculated_directions)

    rotated_directions = model_ransac.predict(measured_directions)

    #
    # A hack to get the Rotation matrix
    #
    R = model_ransac.predict(np.eye(3)).T

    return R, rotated_directions


def mean_with_outliers(data, thresh_ratio=2):
    """Calculate mean excluding outliers."""

    mean = np.mean(data, axis=0)
    norm = np.linalg.norm(data-mean, axis=1)
    thresh = np.mean(norm)
    indices = norm < (thresh * thresh_ratio)
    mean = np.mean(data[indices], axis=0)

    return mean, indices


def dict2buff(d, do_compression=True):
    """Saves a dict as mat file in a string buffer."""

    f = StringIO.StringIO()
    sio.savemat(f, d, do_compression=do_compression)

    return f.getvalue()


def buff2dict(buff):
    """Convert a mat file in the form of a string buffer to a dict."""

    f = StringIO.StringIO(buff)
    d = sio.loadmat(f)

    return d


def safe_make_dirs(path):
    """Safely create path"""

    if os.path.exists(path):
        return

    os.makedirs(path)


def extractImgArray(matfile):
    """Extract the image from matfile"""

    #
    # This function is used in the GUI.
    # I am not sure that PIL is installed the same on the odroid.
    # Therefore I import Image from here inside the function.
    #
    from PIL import Image

    data = buff2dict(matfile)
    img_array = data["img_array"]

    #
    # This if handles both the case where "jpeg" type
    # is int (quality) and bool.
    #
    if data["jpeg"]:
        buff = StringIO.StringIO(img_array.tostring())
        img = Image.open(buff)
        width, height = img.size
        array = np.array(img.getdata(), np.uint8)

        #
        # Handle gray scale image
        #
        if array.ndim == 1:
            array.shape = (-1, 1)
            array = np.hstack((array, array, array))

        img_array = array.reshape(height, width, 3)
    else:
        img_array = np.ascontiguousarray(img_array)

    return img_array


def getImagesDF(query_date, force=False):
    """Get dataframe of images captures at a specific date.

    Args:
        query_date (datetime object): Day to query.
        force (bool, optional): Force the recreation of the database.

    Returns:
        Database of images in the form of a pandas dataframe.
    """

    base_path = os.path.join(
        gs.CAPTURE_PATH, query_date.strftime("%Y_%m_%d"))

    if not os.path.isdir(base_path):
        raise Exception('Non existing day: {}'.format(base_path))

    image_list = sorted(glob.glob(os.path.join(base_path, '*.mat')))

    #
    # Check if there is a valid database.
    #
    database_path = os.path.join(base_path, "database.pkl")
    if os.path.exists(database_path) and not force:
        df = pd.read_pickle(database_path)

        if df.shape[0] == len(image_list):
            return df

    datetimes = []
    hdrs = []
    alts = []
    lons = []
    lats = []
    sns = []
    for image_path in image_list:
        path = os.path.splitext(image_path)[0]

        #
        # Parse the time and exposure
        #
        tmp_parts = os.path.split(path)[-1].split('_')
        datetimes.append(datetime(*[int(i) for i in tmp_parts[1:-1]]))
        hdrs.append(tmp_parts[-1])

        try:
            with open("{}.pkl".format(path), "rb") as f:
                data = cPickle.load(f)

            alts.append(data.altitude)
            lons.append(data.longitude)
            lats.append(data.latitude)
            sns.append(data.camera_info["serial_num"])
        except:
            logging.error("Failed parsing data file: {}\n{}".format(
                "{}.pkl".format(path), traceback.format_exc())
            )
            alts.append(None)
            lons.append(None)
            lats.append(None)
            sns.append(None)

    new_df = pd.DataFrame(
        data=dict(
            Time=datetimes,
            hdr=hdrs,
            path=image_list,
            longitude=lons,
            latitude=lats,
            altitude=alts,
            serial_num=sns
            ),
        columns=('Time', 'hdr', 'path', "longitude", "latitude", "altitude", "serial_num")
        ).set_index(['Time', 'hdr'])

    #
    # Save the new database
    #
    pd.to_pickle(new_df, database_path)

    return new_df


class PuritanicalIOLoop(ZMQIOLoop):
    """A loop that quits when it encounters an Exception.
    """

    def handle_callback_exception(self, callback):
        exc_type, exc_value, tb = sys.exc_info()
        raise exc_value


IOLoop = PuritanicalIOLoop


if __name__ == '__main__':
    pass
