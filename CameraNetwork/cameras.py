from __future__ import division
import CameraNetwork.global_settings as gs
from CameraNetwork.utils import CameraException
import logging
import numpy as np
import os


DEFAULT_IDS_SETTINGS = {
    "exposure_us": 1000,
    "gain_db": 0,
    "gain_boost": True,
    "color_mode": gs.COLOR_RGB
}
#
# On the Odroid U3 pixelclock above 30 starts throwing USB errors.
#
IDS_MAX_PIXEL_CLOCK = 25


class IDSCamera(object):
    """A wrapper for the IDS cameras"""

    def __init__(self, *params, **kwds):

        try:
            import ids
            globals()['ids'] = ids

            self.capture_dev = ids.Camera(nummem=10)

            #
            # Turn off white balance.
            #
            self.capture_dev.auto_white_balance = False

        except Exception as e:
            logging.exception(
                'Error while initializing the camera:\n{}'.format(repr(e)))
            raise CameraException('Failed initializing the camera')

        #
        # set the color mode map
        #
        self._ids_color_map = {
            gs.COLOR_RAW: ids.ids_core.COLOR_BAYER_8,
            gs.COLOR_RGB: ids.ids_core.COLOR_RGB8
        }

        #
        # Last exposure used for checking if there is a need
        # to throw the next image.
        #
        self._last_settings = None

    def close(self):
        """Close the camera"""

        if self.capture_dev is not None:
            logging.info("Closing the camera.")
            try:
                self.capture_dev.close()
            except Exception as e:
                #
                # Some times the close() method can throw an exception
                # for example when the device in already closed.
                #
                logging.error(
                    'Closing the camera raised an exception: {}'.format(e))
            del self.capture_dev
            self.capture_dev = None

    def __del__(self):
        """Destructor"""

        if self.capture_dev is not None:
            self.close()

    @property
    def info(self):
        """Return information about the camera in a form of a dict"""

        return self.capture_dev.info.copy()

    def small_size(self):
        """Set small frame size."""

        logging.debug('Setting camera to small size')
        self.capture_dev.subsampling = \
            ids.ids_core.SUBSAMPLING_4X_HORIZONTAL | ids.ids_core.SUBSAMPLING_4X_VERTICAL
        logging.debug('Free up memory')
        self.capture_dev.free_all()
        self.capture_dev.width = 400
        self.capture_dev.height = 300
        logging.debug('Allocating memory')
        self.capture_dev._allocate_memory()
        _ = self.capture({"exposure_us": 50})

    def large_size(self):
        """Set large frame size."""

        if self.capture_dev.subsampling == ids.ids_core.SUBSAMPLING_DISABLE:
            logging.debug('Camera is already set to large size.')
            return

        logging.debug('Setting camera to large size')
        self.capture_dev.subsampling = ids.ids_core.SUBSAMPLING_DISABLE
        logging.debug('Free up memory')
        self.capture_dev.free_all()
        self.capture_dev.width = 1600
        self.capture_dev.height = 1200
        logging.debug('Allocating memory')
        self.capture_dev._allocate_memory()
        _ = self.capture({"exposure_us": 50})

    def update_settings(self, new_settings={}):
        """Update capture settings of the camera.

        The function returns False if the settings
        didn't change from last capture.
        """

        update_settings = False

        if self._last_settings is None:
            update_settings = True
            self._last_settings = DEFAULT_IDS_SETTINGS.copy()
        else:
            for k, v in new_settings.items():
                if self._last_settings[k] != v:
                    update_settings = True
                    break

        if not update_settings:
            return False

        #
        # Calculate the new settings.
        # Note:
        # No need to explicitly update the self._last_settings
        #
        settings = self._last_settings
        settings.update(new_settings)

        #
        # Capture the image
        #
        if settings["exposure_us"] is None:
            #
            # Set pixelclock to minimum to enable long exposure times.
            #
            self.capture_dev.pixelclock = self.capture_dev.pixelclock_range[0]
            self.capture_dev.framerate = 100
            self.capture_dev.auto_exposure = True
        else:
            #
            # Set pixelclock. These values are empiric.
            #
            if settings["exposure_us"] > 1000000:
                #
                # Set pixel rate to minimum to allow long exposure times.
                #
                self.capture_dev.pixelclock = \
                    self.capture_dev.pixelclock_range[0]
            else:
                #
                # Set pixel rate to maximum to allow short exposure times.
                #
                self.capture_dev.pixelclock = IDS_MAX_PIXEL_CLOCK

            logging.info("Setting pixel clock to {}.".format(
                self.capture_dev.pixelclock))

            self.capture_dev.framerate = min(100, 1e6 / settings["exposure_us"])
            self.capture_dev.auto_exposure = False
            self.capture_dev.exposure = settings["exposure_us"] * 1e-3

        #
        # Set the gain boost
        #
        self.capture_dev.gain_boost = settings["gain_boost"]

        #
        # Set the gain
        #
        if settings["gain_db"] is None:
            gain_db = 0
            logging.debug('The IDS wrapper does not support automatic gain')
        else:
            gain_db = settings["gain_db"]

        self.capture_dev.gain = gain_db

        #
        # Set the color_mode mode.
        #
        self.capture_dev.color_mode = self._ids_color_map[settings["color_mode"]]

        return True

    def capture(self, settings, frames_num=1):

        if self.capture_dev is None:
            raise Exception('Failed to init the capture device (in class constructor).')

        try:
            if self.update_settings(settings):
                #
                # Settings changed. Throw away first image.
                #
                self.capture_dev.continuous_capture = True
                _ = self.capture_dev.next()
                self.capture_dev.continuous_capture = False

            #
            # Capture the required number of frames.
            #
            img_arrays = []
            self.capture_dev.continuous_capture = True
            for i in range(frames_num):
                logging.debug('Capturing frame {}.'.format(i))
                img_array, meta_data = self.capture_dev.next()
                img_arrays.append(img_array)
                logging.debug('Finished capturing frame {}.'.format(i))
        finally:
            self.capture_dev.continuous_capture = False

        #
        # Concatenate the frames.
        #
        img_arrays = np.concatenate(
            [np.expand_dims(a, axis=a.ndim) for a in img_arrays],
            axis=img_arrays[0].ndim
        )
        img_arrays = np.squeeze(img_arrays)

        return img_arrays, self.capture_dev.exposure * 1e3, self.capture_dev.gain
