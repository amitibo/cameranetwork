"""Fix camera position

This script sets the camera position in all image data files according to the
current camera positions as stored in the camera settings file.
"""
from __future__ import division
import CameraNetwork.global_settings as gs
from CameraNetwork.utils import load_camera_data
import cPickle
import os
import time
import traceback


def main():
    #
    # Load the current camera location.
    #
    camera_settings, capture_settings = load_camera_data(
        gs.GENERAL_SETTINGS_PATH, gs.CAPTURE_SETTINGS_PATH
    )
    longitude = camera_settings[gs.CAMERA_LONGITUDE]
    latitude = camera_settings[gs.CAMERA_LATITUDE]
    altitude = camera_settings[gs.CAMERA_ALTITUDE]

    #
    # Collect all images data files.
    #
    data_files = []
    for root, dirs, files in os.walk(gs.CAPTURE_PATH):
        for file in files:
            if file.endswith(".pkl"):
                data_files.append(os.path.join(root, file))

    print(len(data_files), longitude, latitude, altitude)

    #
    # Fix the location in the data files and update the files.
    # Note:
    # I sleep for 1 second just to be sure not to corrupt an image
    # data file that is written by the "controller"
    #
    time.sleep(1)
    for path in data_files:
        print("Processing path: {}".format(path))
        try:
            with open(path, "rb") as f:
                data = cPickle.load(f)

            data.longitude = longitude
            data.latitude = latitude
            data.altitude = altitude

            with open(path, "wb") as f:
                cPickle.dump(data, f)
        except Exception, e:
            traceback.print_exc()


if __name__ == "__main__":
    gs.initPaths()
    main()
