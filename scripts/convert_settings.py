#!/usr/bin/env python
"""
Update the camera settings from old to new format.
"""

from __future__ import division
import CameraNetwork
import CameraNetwork.global_settings as gs
import argparse


def main():
    """Main doc """

    #
    # Load the current settings.
    #
    camera_settings, capture_settings = CameraNetwork.load_camera_data(
        gs.GENERAL_SETTINGS_PATH, gs.CAPTURE_SETTINGS_PATH
    )

    #
    # Update the settings to new format.
    #
    if gs.CAMERA_LATITUDE in camera_settings:
        print("Settings already update. Nothing to do.")
        return

    for field in (gs.SUNSHADER_MIN_ANGLE, gs.SUNSHADER_MAX_ANGLE,
                  gs.CAMERA_LONGITUDE, gs.CAMERA_LATITUDE, gs.CAMERA_ALTITUDE,
                  gs.INTERNET_FAILURE_THRESH):
        camera_settings[field] = capture_settings[field]
        del capture_settings[field]

    capture_settings[gs.DAY_SETTINGS][gs.LOOP_DELAY] = \
        capture_settings[gs.LOOP_DELAY]
    capture_settings[gs.NIGHT_SETTINGS][gs.LOOP_DELAY] = \
        gs.CAPTURE_SETTINGS[gs.NIGHT_SETTINGS][gs.LOOP_DELAY]

    del capture_settings[gs.LOOP_DELAY]

    #
    # Save the camera settings.
    #
    CameraNetwork.save_camera_data(
        gs.GENERAL_SETTINGS_PATH, gs.CAPTURE_SETTINGS_PATH,
        camera_settings=camera_settings, capture_settings=capture_settings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update the camera settings from old to new format.')
    parser.add_argument('--local_path', type=str, default=None, help='If set, the script will use the given path as home folder.')
    args = parser.parse_args()

    gs.initPaths(args.local_path)

    main()



