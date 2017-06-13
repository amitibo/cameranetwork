#!/usr/bin/env python
##
## Copyright (C) 2017, Amit Aides, all rights reserved.
## 
## This file is part of Camera Network
## (see https://bitbucket.org/amitibo/cameranetwork_git).
## 
## Redistribution and use in source and binary forms, with or without modification,
## are permitted provided that the following conditions are met:
## 
## 1)  The software is provided under the terms of this license strictly for
##     academic, non-commercial, not-for-profit purposes.
## 2)  Redistributions of source code must retain the above copyright notice, this
##     list of conditions (license) and the following disclaimer.
## 3)  Redistributions in binary form must reproduce the above copyright notice,
##     this list of conditions (license) and the following disclaimer in the
##     documentation and/or other materials provided with the distribution.
## 4)  The name of the author may not be used to endorse or promote products derived
##     from this software without specific prior written permission.
## 5)  As this software depends on other libraries, the user must adhere to and keep
##     in place any licensing terms of those libraries.
## 6)  Any publications arising from the use of this software, including but not
##     limited to academic journal and conference publications, technical reports and
##     manuals, must cite the following works:
##     Dmitry Veikherman, Amit Aides, Yoav Y. Schechner and Aviad Levis, "Clouds in The Cloud" Proc. ACCV, pp. 659-674 (2014).
## 
## THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
## WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
## MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
## EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
## INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
## BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
## OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
## ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.##
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



