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
