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
This script statrts the camera of the drone in a loop.
"""

#from droneapi.lib import VehicleMode
#from pymavlink import mavutil
import ids
from datetime import datetime
from CameraNetwork import sync_time
import json
import os


BASE_FOLDER = '/home/odroid/test_imgs'

def main():
    #
    # Sync the clock.
    #
    sync_time()

    #
    # get our vehicle
    #
    #api = local_connect()
    #vehicle = api.get_vehicles()[0]
 
    #
    # Get the camera
    #
    dev = ids.Camera()
    dev.continuous_capture = False
 
    #
    # Configure the camera
    #
    #dev.pixelclock = dev.pixelclock_range[1]
    dev.auto_white_balance = False
    dev.color_mode = ids.ids_core.COLOR_RGB8
    exposure_us = 100000
    dev.auto_exposure = False
    dev.exposure = exposure_us * 1e-3
    dev.gain = 0

    #
    # Start capturing images
    #
    dev.continuous_capture = True
    while True:
        print 'Capturing Image'
        t = datetime.now()
        base_name = os.path.join(
            BASE_FOLDER,
            t.strftime("/home/odroid/test_imgs/%Y%m%d_%H:%M:%S.%f")
        )
        img_name = base_name+'.jpg'
        data_name = base_name+'.txt'

        dev.next_save(img_name)

        data = {
            "img_path": img_name,
            "Time": t.strftime("%Y%m%d_%H:%M:%S.%f"),
            "Exposure": dev.exposure,
            #"location": str(vehicle.location)
        }

        with open(data_name, 'w') as f:
            json.dump(data, f)

    dev.continuous_capture = False


main()
