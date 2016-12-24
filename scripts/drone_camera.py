#!/usr/bin/env python
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
