#!/usr/bin/env python
"""
Setup the camera instance.
This code should be run after installing the package.
"""

from __future__ import division
import CameraNetwork
import CameraNetwork.global_settings as gs


def check_camera_index(camera_id):
    
    try:
        camera_id = int(camera_id)
    except:
        print 'Wrong camera id: ' + camera_id
        raise

    if camera_id < 0:
        raise Exception('Camera id should be a positive number: ' + camera_id)

    return camera_id


def main():
    """Main doc """
    
    gs.initPaths()
    camera_data = gs.CAMERA_SETTINGS.copy()

    #
    # Get camera ID
    #
    camera_id = raw_input("Enter camera id number:")
    camera_id = check_camera_index(camera_id)
    camera_data[gs.CAMERA_IDENTITY] = camera_id

    #
    # Get cellular company
    # Note:
    # This is not needed any more as the modem is started in the rc.local file using
    # the NetworkManager.
    #
    # cell_company = None
    # while cell_company not in CELL_COMPANIES:
    #     cell_company = raw_input("Enter cellular company {companies}:".format(companies=CELL_COMPANIES))
    # camera_data[CELL_COMPANY] = cell_company
        
    #
    # Save the camera settings.
    #
    CameraNetwork.save_camera_data(
        gs.GENERAL_SETTINGS_PATH, gs.CAPTURE_SETTINGS_PATH,
        camera_settings=camera_data, capture_settings=gs.CAPTURE_SETTINGS)  

    #
    # Create necessary paths
    #
    
if __name__ == '__main__':
    main()

    
    
