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

    
    
