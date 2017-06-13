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
"""Start a reverse tunnel to the proxy

This tunnel can be used to SSH from a client to the Camera server. Note that
this tunnel is a backup to the tunnels opened through the server api.
"""

from __future__ import division
import CameraNetwork
import CameraNetwork.global_settings as gs
import json
import logging
import argparse
import time
import os


def main():
    parser = argparse.ArgumentParser(
        description='Start the modem and setup the default tunnel')
    parser.add_argument(
        '--skip_tunnel',
        action='store_true',
        help='Skip starting the default tunnel')
    parser.add_argument(
        '--log_level',
        default='INFO',
        help='Set the log level (possible values: info, debug, ...)')
    args = parser.parse_args()

    #
    # Initialize paths.
    #
    gs.initPaths()

    #
    # Initialize the logger
    #
    CameraNetwork.initialize_logger(
        log_path=gs.DEFAULT_LOG_FOLDER,
        log_level=args.log_level,
        postfix='_tunnel')

    #
    # Set the autossh debug environment variable
    #
    os.environ["AUTOSSH_DEBUG"] = "1"

    camera_settings, capture_settings = CameraNetwork.load_camera_data(
        gs.GENERAL_SETTINGS_PATH, gs.CAPTURE_SETTINGS_PATH
    )
    identity = str(camera_settings[gs.CAMERA_IDENTITY])

    #
    # Start the tunnel
    #
    if not args.skip_tunnel:
        logging.info('starting tunnel')

        #
        # Loop till network reached.
        #
        network_reached = False
        failures_cnt = 0
        while not network_reached:
            try:
                proxy_params = CameraNetwork.retrieve_proxy_parameters()
                network_reached = True
            except:
                #
                # There is probably some problem with the internet connection.
                #
                failures_cnt += 1

                if failures_cnt > camera_settings[gs.INTERNET_FAILURE_THRESH]:
                    logging.error('Failed to connect 3G modem. Will reboot...')
                    os.system('sudo reboot')

                logging.error(
                    'Failed to retrieve proxy parameters. will sleep and try again later.')
                time.sleep(gs.WD_TEST_INTERNET_PERIOD)

        _, tunnel_port = CameraNetwork.setup_reverse_ssh_tunnel(**proxy_params)

        #
        # Upload the tunnel port to the proxy.
        #
        file_name = "tunnel_port_{}.txt".format(identity)

        with open(file_name, 'w') as f:
            json.dump({"password": "odroid", "tunnel_port": tunnel_port}, f)

        CameraNetwork.upload_file_to_proxy(
            src_path=os.path.abspath(file_name),
            dst_path=file_name,
            **proxy_params
        )

    #
    # Internet watchdog loop.
    #
    failures_cnt = 0
    while True:
        time.sleep(gs.WD_TEST_INTERNET_PERIOD)

        #
        # Check internet connection every 2 WD_TEST_INTERNET_PERIOD secs.
        # If fails for more than WS_INTERNET_FAILURE_TRHESH consecutive times,
        # do a restart of the system.
        #
        if not CameraNetwork.check_connection():
            failures_cnt += 1
            logging.debug('Internet watchdog: failure number: %d.' % failures_cnt)
            if failures_cnt > camera_settings[gs.INTERNET_FAILURE_THRESH]:
                logging.error('Failed to connect 3G modem. Will reboot...')
                os.system('sudo reboot')
        else:
            logging.debug('Internet watchdog: succeed.')
            failures_cnt = 0


if __name__ == '__main__':
    main()
