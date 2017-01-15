"""
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
    parser = argparse.ArgumentParser(description='Start the modem and setup the default tunnel')
    parser.add_argument('--skip_tunnel', action='store_true', help='Skip starting the default tunnel')
    parser.add_argument('--log_level', default='INFO', help='Set the log level (possible values: info, debug, ...)')
    args = parser.parse_args()

    #
    # Initialize paths.
    #
    gs.initPaths()

    #
    # Initialize the logger
    #
    CameraNetwork.initialize_logger(log_path=gs.DEFAULT_LOG_FOLDER, log_level=args.log_level, postfix='_tunnel')

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

        network_reached = False
        #
        # Loop till network reached.
        #
        failures_num = 0
        while not network_reached:
            try:
                proxy_params = CameraNetwork.retrieve_proxy_parameters()
                network_reached = True
            except:
                #
                # There is probably some problem with the internet connection.
                #
                failures_num += 1

                if failures_num > camera_settings[gs.INTERNET_FAILURE_THRESH]:
                    logging.error('Failed to connect 3G modem. Will reboot...')
                    os.system('sudo reboot')

                logging.error('Failed to retrieve proxy parameters. will sleep and try again later.')
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
    # Check internet connection every 2 WD_TEST_INTERNET_PERIOD secs.
    # If fails for more than WS_INTERNET_FAILURE_TRHESH consecutive times,
    #
    #
    failures_num = 0
    while True:
        time.sleep(gs.WD_TEST_INTERNET_PERIOD)

        if not CameraNetwork.check_connection():
            failures_num += 1
            logging.debug('Internet watchdog: failure number: %d.' % failures_num)
            if failures_num > capture_settings[gs.INTERNET_FAILURE_THRESH]:
                logging.error('Failed to connect 3G modem. Will reboot...')
                os.system('sudo reboot')
        else:
            logging.debug('Internet watchdog: succeed.')
            failures_num = 0


if __name__ == '__main__':
    main()
