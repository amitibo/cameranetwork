"""Run a GUI Client.

A GUI client allows easy access to cameras thier settings and their
measurements.
"""
from __future__ import division

import argparse

import CameraNetwork
import CameraNetwork.global_settings as gs
from CameraNetwork.gui.main import startGUI


def main(local_mode):
    """Main doc"""

    gs.initPaths()

    #
    # Setup logging
    #
    CameraNetwork.initialize_logger(
        log_path='client_logs',
    )

    startGUI(local_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the camera client application')
    parser.add_argument('--local', action='store_true', help='Run in local mode.')
    args = parser.parse_args()

    main(args.local)