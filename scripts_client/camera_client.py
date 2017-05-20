"""Run a GUI Client.

A GUI client allows easy access to cameras thier settings and their
measurements.
"""
from __future__ import division

import argparse

import CameraNetwork
import CameraNetwork.global_settings as gs
from CameraNetwork.gui.main import startGUI


def main(local_mode, view_local):
    """Main doc"""

    gs.initPaths()

    #
    # Setup logging
    #
    CameraNetwork.initialize_logger(
        log_path='client_logs',
    )

    startGUI(local_mode, view_local)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start the camera client application')
    parser.add_argument('--local', action='store_true', help='Run in local mode.')
    parser.add_argument('--view_local', action='store_true', help='View local cameras.')
    args = parser.parse_args()

    main(args.local, args.view_local)