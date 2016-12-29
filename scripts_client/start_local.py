"""Start Cameras, Proxy and Client locally.

This script will run cameras, a proxy and a client locally on a
single computer. It is helpfull for replaying saved images.
Given a folder where the camera's data was saved, it will
start the right number of cameras and point their home
folder to the corresponding stored data.
"""

from __future__ import division, absolute_import, print_function
import argparse
import CameraNetwork
from glob import glob
import os
import subprocess as sbp


def main(base_path, debug_mode=False):
    camera_paths = sorted(glob(os.path.join(base_path, '*')))
    camera_paths = filter(lambda p: os.path.isdir(p), camera_paths)


    #
    # Start the proxy.
    #
    proxy = sbp.Popen(['python'] + ['../scripts_proxy/start_proxy.py'])
    
    #
    # Start the client.
    #
    if not debug_mode:
        client = sbp.Popen(['python'] +
                           ['../scripts_client/camera_client.py', '--local'])
    
    #
    # Start all cameras.
    #
    servers = []
    for path in camera_paths:
        servers.append(sbp.Popen(['python'] +
                                 ['../scripts/start_server.py', '--local', path]))

    try:
        client.wait()
    finally:
        proxy.kill()
        for server in servers:
            server.kill()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a complete local session.')
    parser.add_argument(
        '--debug_mode', '-d', action='store_true',
        help="Do not start the client. The client will be started from a debugger.")
    parser.add_argument('base_path', help='Base path of cameras data.')
    args = parser.parse_args()

    main(args.base_path)
