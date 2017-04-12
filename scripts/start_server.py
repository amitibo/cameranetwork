#!/usr/bin/env python
"""
This script statrts the camera activity.
Should be run at startup of the raspberry Pi.
"""

import argparse
import CameraNetwork
from CameraNetwork.controller import Controller
import CameraNetwork.global_settings as gs
from CameraNetwork.server import Server
from CameraNetwork.utils import IOLoop
from CameraNetwork.utils import RestartException
from CameraNetwork.utils import CameraException
import logging
import os
import sys
import time
from tornado import gen


def restart_program():
    """Restarts the current program.

    https://www.daniweb.com/programming/software-development/code/260268/restart-your-python-program
    """

    logging.info("Performing restart")
    logging.shutdown()
    python = sys.executable
    os.execl(python, python, *sys.argv)


@gen.coroutine
def main():
    parser = argparse.ArgumentParser(description='Start the server application')
    parser.add_argument('--log_level', default='DEBUG', help='Set the log level (possible values: info, debug, ...)')
    parser.add_argument('--unloop', action='store_false', help='When set, the camera will not start in loop mode')
    parser.add_argument('--identity', type=str, default=None, help='ID of the server (defaults to the value in the camera settings).')
    parser.add_argument('--offline', action='store_true', help='When set, the server will work without camera nor shader. This setup is useful for offline running.')
    parser.add_argument('--local_path', type=str, default=None, help='If set, the camera will work in local and offline mode, using the given path as home.')
    args = parser.parse_args()

    gs.initPaths(args.local_path)


    #
    # Initialize the logger
    #
    if args.local_path is None:
        CameraNetwork.initialize_logger(
            log_path=gs.DEFAULT_LOG_FOLDER,
            log_level=args.log_level,
            postfix='_camera')

    #
    # In local mode, the camera is working offline.
    #
    if args.local_path is not None or args.offline:
        offline = True
    else:
        offline = False

    try:
        #
        # Setup.
        # Note:
        # The controller is intialized first, for some reasons:
        # - Initialize and get camera infor.
        # - Pass a pointer to the controller to the server.
        #
        controller = Controller(offline=offline, local_path=args.local_path)
        server = Server(
            controller=controller, identity=args.identity,
            offline=offline, local_path=args.local_path)

        #
        # Start the server and controller
        #
        controller.start()
        server.start()

    except RestartException as e:
        #
        # The user requested a restart of the software.
        #
        logging.exception("User requested to restart program.")
        restart_program()

    except KeyboardInterrupt as e:
        #
        # User stopped the program.
        #
        logging.exception('Program stopped by user.')
    except CameraException as e:
        #
        # Failed starting the camera, might be some USB problem.
        # Note:
        # I delay the reboot so that the tunnel will stay open and
        # enable debugging.
        #
        logging.exception('Failed starting the camera. Rebooting.')
        logging.shutdown()
        time.sleep(120)
        os.system('sudo reboot')
    except Exception as e:
        #
        # Failed starting the camera, might be some USB problem.
        # Note:
        # I delay the reboot so that the tunnel will stay open and
        # enable debugging.
        #
        logging.exception('Unkown error:\n{}'.format(repr(e)))
        logging.shutdown()
        time.sleep(120)
        os.system('sudo reboot')


if __name__ == '__main__':
    main()
    try:
        IOLoop.instance().start()
    except RestartException as e:
        #
        # The user requested a restart of the software.
        #
        logging.exception("User requested to restart program.")
        restart_program()

    except KeyboardInterrupt as e:
        #
        # User stopped the program.
        #
        logging.exception('Program stopped by user.')
    except CameraException as e:
        #
        # Failed starting the camera, might be some USB problem.
        # Note:
        # I delay the reboot so that the tunnel will stay open and
        # enable debugging.
        #
        logging.exception('Failed starting the camera. Rebooting.')
        logging.shutdown()
        time.sleep(120)
        os.system('sudo reboot')
    except Exception as e:
        #
        # Failed starting the camera, might be some USB problem.
        # Note:
        # I delay the reboot so that the tunnel will stay open and
        # enable debugging.
        #
        logging.exception('Unkown error:\n{}'.format(repr(e)))
        logging.shutdown()
        time.sleep(120)
        os.system('sudo reboot')
