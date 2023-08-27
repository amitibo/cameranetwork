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
This script starts the camera activity.
Should be run at startup of the raspberry Pi / Odroid XU4.
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
    parser.add_argument('--local_proxy', action='store_true', help='When set, the server will work against a local proxy. This setup is useful for offline running.')
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
        # The controller is initialized first, for some reasons:
        # - Initialize and get camera info.
        # - Pass a pointer to the controller to the server.
        #
        controller = Controller(offline=offline, local_path=args.local_path)
        server = Server(
            controller=controller, identity=args.identity,
            offline=offline, local_path=args.local_path, local_proxy=args.local_proxy)

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
        # TODO Long term fix
        os.system('sync; sudo reboot -f')  # Changed from 'sudo reboot', workaround for reboot hanging
    except Exception as e:
        #
        # Failed starting the camera, might be some USB problem.
        # Note:
        # I delay the reboot so that the tunnel will stay open and
        # enable debugging.
        #
        logging.exception('Rebooting. Unknown error:\n{}'.format(repr(e)))
        logging.shutdown()
        time.sleep(120)
        # TODO Long term fix
        os.system('sync; sudo reboot -f')  # Changed from 'sudo reboot', workaround for reboot hanging


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
        # TODO Long term fix
        os.system('sync; sudo reboot -f')  # Changed from 'sudo reboot', workaround for reboot hanging
    except Exception as e:
        #
        # Failed starting the camera, might be some USB problem.
        # Note:
        # I delay the reboot so that the tunnel will stay open and
        # enable debugging.
        #
        logging.exception('Rebooting. Unknown error:\n{}'.format(repr(e)))
        logging.shutdown()
        time.sleep(120)
        # TODO Long term fix
        os.system('sync; sudo reboot -f')  # Changed from 'sudo reboot', workaround for reboot hanging
