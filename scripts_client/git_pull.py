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
"""Run ``git pull`` to update all camera servers.

This code uses the ``fabric`` library to run remote commands on the cameras
To use it run::

   >  fab -f git_pull.py set_hosts git_pull -p odroid

To use a specific remote/branch:

   >  fab -f git_pull.py set_hosts git_pull:remote=origin,branch=dev -p odroid

"""
from __future__ import division
import CameraNetwork
from CameraNetwork.mdp import MDP
from fabric.api import env, run, abort, cd
import logging
logging.basicConfig(level=logging.INFO)
import time

#
# List here hosts that you would not like to update.
#
HOSTS_IGNORE_LIST = ["219"]

def set_hosts():
    #
    # Start the camera client.
    #
    camera_client = CameraNetwork.CLIclient()
    proxy_params = CameraNetwork.retrieve_proxy_parameters()
    camera_client.start(proxy_params)

    #
    # Let the camera clinet get the servers list.
    #
    time.sleep(3)

    #
    # Get the servers credentials.
    #
    credentials_dict = camera_client.send_mmi(MDP.MMI_TUNNELS)

    ports = {
        k: v['tunnel_port'] for  k, v in credentials_dict.items() \
        if not k.endswith("L") and \
        k in camera_client.servers_list and \
        k not in HOSTS_IGNORE_LIST
    }

    global hosts_map
    hosts_map = {
        'odroid@{ip}:{port}'.format(ip=proxy_params['ip'], port=port): k for k, port in ports.items()
    }
    env.hosts = hosts_map.keys()


def git_pull(remote="origin", branch="master"):
    """Run git pull on CameraNetwork code."""

    logging.info("Accessing camera: {}".format(hosts_map[env.host_string]))

    with cd('code/cameranetwork'):
        result = run(
            "git pull {remote} {branch}".format(remote=remote, branch=branch)
        )
        print result
        if result.failed:
            abort("Failed connecting.")