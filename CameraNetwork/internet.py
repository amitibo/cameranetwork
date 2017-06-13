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
from .global_settings import *
import urllib2
from bs4 import BeautifulSoup
import platform
import posixpath
import re
import json
import subprocess
import datetime
import logging
import getpass
import time
import traceback
import os


__all__ = [
    'check_connection',
    'retrieve_proxy_parameters',
    'setup_config_server',
    'setup_forward_ssh_tunnel'
]


def check_connection():
    """Check if there is a connection to the internet."""

    try:
        response=urllib2.urlopen('http://www.google.com', timeout=8)
        return True
    except:
        pass

    return False


###############################################################################################
# Functions called by the RaspberryPi
#
def retrieve_proxy_parameters(local_mode=False):
    """Retrieve proxy parameters for the camera.

    Proxy settings are stored as a json file.
    """

    proxy_params = json.loads(DEFAULT_PROXY_PARAMS)

    logging.debug("Proxy parameters:\n{}".format(proxy_params))

    if local_mode:
        proxy_params['ip'] = '127.0.0.1'

    return proxy_params



###############################################################################################
# Functions called by the client applications
#
def setup_config_server(
    **kwds
    ):
    """Setup up the configuration server

    This server tells the cameras how to
    connect to the proxy server. Proxy settings are stored as a json file.
    """

    import paramiko

    ssh = paramiko.SSHClient()
    sftp = None

    try:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        pw = getpass.getpass()
        ssh.connect(hostname=CONFIGURATION_SERVER, username=CONFIGURATION_SERVER_USER, password=pw)
        sftp = ssh.open_sftp()

        #
        # Delete previous settings
        #
        stdin, stdout, stderr = ssh.exec_command(
            'rm -f -r %s; mkdir %s' % (CONFIGURATION_SERVER_BASE_PATH, CONFIGURATION_SERVER_BASE_PATH)
        )
        print stdout.readlines()
        sftp.chdir(CONFIGURATION_SERVER_BASE_PATH)

        #
        # Create settings for proxy server
        #
        temp_settings_dict = dict(PROXY_SETTINGS_DICT)
        temp_settings_dict.update(kwds)

        #
        # Store it as a json file
        #
        json_file_name = PROXY_SETTINGS_FILE_NAME.format(timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
        with sftp.open(json_file_name, 'wb') as f:
            f.write(json.dumps(temp_settings_dict, sort_keys=True, indent=4, separators=(',', ': ')))

    finally:
        ssh.close()
        if sftp is not None:
            sftp.close()


#----------------------------------------------------------------------
def setup_forward_ssh_tunnel(
    server_ip,
    server_user,
    local_port=22,
    tunnel_port=22220,
    key=PROXY_SERVER_KEY_FILE,
    **kwds
    ):
    """Create the (forward) ssh tunnel from client to proxy server.

    Args:
        server_ip (str): SERVER_IP of the proxy server.
        server_user (str): User name to log on the proxy server.
        local_port (int): Local port on which to connect to the proxy server.
        tunnel_port (int): Tunnel port (on the remote server).
    """

    import zmq.ssh.tunnel

    tunnel_proc = zmq.ssh.tunnel.paramiko_tunnel(
        lport=local_port,
        rport=tunnel_port,
        server='%s@%s' % (server_user, server_ip),
        keyfile=key
    )

    return tunnel_proc

