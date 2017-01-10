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
    
    logging.debug('Trying to load proxy parameters from url: %s' % CONFIGURATION_SERVER_URL_BASE)

    #
    # Note:
    # For some reason the odroid is caching an old result.
    # I am trying to set no-cache, but it doesn't help.
    #
    header = {"pragma-directive" : "no-cache"}
    req = urllib2.Request(CONFIGURATION_SERVER_URL_BASE, headers=header)
    response = urllib2.urlopen(req)
    soup = BeautifulSoup(response)
    
    #
    # Selected the newest proxy params file
    #
    proxy_file = sorted([f for f in soup.find_all(name='a', text=re.compile('.json'))])[-1]
    proxy_file_path = posixpath.join(CONFIGURATION_SERVER_URL_BASE, proxy_file.text)
    params = json.load(urllib2.urlopen(proxy_file_path))
    
    logging.debug('Retrieved the following proxy parameters: %s' % str(params))

    if local_mode:
        params['ip'] = '127.0.0.1'

    return params



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

