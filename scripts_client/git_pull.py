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

    ports = {k: v['tunnel_port'] for \
             k, v in credentials_dict.items() if not k.endswith("L") and k in camera_client.servers_list}

    global hosts_map
    hosts_map = {'odroid@{ip}:{port}'.format(ip=proxy_params['ip'], port=port): k for k, port in ports.items()}
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