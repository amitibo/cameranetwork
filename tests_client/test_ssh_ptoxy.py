""""""
from __future__ import division
import paramiko
import CameraNetwork
import CameraNetwork.global_settings as gs


def main():
    """"""
    
    proxy_params = CameraNetwork.retrieve_proxy_parameters()

    key = paramiko.RSAKey.from_private_key_file(gs.PROXY_SERVER_KEY_FILE)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=proxy_params['ip'], username="ubuntu", pkey=key)    

    stdin, stdout, stderr = ssh.exec_command('ps -ef | grep python')
    print stdout.read()

if __name__ == '__main__':
    main()