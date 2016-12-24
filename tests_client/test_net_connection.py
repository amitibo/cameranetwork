"""
"""

from __future__ import division
import CameraNetwork
import time

def main():
    """Main doc """
    
    proxy_params = CameraNetwork.retrieve_proxy_parameters()

    client = CameraNetwork.Client(proxy_params)    
    client.start()
    
    
if __name__ == '__main__':
    main()

    
    