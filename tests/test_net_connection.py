#!/usr/bin/env python
"""
"""

from __future__ import division
import CameraNetwork
import time
import os



def main():
    """Main doc """
    
    tunnel_params = CameraNetwork.retrieve_proxy_parameters()
    
    CameraNetwork.TunnelThread(
        **tunnel_params
    ).start()

    time.sleep(10)

    
if __name__ == '__main__':
    main()

    
    
