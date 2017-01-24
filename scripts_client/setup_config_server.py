"""
"""

from __future__ import division
import CameraNetwork

AMAZON_SERVER = {
    #'ip': '52.59.36.61',
    #"ip": "10.40.0.220",
    #"ip": "vislbatch2.technion.ac.il",
    #'ip': 'localhost',
    #'ip': '132.68.58.206',
    'ip': '35.156.53.208',
    'proxy_port': 1980,
    'client_port': 1985,
    'hb_port': 1990,
}


def main():
    """Main doc """

    CameraNetwork.setup_config_server(**AMAZON_SERVER)


if __name__ == '__main__':
    main()



