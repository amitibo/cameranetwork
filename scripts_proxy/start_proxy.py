"""
Setup the proxy server. This script should be run on the proxy server that connects the
clients to the servers.
"""
from CameraNetwork import mdp
import CameraNetwork
import CameraNetwork.global_settings as gs
import argparse
import logging
from zmq.eventloop.ioloop import IOLoop
import zmq


def main ():
    parser = argparse.ArgumentParser(
        description='Start the proxy application'
    )
    parser.add_argument(
        '--log_level',
        default='INFO',
        help='Set the log level (possible values: info, debug, ...)'
    )
    parser.add_argument(
        '--log_path',
        default='proxy_logs',
        help='Set the log folder'
    )
    args = parser.parse_args()

    gs.initPaths()

    #
    # Initialize the logger
    #
    CameraNetwork.initialize_logger(
        log_path=args.log_path,
        log_level=args.log_level,
        postfix='_proxy'
    )
    proxy_params = CameraNetwork.retrieve_proxy_parameters()

    #
    # Start the broker pattern.
    #
    context = zmq.Context()
    broker = mdp.MDPBroker(
        context,
        main_ep="tcp://*:{proxy_port}".format(**proxy_params),
        client_ep="tcp://*:{client_port}".format(**proxy_params),
        hb_ep="tcp://*:{hb_port}".format(**proxy_params),
    )
    IOLoop.instance().start()
    broker.shutdown()


if __name__ == '__main__':
    main()
