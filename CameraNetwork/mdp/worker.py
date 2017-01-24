# -*- coding: utf-8 -*-

"""Module containing worker functionality for the MDP implementation.

For the MDP specification see: http://rfc.zeromq.org/spec:7
"""

__license__ = """
    This file is part of MDP.

    MDP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    MDP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with MDP.  If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = 'Guido Goldstein'
__email__ = 'gst-py@a-nugget.de'


import sys
import time
from pprint import pprint

import zmq
from zmq.eventloop.zmqstream import ZMQStream
from zmq.eventloop.ioloop import IOLoop, DelayedCallback, PeriodicCallback

from util import split_address
from MDP import *

import logging


class MDPWorker(object):

    """Class for the MDP worker side.

    Thin encapsulation of a zmq.DEALER socket.
    Provides a send method with optional timeout parameter.

    Will use a timeout to indicate a broker failure.
    """

    _proto_version = W_WORKER

    # TODO: integrate that into API
    HB_INTERVAL = 2500 # in milliseconds
    HB_LIVENESS = 5    # HBs to miss before connection counts as dead
    RECONNECT_PERIOD = 5000

    def __init__(self, context, endpoint, service, endpoint_callback=None):
        """Initialize the MDPWorker.

        context is the zmq context to create the socket from.
        service is a byte-string with the service name.
        """
        self.context = context
        self.endpoint = endpoint
        self.service = service
        self.endpoint_callback = endpoint_callback
        self.stream = None
        self.ticker = None
        self._delayed_reconnect = None
        self._create_stream()

    def _create_stream(self):
        """Helper to create the socket and the stream.
        """

        logging.debug('Worker creating stream')
        socket = self.context.socket(zmq.DEALER)
        ioloop = IOLoop.instance()
        self.stream = ZMQStream(socket, ioloop)
        self.stream.on_recv(self._on_message)
        self.stream.socket.setsockopt(zmq.LINGER, 0)
        self.stream.connect(self.endpoint)
        self.ticker = PeriodicCallback(self._tick, self.HB_INTERVAL)
        self._send_ready()
        self.ticker.start()

    def _send_ready(self):
        """Helper method to prepare and send the workers READY message.
        """

        ready_msg = [EMPTY_FRAME, self._proto_version, W_READY, self.service]
        self.stream.send_multipart(ready_msg)
        self.curr_liveness = self.HB_LIVENESS

    def _tick(self):
        """Method called every HB_INTERVAL milliseconds.
        """

        self.curr_liveness -= 1
        logging.debug('Worker HB tick, current liveness: %d' % self.curr_liveness)

        self.send_hb()
        if self.curr_liveness >= 0:
            return

        #
        # Ouch, connection seems to be dead
        #
        logging.debug('Worker lost connection')
        self.shutdown()

        #
        # try to recreate the connection
        #
        self._delayed_reconnect = DelayedCallback(self._recreate_stream, self.RECONNECT_PERIOD)
        self._delayed_reconnect.start()

    def _recreate_stream(self):

        logging.debug('Worker trying to recreate stream')

        if self.endpoint_callback is not None:
            #
            # Check, maybe the ip of the proxy changed.
            #
            try:
                self.endpoint = self.endpoint_callback()
            except:
                #
                # Probably some problem in accessing the server.
                #
                self._delayed_reconnect = DelayedCallback(self._recreate_stream, self.RECONNECT_PERIOD)
                self._delayed_reconnect.start()
                return

        self._create_stream()

    def send_hb(self):
        """Construct and send HB message to broker.
        """

        msg = [EMPTY_FRAME, self._proto_version, W_HEARTBEAT]
        self.stream.send_multipart(msg)

    def shutdown(self):
        """Method to deactivate the worker connection completely.

        Will delete the stream and the underlying socket.
        """

        logging.debug('Shutdown of the worker')

        if self.ticker:
            logging.debug('Stopping the HB ticker')
            self.ticker.stop()
            self.ticker = None

        if not self.stream:
            return

        logging.debug('Closing the stream')

        self.stream.socket.close()
        self.stream.close()
        self.stream = None
        self.timed_out = False
        self.connected = False

    def reply(self, msg):
        """Send the given message.

        msg can either be a byte-string or a list of byte-strings.
        """

        #
        # prepare full message
        #
        to_send = self.envelope
        self.envelope = None

        if isinstance(msg, list):
            to_send.extend(msg)
        else:
            to_send.append(msg)

        self.stream.send_multipart(to_send)

    def _on_message(self, msg):
        """Helper method called on message receive.

        msg is a list w/ the message parts
        """

        logging.debug('Received message: {}'.format(msg))

        #
        # 1st part is empty
        #
        msg.pop(0)

        #
        # 2nd part is protocol version
        # TODO: version check
        #
        proto = msg.pop(0)

        #
        # 3rd part is message type
        #
        msg_type = msg.pop(0)

        #
        # XXX: hardcoded message types!
        # any message resets the liveness counter
        #
        self.curr_liveness = self.HB_LIVENESS

        if msg_type == W_DISCONNECT:
            #
            # Disconnect. Reconnection will be triggered by hb timer
            #
            self.curr_liveness = 0
        elif msg_type == W_REQUEST:
            #
            # Request. Remaining parts are the user message
            #
            envelope, msg = split_address(msg)
            envelope.append(EMPTY_FRAME)
            envelope = [EMPTY_FRAME, self._proto_version, W_REPLY] + envelope
            self.envelope = envelope

            self.on_request(msg)
        else:
            #
            # invalid message
            # ignored
            #
            pass

    def on_request(self, msg):
        """Public method called when a request arrived.

        Must be overloaded!
        """

        raise NotImplementedError('on_request must be implemented by the subclass.')