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
# -*- coding: utf-8 -*-

"""Module containing client functionality for the MDP implementation.

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


import zmq
from zmq.eventloop.zmqstream import ZMQStream
from zmq.eventloop.ioloop import IOLoop, DelayedCallback

from MDP import *


class MDPClient(object):

    """Class for the MDP client side.

    Thin asynchronous encapsulation of a zmq.REQ socket.
    Provides a :func:`request` method with optional timeout.

    Objects of this class are ment to be integrated into the
    asynchronous IOLoop of pyzmq.

    :param context:  the ZeroMQ context to create the socket in.
    :type context:   zmq.Context
    :param endpoint: the enpoint to connect to.
    :type endpoint:  str
    :param service:  the service the client should use
    :type service:   str
    """

    _proto_version = C_CLIENT

    def __init__(self, context, endpoint):
        """Initialize the MDPClient.
        """
        
        self.context = context
        self.endpoint = endpoint
        
    def start(self):
        """
        Initialize the zmq sockets on a ioloop stream.
        The separation of this part from the init is useful if
        we start the client on a separate thread with a new ioloop
        (for example to enable use in an ipython notebook)
        """
        socket = self.context.socket(zmq.DEALER)
        ioloop = IOLoop.instance()
        self.stream = ZMQStream(socket, ioloop)
        self.stream.on_recv(self._on_message)
        self._proto_prefix = [EMPTY_FRAME, self._proto_version]
        self._delayed_timeout = None
        self.timed_out = False
        socket.connect(self.endpoint)

    def shutdown(self):
        """Method to deactivate the client connection completely.

        Will delete the stream and the underlying socket.

        .. warning:: The instance MUST not be used after :func:`shutdown` has been called.

        :rtype: None
        """
        
        if not self.stream:
            return
        
        self.stream.socket.setsockopt(zmq.LINGER, 0)
        self.stream.socket.close()
        self.stream.close()
        self.stream = None

    def request(self, service, msg, msg_extra=STANDARD, timeout=None):
        """Send the given message.

        :param msg:     message parts to send.
        :type msg:      list of str
        :param msg_extra: Extra message flags (e.g. STANDARD or BROADCAST)
        :type msg_extra: int
        :param timeout: time to wait in milliseconds.
        :type timeout:  int
        
        :rtype None:
        """
        if type(msg) in (bytes, unicode):
            msg = [msg]
            
        #
        # prepare full message
        #
        to_send = [chr(msg_extra)] + self._proto_prefix[:]
        to_send.extend([service])
        to_send.extend(msg)
        
        self.stream.send_multipart(to_send)
        
        if timeout:
            self._start_timeout(timeout)

    def _on_timeout(self):
        """Helper called after timeout.
        """
        
        self.timed_out = True
        self._delayed_timeout = None
        self.on_timeout()

    def _start_timeout(self, timeout):
        """Helper for starting the timeout.

        :param timeout:  the time to wait in milliseconds.
        :type timeout:   int
        """
        
        self._delayed_timeout = DelayedCallback(self._on_timeout, timeout)
        self._delayed_timeout.start()

    def _on_message(self, msg):
        """Helper method called on message receive.

        :param msg:   list of message parts.
        :type msg:    list of str
        """
        
        if self._delayed_timeout:
            # 
            # disable timout
            #
            self._delayed_timeout.stop()
            self._delayed_timeout = None
            
        self.on_message(msg)

    def on_message(self, msg):
        """Public method called when a message arrived.

        .. note:: Does nothing. Should be overloaded!
        """

        raise NotImplementedError('on_message must be implemented by the subclass.')
        
    def on_timeout(self):
        """Public method called when a timeout occured.

        .. note:: Does nothing. Should be overloaded!
        """
        raise NotImplementedError('on_timeout must be implemented by the subclass.')
        

from zmq import select

def mdp_request(socket, service, msg, timeout=None):
    """Synchronous MDP request.

    This function sends a request to the given service and
    waits for a reply.

    If timeout is set and no reply received in the given time
    the function will return `None`.

    :param socket:    zmq REQ socket to use.
    :type socket:     zmq.Socket
    :param service:   service id to send the msg to.
    :type service:    str
    :param msg:       list of message parts to send.
    :type msg:        list of str
    :param timeout:   time to wait for answer in seconds.
    :type timeout:    float

    :rtype list of str:
    """
    
    if not timeout or timeout < 0.0:
        timeout = None
    
    if type(msg) in (bytes, unicode):
        msg = [msg]
    
    to_send = [C_CLIENT, service]
    to_send.extend(msg)
    socket.send_multipart(to_send)
    ret = None
    
    rlist, _, _ = select([socket], [], [], timeout)
    
    if rlist and rlist[0] == socket:
        ret = socket.recv_multipart()
        ret.pop(0) # remove service from reply
        
    return ret
