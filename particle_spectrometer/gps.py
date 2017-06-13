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
from __future__ import division
import serial.tools.list_ports as serports
from pymavlink import mavutil
from datetime import datetime
from sortedcontainers import SortedDict
from urlparse import urlparse
from twisted.internet import reactor, threads
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.serialport import SerialPort
from twisted.protocols.basic import LineReceiver
import numpy as np
import json
import sys
import os


BASE_TIMESTAMP = '%Y_%m_%d_%H_%M_%S_%f'


def wait_heartbeat(m):
    """wait for a heartbeat so we know the target system IDs"""

    print 'Waiting for heartbeat'
    m.wait_heartbeat()
    print 'Recieved heartbeat'


def _send_data_request(master, rate):
    #
    # wait for the heartbeat msg to find the system ID
    #
    wait_heartbeat(master)
    #
    # Setting requested streams and their rate.
    #
    for i in range(3):
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            rate,
            1
        )


continue_messages = True

def monitorMessages(m, rate):
    """show incoming mavlink messages"""

    _send_data_request(m, rate)

    while continue_messages:
        msg = m.recv_match(blocking=True)

        if not msg or msg.get_type() == "BAD_DATA":
            continue

        if msg.get_type() == "GLOBAL_POSITION_INT":
            t = datetime.now()
            flight_data = {}            
            flight_data['timestamp'] = t.strftime(BASE_TIMESTAMP)
            flight_data['lon'] = msg.lon
            flight_data['lat'] = msg.lat
            flight_data['alt'] = msg.alt
            flight_data['relative_alt'] = msg.relative_alt
            flight_data['hdg'] = msg.hdg

            reactor.callFromThread(addPHdata, flight_data)


def addPHdata(flight_data):
    """Add new flight data message to the records"""

    global flight_data_log

    flight_data_log[flight_data['timestamp']] = flight_data


def queryPHdata(timestamp):
    """Query the closest flight data records to some timestamp"""

    if len(flight_data_log.values()) == 0:
        return None

    index = flight_data_log.bisect(timestamp)

    r_index = max(index-1, 0)

    return flight_data_log.values()[r_index]


def initPixHawk(device='/dev/ttyACM0', baudrate=115200, rate=4):
    """Start the thread that monitors the PixHawk Mavlink messages.

    Parameters
    ----------
    device: str
        Address of serialport device.
    baudrate: int
        Serialport baudrate (defaults to 57600)
    rate: int
        Requested rate of messages.
    """

    global flight_data_log
    flight_data_log = SortedDict()

    #
    # create a mavlink serial instance
    #
    master = mavutil.mavlink_connection(device, baud=baudrate)

    #
    # Start the messages thread
    #
    d = threads.deferToThread(monitorMessages, master, rate)


def stopPixHawk():
    global continue_messages

    continue_message = False


serServ = None

class GRIMMclient(LineReceiver):
    def connectionMade(self):
        global serServ
        serServ = self
        print 'GRIMM device: ', serServ, ' is connected.'

    def makeConnection(self, transport):
        print transport

    def cmdReceived(self, cmd):
        serServ.transport.write(cmd)
        print cmd, ' - sent to GRIMM.'

    def lineReceived(self, line):
        self.factory.receivedData(line)


class GRIMMcontrolFactory(Factory):
    protocol = GRIMMclient

    def __init__(self, dst_folder):
        self.state = 0
        self.data = SortedDict()
        self.current_time = None
        self.current_data = []
        self.dst_folder = dst_folder

    def receivedData(self, line):

        line = line.lower()

        if self.state == 0:
            if line.startswith('p'):
                self.state = 1
                if self.current_time is not None:
                    data = {'data': self.current_data, 'coords': self.current_coords}
                    with open(os.path.join(self.dst_folder, '{}.json').format(self.current_time.strftime(BASE_TIMESTAMP)), 'w') as f:
                        json.dump(data, f)
                    if data['coords'] is None:
                        print data
                    else:
                        print '{t}\t{lat}\t{lon}\t{alt}\t{relative_alt}\t{data}'.format(
                            t=data['coords']['timestamp'],
                            lat=data['coords']['lat'],
                            lon=data['coords']['lon'],
                            alt=data['coords']['alt'],
                            relative_alt=data['coords']['relative_alt'],
                            data=data['data']
                        )
                    self.data[self.current_time] = data
                self.current_data = []
                self.current_time = datetime.now()
                self.current_coords = queryPHdata(self.current_time.strftime(BASE_TIMESTAMP))

            return 

        if not line.startswith('c'):
            self.state = 0
            return

        self.state = (self.state + 1) % 5

        self.current_data = self.current_data + [int(i) for i in line.strip().split()[1:]]


def initGRIMM(dst_folder, device='/dev/ttyUSB0'):

    SerialPort(
        GRIMMcontrolFactory(dst_folder=dst_folder).buildProtocol(None),
        device,
        reactor,
        baudrate='9600',
        xonxoff=True
    )


def main():
    #
    # Destination Folder
    #
    dst_folder = './data/{}'.format(datetime.now().strftime(BASE_TIMESTAMP))
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    #
    # Start the GPS monitoring.
    #
    initPixHawk()

    #
    # STart the GRIMM particle monitoring.
    #
    initGRIMM(dst_folder=dst_folder)

    reactor.run()


if __name__ == '__main__':
    main()
