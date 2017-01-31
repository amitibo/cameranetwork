from __future__ import division
from concurrent import futures
import cPickle
from functools import partial
import logging
import numpy as np
from random import randint
from threading import Thread
import time
import traceback
import zmq
from zmq.eventloop import ioloop, zmqstream

import CameraNetwork.global_settings as gs
from CameraNetwork.mdp import *
from CameraNetwork.server import Server
from CameraNetwork.utils import buff2dict

__all__ = ['Client', 'CLIclient']


class Client(MDPClient):
    """
    A base class for  communication with servers.
    """

    def __init__(self, proxy_params, mmi_period=1000):

        self.ctx = zmq.Context()

        self.proxy_params = proxy_params
        self.mmi_period = mmi_period

        #
        # Setup the mdp client
        #
        super(Client, self).__init__(
            context=self.ctx,
            endpoint="tcp://{ip}:{client_port}".format(**proxy_params)
        )

        self._servers_set = set()

    def __del__(self):

        self.loop.stop()
        self.shutdown()
        self.ctx.term()

    def start(self, delay_start=0):
        """Start the client loop."""

        #
        # Create a new instance of the ioloop.
        # This is important for example when running from an ipython notebook (which
        # is itself an ioloop.)
        #
        ioloop.IOLoop.clear_current()
        ioloop.IOLoop.clear_instance() # or del IOLoop._instance in tornado < 3.3

        #
        # Set the (Tornado) loop
        #
        self.loop = ioloop.IOLoop().instance()

        #
        # Start the MDP client.
        #
        super(Client, self).start()

        #
        # Start the ioloop.
        #
        time.sleep(delay_start)
        ioloop.PeriodicCallback(
            partial(self.send_mmi, service=MDP.MMI_SERVICES, msg=[]), self.mmi_period, self.loop
            ).start()
        self.loop.start()

    def send(self, server_address, cmd, msg_extra=MDP.STANDARD, args=(), kwds={}):
        """Send a message to a specific server"""

        msg = (cmd, args, kwds)
        msg_out = cPickle.dumps(msg)

        self.request(
            service=server_address,
            msg_extra=msg_extra,
            msg=msg_out
        )

    def handle_new_server(self, server):
        """Callback on connection of a new server. Derived classes should override this method."""
        pass

    def handle_server_failure(self, server):
        """Callback on disconnection of a server. Derived classes should override this method."""
        pass

    def handle_receive(self, msg_extra, service, status, cmd, args, kwds):
        """Callback to handle receive. This is called only if there are no other callbacks to handle the message. Derived classes should override this method."""

        raise Warning('Unattended message: ', str((status, cmd, args, kwds)))

    def on_message(self, msg):
        """Public method called when a message arrived."""

        # 1st part is msg type
        msg_extra = ord(msg.pop(0))

        # 2nd part is empty
        msg.pop(0)

        # 3nd part is protocol version
        # TODO: version check
        proto = msg.pop(0)

        # 4rd part is service type
        service = msg.pop(0)
        if service.startswith(b'mmi.'):
            self.on_mmi(service, msg)
            return

        status, cmd, args, kwds = cPickle.loads(msg[0])

        #
        # Call the coresponding cmd callback.
        #
        self.handle_receive(msg_extra, service, status, cmd, args, kwds)

    def on_timeout(self):
        """Public method called when a timeout occured.

        .. note:: Does nothing. Should be overloaded!
        """
        pass

    def send_mmi(self, service, msg=[]):
        """Check the list of available servers"""

        self.request(service=service, msg=msg)

    def on_mmi(self, service, msg):
        """handle mmi requests"""

        if service == MDP.MMI_SERVICES:
            self.calculate_server_changes(msg)
        elif service == MDP.MMI_TUNNELS:
            self.tunnels_cb(cPickle.loads(msg[0]))
        else:
            raise Warning('Unknown mmi msg: %s, %s' % (service, str(msg)))

        return

    def tunnels_cb(self, tunnels_dict):
        raise NotImplementedError("'tunnels_cb' should be implemented by a subclass.")

    def calculate_server_changes(self, updated_servers_list):
        """
        Send a ping to all servers connected to the client. This is used for checking
        which servers are alive.
        """

        #
        # Check the previous responses. This is done here as we expect that we
        # received all responses.
        #
        updated_servers_list = set(updated_servers_list)
        good_servers = self._servers_set.intersection(updated_servers_list)
        server_failures = self._servers_set.difference(good_servers)
        new_servers = updated_servers_list.difference(good_servers)

        map(self._handle_new_server, new_servers)
        map(self._handle_server_failure, server_failures)

    def _handle_new_server(self, server):
        """Handling the connection of a new server"""

        logging.debug("yay, got new server {}!".format(server))

        #
        # Update server list
        #
        self._servers_set.add(server)

        #
        # Call callback
        #
        self.handle_new_server(server)

    def handle_new_server(self, server):
        """Callback on connection of a new server. Derived classes should override this method."""
        pass

    def _handle_server_failure(self, server):
        """Handling the disconnection of a server"""

        logging.debug("Server {} failed :(".format(server))

        #
        # Update server list
        #
        self.handle_server_failure(server)

        #
        # Call callback
        #
        self._servers_set.remove(server)

    def handle_server_failure(self, server):
        """Callback on disconnection of a server. Derived classes should override this method."""
        pass

    @property
    def servers(self):

        return list(self._servers_set)


class ServerProxy(object):
    """Helper class to 'automatically implement cmd api for the CLI client.
    """

    def __init__(self, client, server_id):

        self._client = client
        self._server_id = server_id

    def __getattr__(self, name):
        """Dynamically create messages."""

        if not hasattr(Server, 'handle_{}'.format(name)):
            raise AttributeError("Unkown server command: {}".format(name))

        #
        # Create sendmessage method.
        #
        def autocmd(*args, **kwds):

            #
            # Send message
            #
            status, _, args_answer, kwds_answer = \
                self._client.send_message(
                    server_id=self._server_id,
                    cmd=name,
                    args=args,
                    kwds=kwds
                )

            #
            # Check the reply status
            #
            if status != gs.MSG_STATUS_OK:
                raise gs.MSG_EXCEPTION_MAP[status](args_answer[0])

            return kwds_answer

        autocmd.__doc__ = getattr(Server, 'handle_{}'.format(name)).__doc__

        return autocmd


class CLIclient(object):
    """'Command Line' client.

    Useful for interfacing with cameras from Ipython or from scripts.
    """

    def __init__(self):

        self.futures = {}
        self.servers_list = []

    def __getitem__(self, i):

        if i not in self.client_instance.servers:
            raise IndexError(
                'Unkown server: {}. List of known servers: {}.'.format(
                    i, self.client_instance.servers
                )
            )

        return ServerProxy(self, i)

    def start(self, proxy_params):

        client_instance = Client(proxy_params)

        #
        # Bind callbacks
        #
        client_instance.handle_new_server = self.add_server
        client_instance.handle_server_failure = self.remove_server
        client_instance.handle_receive = self.receive_message
        client_instance.tunnels_cb = self.tunnels_cb

        self.client_instance = client_instance

        #
        # Start the camera thread
        #
        thread = Thread(target=self.client_instance.start, args=(0,))
        thread.daemon = True
        thread.start()

    def send_message(self, server_id, cmd, args=(), kwds={}, timeout=30):

        future = futures.Future()
        self.futures[server_id] = future

        loop = ioloop.IOLoop.instance()
        loop.add_callback(self.client_instance.send, server_address=server_id, cmd=cmd, args=args, kwds=kwds)

        return future.result(timeout=timeout)

    def send_mmi(self, service, msg=[], timeout=30):

        future = futures.Future()
        self.futures['mmi'] = future

        loop = ioloop.IOLoop.instance()
        loop.add_callback(
            self.client_instance.send_mmi,
            service=service,
            msg=msg
        )

        return future.result(timeout=timeout)

    def tunnels_cb(self, tunnels):
        """Return the tunnels data."""

        self.futures['mmi'].set_result(tunnels)

    def add_server(self, server_id):
        logging.info('Adding the new server: {}'.format(server_id))

        self.servers_list.append(server_id)
        self.servers_list = sorted(self.servers_list)

    def remove_server(self, server_id):
        logging.info('Removing the server: {}'.format(server_id))

        self.servers_list.remove(server_id)

    def receive_message(self, msg_extra, server_id, status, cmd, args, kwds):

        if server_id in self.futures.keys():
            self.futures[server_id].set_result((status, cmd, args, kwds))

    def get_array(
        self,
        server_id,
        exposure_us=500,
        gain_db=0,
        resolution=301,
        frames_num=1,
        color_mode=gs.COLOR_RAW,
        gain_boost=False,
        normalize=True
        ):

        status, cmd, args, kwds = self.send_message(
            server_id,
            cmd=gs.MSG_TYPE_ARRAY,
            kwds=dict(
                exposure_us=exposure_us,
                gain_db=gain_db,
                resolution=resolution,
                frames_num=frames_num,
                color_mode=color_mode,
                gain_boost=gain_boost,
                normalize=normalize
            )
        )
        if status == gs.MSG_STATUS_ERROR:
            raise Exception(args[0])

        img_array = np.ascontiguousarray(buff2dict(kwds['matfile'])['img_array'])
        img_data = kwds['img_data']

        return img_array, img_data

    def sunshader(
        self,
        server_id,
        angle,
        ):

        assert angle >= 20 and angle <= 160, \
               'angle must be between 20-160, got {}'.format(angle)
        status, cmd, args, kwds = self.send_message(
            server_id,
            cmd=gs.MSG_TYPE_SUNSHADER,
            kwds=dict(
                angle=angle
            )
        )
        if status == gs.MSG_STATUS_ERROR:
            raise Exception(args[0])

    def query(
        self,
        server_id,
        query_day
        ):

        status, cmd, args, kwds = self.send_message(
            server_id,
            cmd=gs.MSG_TYPE_QUERY,
            kwds=dict(
                query_date=query_day
            )
        )
        if status == gs.MSG_STATUS_ERROR:
            raise Exception(args[0])

        images_df = kwds['images_df']

        return images_df

    def seek(
        self,
        server_id,
        seek_time,
        hdr_index,
        resolution
        ):

        status, cmd, args, kwds = self.send_message(
            server_id,
            cmd=gs.MSG_TYPE_SEEK,
            kwds=dict(
                seek_time=seek_time,
                hdr_index=hdr_index,
                normalize=True,
                resolution=resolution
            )
        )
        if status == gs.MSG_STATUS_ERROR:
            raise Exception(args[0])

        img_array = np.ascontiguousarray(buff2dict(kwds['matfile'])['img_array'])
        img_data = kwds['img_data']

        return img_array, img_data


def main ():

    import CameraNetwork
    from CameraNetwork.sunphotometer import findClosestImageTime
    c = CameraNetwork.CLIclient()
    proxy_params = CameraNetwork.retrieve_proxy_parameters(local_mode=True)
    c.start(proxy_params)

    qdf_102 = c.query('102', '2016-10-23')
    closest_time = findClosestImageTime(qdf_102, '2016-10-23 05:13:07', hdr='2')
    img, img_data = c.seek('102', closest_time, -1, 301)


if __name__ == '__main__':
    main()
