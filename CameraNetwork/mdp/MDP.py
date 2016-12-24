"""Majordomo Protocol definitions"""
#  This is the version of MDP/Client we implement
C_CLIENT = b'MDPC01'

#  This is the version of MDP/Worker we implement
W_WORKER = b'MDPW01'

#  MDP/Server commands, as strings
EMPTY_FRAME     = b''
W_READY         = b'\x01'
W_REQUEST       = b'\x02'
W_REPLY         = b'\x03'
W_HEARTBEAT     = b'\x04'
W_DISCONNECT    = b'\x05'

commands = [None, "READY", "REQUEST", "REPLY", "HEARTBEAT", "DISCONNECT"]

UNKNOWN_SERVICE = b'404'
KNOWN_SERVICE   = b'200'
UNKNOWN_COMMAND = b'501'

MMI_SERVICE = b'mmi.service'
MMI_SERVICES = b'mmi.services'
MMI_TUNNELS = b'mmi.tunnels'

#
# Hack to add broadcast capabilities to clients.
#
STANDARD = 0
BROADCAST = 1