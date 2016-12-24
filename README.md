CameraNetwork
=============

Code for running and analyzing the Camera Network

Latest version can be downloaded from [bitbucket](http://bitbucket.org/amitibo/CameraNetwork).

Prerequisites
-------------

It might be useful, on *raspberryPi*, to install *miniconda* to help installing some python packages. Follow
the instrucions in the following [link](http://continuum.io/blog/raspberry).

To use *CameraNetwork* the following packages need to be installed:

* setuptools (> wget https://bootstrap.pypa.io/ez_setup.py -O - | python)
* paramiko (using easy_install from setuptools)
* cython (for installing the latest version of zmq)
* zmq (tested against version 4.0.4) - should download and compile from source.
* pyzmq (tested against version 14.0.0dev) - should download and compile from source.
* pip (should be helpful in installing other stuff)
* tornado
* futures
* numpy
* scipy
* matplotlib
* ipython
* opencv
* beautifulsoup4 (pip install beautifulsoup4)

The code that runs on the *raspberryPi* expects the user 'pi' and installs some binaries
on '/home/pi/.local/bin'. You need to add this path to '/home/pi/.bashrc'.

Install picamera
> pip install picamera
or
> sudo apt-get install python-picamera

### Install software for modem (reqiured for RaspberrPi)

Install Network-Manger:

> sudo apt-get install network-manager
> sudo apt-get install network-manager-gnome

The first instal nmcli (used for activating the connection). The second intalls nmcli-connection-editor
used for defining the mobile network connection.

Install a recent version of usb_modeswitch (required on raspberryPi). Follow the [link][http://www.draisberghof.de/usb_modeswitch/].
To compile the above code you will need to install the libusb-1 dev files:

> sudo apt-get install libusb-1.0-0-dev

Prepare a device reference file from the following [link][http://www.draisberghof.de/usb_modeswitch/device_reference.txt] and run
it using the command:

> sudo usb_modeswitch -c <path to device file>

Package Installation
--------------------
Run the setup.py code

> python setup.py install

if you use *miniconda* you don't need root privileges. Or,

> python setup.py develop --user

Run the camera setup script to setup the camera environment.

> setup_camera.py


Deployment
----------
Run 'setup_camera.py' and enter a unique camera id and other necessary parameters.

To start at boot edit the rc.local file

> sudo nano /etc/rc.local

Add the following line before the 'exit 0' line in pi:
> exec 2> /tmp/rc.local.log
> exec 1>&2
> set -x
>
> sleep 40
> nmcli -p con up id "mobile"
> sleep 20
>
> su -l pi -c "/usr/bin/screen -dmS sn_tunnel bash -c '/home/pi/miniconda/bin/start_tunnel.py; exec bash'" &
> su -l pi -c "/usr/bin/screen -dmS sn_camera bash -c '/home/pi/miniconda/bin/start_camera.py; exec bash'" &

In the odroid:
> exec 2> /tmp/rc.local.log
> exec 1>&2
> set -x
>
> sleep 40
> nmcli -p con up id "mobile"
> sleep 20
>
> su -l odroid -c "/usr/bin/screen -dmS sn_tunnel bash -c '/home/odroid/.local/bin/start_tunnel.py; exec bash'" &
> su -l odroid -c "/usr/bin/screen -dmS sn_camera bash -c '/home/odroid/.local/bin/start_camera.py; exec bash'" &

<TODO: See if the next is relevant to newer images>
In the current image we need to install ntpdate, sshpass, screen:
> sudo apt-get install ntpdate
> sudo apt-get install sshpass
> sudo apt-get install screen

If there is a need to connect to the raspberryPi directly from a computer using
a network cable, then there is a need to set a static ip. To do so edit the file:
Note: It is better to use a switch or router instead.

    /etc/network/interfaces

And change from 'dhcp' to 'static':

    iface eth0 inet static
    address 192.168.0.101
    gateway 192.168.0.1
    netmask 255.255.255.0

The following [link][http://www.sudo-juice.com/how-to-set-a-static-ip-in-ubuntu-the-proper-way/] might
be helpful.

It is possible that for enabling the reverse ssh creation. We need to ssh one time to store the id or
something else in the ssh keys or something.

### Time

The odroid comes with timezone set to Australia and sync time will not work for it. Follow
the [link][http://www.wikihow.com/Change-the-Timezone-in-Linux] to set a different timezone.

### Modem

Define a new network connection (note the gksudo which is important when connected through ssh,
in this situation it is not possible to open xwindow apps when using just sudo)

> gksudo nm-connection-editor

Call the connection 'mobile'. Then store the connection password in the connection file:

> sudo vi /etc/NetworkManager/system-connections/mobile

Change the line 'password-flags=1' to 'password=<pass>' where <pass> is the mobile provider password.

### Sudo

It is needed to add the ability to sudo without password in the odroid. To do so add a file inside
the folder '/etc/sudoers.d/' with the line:

> odroid ALL=(ALL) NOPASSWD: ALL

### Allow Reverse SSH connection without acknowledgement:


Add the following lines to the beginning of /etc/ssh/ssh_config . Taken from this [link][http://superuser.com/questions/125324/how-can-i-avoid-sshs-host-verification-for-known-hosts]

> Host 192.168.0.*
>    StrictHostKeyChecking no
>    UserKnownHostsFile=/dev/null


Proxy Server
------------

Currently the code assumes that the proxy server is and ec2 instance.
You need to install the package on the proxy server too.

To allow the connection to the tunnel you need to allow gatewayports on the proxy. Instructions taken
from [http://www.vdomck.org/2005/11/reversing-ssh-connection.html][link]. Edit /etcs/ssh/sshd_config
and make sure the following options are set:

TCPKeepAlive yes
ClientAliveInterval 30
ClientAliveCountMax 99999
GatewayPorts yes

To run the proxy program, do:

> start_proxy.py

Calibration
-----------

The arduino sketch requires the 'old' makeblock [https://github.com/Makeblock-official/Makeblock-Library][libraries].

To allow control of the USB650, I had to install [https://github.com/ap--/python-oceanoptics][python-oceanoptics].
This library requires pyusb and to copy the libusb-1.0 dll to the 'scripts' folder of the python installation.
This dll can be downloaded from the following [http://sourceforge.net/projects/libusb/files/libusb-1.0/libusb-1.0.20/libusb-1.0.20.7z/download][link].
The following was a useful [http://sourceforge.net/p/pyusb/mailman/message/34745872/][link].
* 