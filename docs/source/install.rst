.. highlight::sh

************
Installation
************

.. contents:: Table of Contents

Installation
============

The ``CameraNetwork`` system is made of three logical parts:

#. *Server*: The camera unit. The server performs the actual measurements.
#. *Client*: A program that enables remote control of servers.
#. *Proxy*: A program that bridges and manages the communication between the *Servers* and *Clients*.

There can be multiple *Servers* and *Clients* but only one *proxy*.

The ``CameraNetwork`` package contains the code for both the *Server*, *Client* and *Proxy* subsystems.
This simplifies the deployment and enables code reuse. The installation procedures is similar for the
three components but differs due to the different platforms.

The ``CameraNetwork`` is implemented completely in `Python <http://www.python.org/>_`.


Installation - Client
---------------------
#. Install conda. Tested on conda 4.7.11
#. Clone the cameranetwork package::

    git clone https://github.com/Addalin/cameranetwork.git
#. Navigate to it::

    cd cameranetwork

#. Create virtual env:

    LINUX: Create conda virtual environment from *cn_client_ubuntu18.yml*

    ::

        conda env create -f cn_client_ubuntu18.yml

    .. Note::

        The first line of sets the new environment's name (currently *cn_client*)

    WINDOWS (exact procedure)::

        # Create new environment with name: cn_client

        conda create -n cn_client --yes

        conda activate cn_client

        conda config --env --set restore_free_channel true

        conda config --env --append channels conda-forge

        conda install python=2.7 pip paramiko cython tornado=4.5.3 futures numpy scipy matplotlib beautifulsoup4 scikit-learn scikit-image pyside requests ephem pandas=0.19.2 ipython pyfirmata joblib pyzmq enaml pillow traits pyqtgraph pyopengl vtk mayavi opencv git mercurial



        # pip install pymap3d, traits-enaml and pyfisheye
        # Note, this installs pyfisheye without cloning it. For development of pyfisheye clone and install manually from https://bitbucket.org/amitibo/pyfisheye (TODO: migrate pyfisheye codebase to github)

        python -m pip install pymap3d==1.1.1 git+https://github.com/enthought/traits-enaml.git@update-data-frame-table  hg+https://bitbucket.org/amitibo/pyfisheye


        
#. Activate the environment::

    conda activate <venv_name>


#. Install the cameranetwork package

    ::

        python setup.py develop --user

    ..    note::

        without --user it installs the scripts for all users (Windows: C:\ProgramData\Anaconda2\Scripts)

#. Verify successful installation by opening the GUI::

    python scripts_client/camera_client.py


Installation - Server
---------------------

The server software is run on an `Odroid U3 <http://www.hardkernel.com/main/products/prdt_info.php?g_code=g138745696275>`_
as at the time of selection it offered a unique balance between capabilities and cost. Nonetheless it should be straight
forward to install the ``CameraNetwork`` package and its prerequisites on other platforms like newer Oroids and even
on the RaspberrPi.

In the following we detail the procedure of installing the required prerequisites and main package. Note that
once the package is installed on one computer, it is much more time effective to create an image of the Odroid
memory card and duplicate it as needed.


Installation - Proxy
--------------------

Currently the code assumes that the proxy server is run on an ec2 instance.
Installation on the proxy follows the same steps of installation on the
client.

Before running make sure to update in the global setting the ip address: `DEFUALT_PROXI_PARAMS <https://github.com/Addalin/cameranetwork/blob/52e915f60d54a1f2679cc2d675f689dd45c1a599/CameraNetwork/global_settings.py#L62-L71>`_
And make sure this is updated in all end units! 

To run the proxy program, do::
    python ./code/cameranetwork/scripts_proxy/start_proxy.py --log_level info
    
    or
    
    start_proxy.py

Installation - Calibration Station
----------------------------------


Camera setup
------------

Arduino connections
`````````````````````````

Savox SunShader Servo pins:

#. Brown (Gnd) = Gnd
#. Red (5V) = 5V
#. Orange (Signal) = PIN NUM



Installation - OLD
============================
Prerequisites
-------------

To use *CameraNetwork* several software package are needed. This can be installed using the following
commands. Copy paste these to a commandline::

    > sudo apt-get install python-pip git mercurial screen autossh
    > sudo pip install paramiko
    > sudo pip install cython
    > sudo pip install pyzmq --install-option="--zmq=bundled"
    > sudo pip install tornado==4.5.3
    > sudo pip install futures
    > sudo apt-get install python-numpy python-scipy python-matplotlib
    > sudo pip install beautifulsoup4
    > sudo pip install sklearn
    > sudo pip install skimage
    > sudo pip install ephem
    > sudo pip install pandas
    > sudo pip install pymap3d
    > sudo pip install ipython
    > sudo pip install pyfirmata
    > sudu pip install joblib

To install opencv3 follow a tutorial relevant to your system, e.g. on Odroid XU4 the following tutorial
was usefull `opencvsh_for_ubuntu_mate <https://github.com/nanuyo/opencvsh_for_ubuntu_mate>`_.

Install the python wrappers to the ids SDK::

    > mkdir code
    > cd code
    > git clone https://github.com/amitibo/ids.git
    > cd ids
    > sudo python setup.py install

Install the pyfisheye module::

    > cd ~/code
    > hg clone https://amitibo@bitbucket.org/amitibo/pyfisheye
    > cd pyfisheye
    > sudo python setup.py install

Some platforms might require the installation of modem software::

    > sudo apt-get install network-manager
    > sudo apt-get install network-manager-gnome

The first instal *nmcli* (used for activating the connection). The second intalls *nmcli-connection-editor*
used for defining the mobile network connection.

Install a recent version of usb_modeswitch (required on raspberryPi). Follow the `usb_modeswitch tutorial <http://www.draisberghof.de/usb_modeswitch/>`_.
To compile the above code you will need to install the *libusb-1* dev files::

    > sudo apt-get install libusb-1.0-0-dev

Prepare a device reference file from the following `device reference file <http://www.draisberghof.de/usb_modeswitch/device_reference.txt>`_ and run
it using the command::

    > sudo usb_modeswitch -c <path to device file>

CameraNetwork Installation
--------------------------

Download and install the package::

    > git clone https://amitibo@bitbucket.org/amitibo/cameranetwork_git.git cameranetwork
    > cd cameranetwork
    > python setup.py develop --user

.. note::

    The first command downloads a *slim* version of the code that only includes the *Server* components.

To make the system start automatically at boot time, we use the *rc.local* script::

    > sudo cp cameranetwork/scripts/rc.local/rc.local /etc/rc.local

Run the camera setup script to setup the camera environment.

    > setup_camera.py

You will be asked for a camera id. Enter a unique camera id number.



Installing the Client
---------------------

It is recommended to install python using the `Anaconda <https://www.continuum.io/downloads>`_ distribution.
Install the ``CameraNetwork`` package::

    > git clone https://amitibo@bitbucket.org/amitibo/cameranetwork_git.git cameranetwork
    > cd cameranetwork
    > python setup.py develop --user

Installing the Calibration Station
----------------------------------

It is recommended to install python using the `Anaconda <https://www.continuum.io/downloads>`_ distribution.
Install the ``CameraNetwork`` package::

    > git clone https://amitibo@bitbucket.org/amitibo/cameranetwork_git.git cameranetwork
    > cd cameranetwork
    > python setup.py develop --user



Shubi reference
---------------

#. Create conda virtual environment::

    conda create --name <venv_name> --no-default-packages
    conda config --add channels conda-forge
    conda activate cnvenv



#. Install prerequisites::

    conda install python=2.7 pip paramiko cython tornado=4.5.3 futures numpy scipy matplotlib beautifulsoup4 scikit-learn scikit-image ephem pandas ipython pyfirmata joblib
    pip install pyzmq --install-option="--zmq=bundled"
    pip install pymap3d
    conda install enaml pillow traits pyqtgraph pyopengl vtk mayavi opencv

#. Install additional modules::

    pip install ephem
    conda install -c anaconda pil
    conda install -c anaconda enaml
    conda install -c anaconda traits pyqtgraph pyopengl
    conda install -c anaconda vtk
    pip install mayavi

#. Install traits-enaml::

    git clone https://github.com/enthought/traits-enaml.git --branch update-data-frame-table
    cd traits-enaml
    python setup.py install
    cd..
    



#. Install the cameranetwork package
    #. Navigate back to cameranetwork::

        cd ..
    #. Install the cameranetwork package::

        python setup.py develop --user

    ..    note::

        without --user it installs the scripts for all users (Windows: C:\ProgramData\Anaconda2\Scripts)
