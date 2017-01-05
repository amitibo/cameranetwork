#!/usr/bin/env python

"""
CameraNetwork: Code for running and analyzing the Camera Network

Authors:
Amit Aides <amitibo@campus.technion.ac.il>
URL: <http://bitbucket.org/amitibo/CameraNetwork>
License: See attached license file
"""

from setuptools import setup
import platform
import glob
import os

NAME = 'CameraNetwork'
PACKAGE_NAME = 'CameraNetwork'
PACKAGES = [PACKAGE_NAME]
VERSION = '0.1'
DESCRIPTION = 'Code for running and analyzing the Camera Network.'
LONG_DESCRIPTION = """
Code for running and analyzing the Camera Network.
"""
AUTHOR = 'Amit Aides'
EMAIL = 'amitibo@tx.technion.ac.il'
KEYWORDS = []
LICENSE = 'GPLv3'
CLASSIFIERS = [
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'Development Status :: 2 - Pre-Alpha',
    'Topic :: Scientific/Engineering'
]
URL = "http://bitbucket.org/amitibo/CameraNetwork"


def choose_scripts():

    scripts = [
        'scripts/start_tunnel.py',
        'scripts/start_server.py',
        'scripts/setup_camera.py',
        'scripts/monitor_folder.py',
        'scripts/drone_camera.py',
        'scripts_client/setup_config_server.py',
        'scripts_proxy/start_proxy.py'
    ]

    return scripts


def main():
    """main setup function"""

    s = setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        license=LICENSE,
        packages=PACKAGES,
        scripts=choose_scripts(),
    )

    if platform.system() != 'Windows':
        try:
            import shutil

            home_path = os.path.expanduser('~')
            bin_path = os.path.join(home_path, '.local/bin')

            if os.uname()[1] == 'raspberrypi':
                src_bin = 'bin_pi/*'
            elif os.uname()[1] == 'odroid':
                src_bin = 'bin_odroid/*'
            else:
                raise

            for file_name in glob.glob(src_bin):
                print 'copying {src} to {dst}'.format(src=file_name, dst=bin_path)
                shutil.copy(file_name, bin_path)
        except:
            print 'Failed copying the skagis executable'


if __name__ == '__main__':
    main()
