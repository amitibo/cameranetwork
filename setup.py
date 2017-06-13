#!/usr/bin/env python
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

"""
CameraNetwork: Code for running and analyzing the Camera Network

Authors:
Amit Aides <amitibo@campus.technion.ac.il>
URL: <https://bitbucket.org/amitibo/CameraNetwork_git>
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
