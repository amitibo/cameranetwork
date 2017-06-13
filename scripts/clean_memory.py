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
Clean memory of the odroid.

The script moves cpatured date to a backup folder. To remove
the backup folder (and clear the memory) use the ``--delete`` flag.
"""

import argparse
import CameraNetwork.global_settings as gs
import datetime
import os
import shutil
import warnings

gs.initPaths()

BACKUP_FOLDER = os.path.expanduser(
    datetime.datetime.now().strftime("~/BACKUP_%Y_%m_%d")
)


def move(src_path):

    _, tmp = os.path.split(src_path)
    dst_path = os.path.join(BACKUP_FOLDER, tmp)

    if not os.path.exists(src_path):
        print("Source path does not exist: {}".format(src_path))
        return

    assert not os.path.exists(dst_path),"Destination path exists: {}".format(dst_path)

    shutil.move(src_path, dst_path)


def main(delete_backup=False):

    if not os.path.exists(BACKUP_FOLDER):
        os.makedirs(BACKUP_FOLDER)
        print("Created backup folder: {}".format(BACKUP_FOLDER))

    move(gs.CAPTURE_PATH)
    move(gs.DEFAULT_LOG_FOLDER)
    move(gs.MASK_PATH)
    move(gs.SUN_POSITIONS_PATH)

    if delete_backup:
        answer = raw_input("Remove backup? [y/n]:")

        if answer == 'y':
            shutil.rmtree(BACKUP_FOLDER)
            print("Backup folder removed!")
        else:
            print("Backup folder NOT removed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean memory of the odroid.")
    parser.add_argument(
        '--delete',
        action='store_true',
        help='When set, the backup folder will be deleted.'
    )
    args = parser.parse_args()

    main(args.delete)
