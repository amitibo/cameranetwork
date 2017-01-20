#!/usr/bin/env python
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
