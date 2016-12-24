#!/usr/bin/env python
"""
This script monitors a folder for new files and uploads these to dropbox.
"""

import os
from pyinotify import WatchManager, Notifier
from pyinotify import EventsCodes, ProcessEvent
from CameraNetwork import global_settings as gs
from datetime import datetime
import json
import logging
import subprocess
import Queue
import thread


BASE_FOLDER = '/home/odroid/test_imgs'
DROPBOX_FOLDER = 'DRONE_UPLOADS/{time_signature}'.format(
    time_signature=datetime.now().strftime("%Y%m%d_%H:%M:%S.%f")
)


def upload_thread(upload_queue):
    """A thread for uploading captured images."""

    while True:
        #
        # Wait for a new upload
        #
        capture_path, upload_path = upload_queue.get()

        #
        # Check if time to quit
        #
        if capture_path is None:
            logging.info('Upload thread stopped')
            break

        #
        # Check if the file is an image.
        #
        if capture_path.endswith('.jpg'):
            #
            # Resize the image to enable upload to dropbox.
            #
            tmp_path = os.path.join(BASE_FOLDER, 'temp_resized.jpg')
            resize_cmd = 'convert {src_path} -resize 640x480 {dst_path}'.format(
                src_path=capture_path,
                dst_path=tmp_path
            )

            resize_log = subprocess.Popen(resize_cmd, shell=True).communicate()
            print resize_log
        else:
            tmp_path = capture_path

        cmd_upload = gs.UPLOAD_CMD.format(
            capture_path=tmp_path,
            upload_path=upload_path
        )

        logging.info('Uploading frame %s' % upload_path)

        #
        # Upload image
        #
        upload_log = subprocess.Popen(cmd_upload, shell=True).communicate()

        logging.debug(str(upload_log))


class PTmp(ProcessEvent):
    def __init__(self, upload_queue, *params, **kwds):
        self.upload_queue = upload_queue
        super(PTmp, self).__init__(*params, **kwds)

    def process_IN_CREATE(self, event):
        if event.name.endswith('.txt'):
            data_path = os.path.join(event.path, event.name)
            dst_path = os.path.join(DROPBOX_FOLDER, event.name)

            self.upload_queue.put((data_path, dst_path))
            self.upload_queue.put((data_path[:-3]+'jpg', dst_path[:-3]+'jpg'))


def main():

    upload_queue = Queue.Queue()
    thread.start_new_thread(upload_thread, (upload_queue,))

    #
    # Setup a watcher on the images folder.
    #
    wm = WatchManager()
    notifier = Notifier(wm, PTmp(upload_queue))
    wm.add_watch(BASE_FOLDER, EventsCodes.OP_FLAGS['IN_CREATE'], rec=True)

    #
    # loop forever
    #
    while True:
        try:
            #
            # process the queue of events
            #
            notifier.process_events()
            if notifier.check_events():
                #
                # read notified events and enqeue them
                #
                notifier.read_events()

        except KeyboardInterrupt:
            #
            # destroy the inotify's instance on this
            # interrupt (stop monitoring)
            #
            notifier.stop()

            #
            # Stop the upload queue.
            #
            upload_queue.put((None, None))
            break


if __name__ == '__main__':
    main()
