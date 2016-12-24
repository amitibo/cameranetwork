from __future__ import division
from CameraNetwork.calibration import VignettingCalibration
import glob
import os
import traceback


def main(base_path):
    #
    # Load the measurements
    #
    camera_paths = \
        [p for p in glob.glob(r'{}\*'.format(base_path)) if os.path.isdir(p)]

    for path in camera_paths:
        try:
            vc = VignettingCalibration.readMeasurements(path)
            vc.save(os.path.join(path, '.vignetting.pkl'))
        except:
            print("Failed processing path: {}\n{}".format(path, traceback.format_exc()))
            

if __name__ == '__main__':
    main('radiometric_calibration')