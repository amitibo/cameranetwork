from __future__ import division
from CameraNetwork.calibration import VignettingCalibration
from CameraNetwork.cameras import IDSCamera
from CameraNetwork import Gimbal, findSpot
from CameraNetwork.image_utils import raw2RGB
import CameraNetwork.global_settings as gs
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import cPickle
import winsound
import ids
import sys
import os
import scipy.io as sio
import fisheye
from CameraNetwork.image_utils import FisheyeProxy, Normalization
import itertools
import traceback
import cPickle

SLEEP_TIME = 0.1

DO_GEOMETRIC_CALIBRATION = True
GEOMETRIC_STEPS = 12

DO_BLACK_IMG = True

VIGNETTING_STEPS = 32

#
# Calibration params
#
NX, NY = 8, 6
GEOMETRIC_EXPOSURE = 120000
VIGNETTING_EXPOSURE = 65000
LED_POWER = {"BLUE(470)": 35, "GREEN(505)": 100, "RED(625)": 50}


def safe_mkdirs(path):
    """Safely create path, warn in case of race."""

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            import errno
            if e.errno == errno.EEXIST:
                warnings.warn(
                    "Failed creating path: {path}, probably a race".format(path=path)
                )


def capture_callback(img, exp, gain):
    """Debug plot of the image."""

    cv2.imshow("image", np.array(img).astype(np.uint8))
    cv2.waitKey(1)


def main():

    import oceanoptics
    spec = oceanoptics.get_a_random_spectrometer()


    p = Gimbal(com="COM5")

    #
    # Put here a break point if you want to adjust the focus.
    # Note: You will have to stop the program to be able to turn on the IDS cockpit.
    #
    p.move(90, 90)

    cv2.namedWindow("image", flags=cv2.WINDOW_NORMAL)
    cam = IDSCamera(callback=capture_callback)

    base_path = os.path.join('vignetting_calibration', cam.info['serial_num'])
    safe_mkdirs(base_path)
    safe_mkdirs(os.path.join(base_path, 'images'))

    #
    # Capture settings.
    #
    settings = {
        "exposure_us": GEOMETRIC_EXPOSURE,
        "gain_db": 0,
        "gain_boost": False,
        "color_mode": gs.COLOR_RAW
    }

    #
    # Store the exposure time
    #
    with open(os.path.join(base_path, 'settings.pkl'), 'wb') as f:
        cPickle.dump(settings, f)

    if DO_GEOMETRIC_CALIBRATION:
        #
        # Geometric Calibration
        #
        imgs = []
        imgsR = []
        for i, (x, y) in \
            enumerate(itertools.product(
                np.linspace(35, 165, GEOMETRIC_STEPS),
                np.linspace(0, 120, GEOMETRIC_STEPS))):
            print x, y
            p.move(int(x), int(y))
            time.sleep(2.5)
            img, _, _ = cam.capture(settings, frames_num=1)
            imgs.append(img)
            imgsR.append(raw2RGB(img)[0].astype(np.uint8))

        #
        # Use the fisheye model
        #
        fe = fisheye.FishEye(nx=NX, ny=NY, verbose=True)

        rms, K, D, rvecs, tvecs = fe.calibrate(
            imgs=imgsR,
            show_imgs=False
        )

        normalization = Normalization(1001, FisheyeProxy(fe))
        img_normalized = normalization.normalize(imgsR[0])
        plt.imshow(img_normalized, cmap='gray')
        plt.show()

        #
        # Save the geometric calibration
        #
        fe.save(os.path.join(base_path, 'fisheye.pkl'))
    else:
        fe = fisheye.load_model(os.path.join(base_path, 'fisheye.pkl'))

    settings["exposure_us"] = VIGNETTING_EXPOSURE

    if DO_BLACK_IMG:
        #
        # Measure noise
        #
        p.move(90, 90)
        time.sleep(SLEEP_TIME)
        raw_input("Turn off the lights and press any key")
        winsound.Beep(5000, 500)

        black_img = np.mean(cam.capture(settings, frames_num=10)[0], axis=2)
        winsound.Beep(5000, 500)
        np.save(os.path.join(base_path, 'black_img.npy'), black_img)

    #
    # Make path to store img measurements.
    #
    safe_mkdirs(base_path)

    #
    # Print the COLIBRI LED POWER
    #
    raw_input("Set COLIBRI LEDS: {}, and press any key.".format(
        [(k, v) for k, v in LED_POWER.items()]))

    #
    # Measure the spectrum
    #
    spec.integration_time(0.3)
    measurements = []
    for i in range(10):
        s = spec.spectrum()
        measurements.append(s[1])

    with open(os.path.join(base_path, 'spec.pkl'), 'wb') as f:
        cPickle.dump((s[0], np.mean(measurements, axis=0)), f)

    #
    # Verify no saturation.
    #
    p.move(90, 90)
    time.sleep(SLEEP_TIME)
    img, exp, gain = cam.capture(settings, frames_num=1)
    if img.max() == 255:
        raise Exception('Saturation')

    print("Maximal color values: {}".format([c.max() for c in raw2RGB(img)]))

    X_grid, Y_grid = np.meshgrid(
        np.linspace(0, 180, VIGNETTING_STEPS),
        np.linspace(0, 180, VIGNETTING_STEPS),
        indexing='xy')
    X_grid = X_grid.astype(np.int32)
    Y_grid = Y_grid.astype(np.int32)

    black_img = np.load(os.path.join(base_path, 'black_img.npy'))

    measurements = []
    for i, (x, y) in enumerate(zip(X_grid.ravel(), Y_grid.ravel())):

        sys.stdout.write('x={}, y={}...\t'.format(x, y))

        p.move(x, y)
        time.sleep(SLEEP_TIME)
        img = np.mean(cam.capture(settings, frames_num=10)[0], axis=2)
        winsound.Beep(8000, 500)
        try:
            measurement = findSpot(np.clip(img-black_img, 0, 255))
        except:
            print 'FAIL'
            print traceback.format_exc()
            measurement = None, None, None

        measurements.append(measurement)
        print measurement

        #
        # Store the measurement image.
        #
        img_path = os.path.join(base_path, 'images', 'img_{:03}.mat'.format(i))
        sio.savemat(img_path, {'img':img}, do_compression=True)

    img = np.zeros(shape=(1200, 1600, 3))

    for x, y, val in measurements:
        if val is None:
            continue
        img[y, x, ...] = val

    #
    # Save the results
    #
    sio.savemat(os.path.join(base_path, 'results.mat'), {'img':img}, do_compression=True)
    with open(os.path.join(base_path, 'measurements.pkl'), 'wb') as f:
        cPickle.dump(measurements, f)

    #
    # Calculate Vignetting correction.
    #
    vc = VignettingCalibration.readMeasurements(base_path)
    vc.save(os.path.join(base_path, '.vignetting.pkl'))
    print("The STD Vignetting Error per color is: {}".format(vc._stds))

    #
    # Visualize the vingetting
    #
    normalization = Normalization(
        gs.DEFAULT_NORMALIZATION_SIZE, FisheyeProxy(fe)
    )
    d = normalization.normalize(np.dstack(raw2RGB(vc.ratio)))
    plt.imshow(d)
    plt.show()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
