from __future__ import division

from CameraNetwork.calibration import Gimbal, GimbalCamera, findSpot
from CameraNetwork.image_utils import raw2RGB
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
from skycameras import FisheyeProxy, Normalization
import itertools
import traceback
import cPickle

SLEEP_TIME = 1
SAMPLING_STEPS = 32

COLORS = ('blue', 'green', 'red')
EXPOSURE_TIME = dict(zip(COLORS, (35, 35, 30)))
DO_COLORS = COLORS#('red',)
DO_GEOMETRIC_CALIBRATION = True
DO_BLACK_IMG = True

#
# Calibration params
#
#NX, NY = 10, 7
NX, NY = 8, 6

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


def main():

    import oceanoptics
    spec = oceanoptics.get_a_random_spectrometer()
    
    p = Gimbal(com="COM3")
    cam = GimbalCamera()

    #
    # Put here a break point if you want to adjust the focus.
    # Note: You will have to stop the program to be able to turn on the IDS cockpit.
    #
    p.move(90, 90)

    base_path = os.path.join('radiometric_calibration', cam._cam.info['serial_num'])
    safe_mkdirs(base_path)    

    #
    # Store the exposure time
    #
    with open(os.path.join(base_path, 'exposures.pkl'), 'wb') as f:
        cPickle.dump(EXPOSURE_TIME, f)
        
    if DO_GEOMETRIC_CALIBRATION:
        #
        # Geometric Calibration
        #
        imgs = []
        imgsR = []
        for i, (x, y) in enumerate(itertools.product(np.linspace(0, 120, 15), np.linspace(35, 165, 15))):
            print x, y
            p.move(int(x), int(y))
            time.sleep(1.5)
            img = cam.measure()
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
    
    if DO_BLACK_IMG:
        #
        # Measure noise
        #
        p.move(90, 90)
        time.sleep(SLEEP_TIME)
        raw_input("Turn off the lights and press any key")
        winsound.Beep(5000, 500)
    
        for color in COLORS:
            exposure = EXPOSURE_TIME[color]
            
            black_img = cam.measure(exposure=exposure, average=100)
            winsound.Beep(5000, 500)
            np.save(os.path.join(base_path, 'black_img_{}.npy'.format(color)), black_img)        

    for color in DO_COLORS:
        #
        # Make path to store img measurements.
        #
        safe_mkdirs(os.path.join(base_path, color))
        
        exposure = EXPOSURE_TIME[color]

        raw_input("Turn on the Colibri led {}, and press any key".format(color))
        winsound.Beep(5000, 500)
        time.sleep(SLEEP_TIME)
        
        #
        # Measure the spectrum
        #
        spec.integration_time(0.3)
        measurements = []
        for i in range(10):
            s = spec.spectrum()
            measurements.append(s[1])    
    
        with open(os.path.join(base_path, 'spec_{}.pkl'.format(color)), 'wb') as f:
            cPickle.dump((s[0], np.mean(measurements, axis=0)), f)    
        #
        # Verify no saturation.
        #
        p.move(90, 90)
        time.sleep(SLEEP_TIME)
        if cam.measure(exposure=exposure).max() == 255:
            raise Exception('Saturation')
        
        X_grid, Y_grid = np.meshgrid(np.linspace(0, 180, SAMPLING_STEPS), np.linspace(0, 180, SAMPLING_STEPS), indexing='xy')
        X_grid = X_grid.astype(np.int32)
        Y_grid = Y_grid.astype(np.int32)
        
        black_img = np.load(os.path.join(base_path, 'black_img_{}.npy'.format(color)))        
        
        measurements = []
        for i, (x, y) in enumerate(zip(X_grid.ravel(), Y_grid.ravel())):
            
            sys.stdout.write('x={}, y={}...\t'.format(x, y))
            
            p.move(x, y)
            time.sleep(SLEEP_TIME)
            img = cam.measure(exposure=exposure)
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
            img_path = os.path.join(base_path, color, 'img_{:03}.mat'.format(i))
            sio.savemat(img_path, {'img':img}, do_compression=True)
                
        img = np.zeros(shape=(1200, 1600, 3))
        
        for x, y, val in measurements:
            if val is None:
                continue
            img[y, x, ...] = val
        
        #
        # Save the results
        #
        sio.savemat(os.path.join(base_path, 'results_{}.mat'.format(color)), {'img':img}, do_compression=True)
        with open(os.path.join(base_path, 'measurements_{}.pkl'.format(color)), 'wb') as f:
            cPickle.dump(measurements, f)

        plt.imshow(img)
        plt.title(color)
        plt.show()

if __name__ == '__main__':
    main()
    