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
"""Calculate the empirical ration between the sunphotometer and its closest camera.
Note:
If the cameras are calibrated, the result should be basically a unit graph (with partial deviations).
"""
import CameraNetwork
from CameraNetwork import sunphotometer as spm
import cv2
import logging
import matplotlib
# We want matplotlib to use a QT backend or else we get an error about qt5.
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

#SUNPHOTOMETER_TIME = '2016-10-23 05:13:07'
SUNPHOTOMETER_TIME = '2016-10-23 09:30:45'


def visualize(almucantar_angles, almucantar_values, almucantar_coords, img_samples, img):
    
    plt.figure()
    plt.plot(almucantar_angles, almucantar_values)    

    plt.figure()
    plt.plot(almucantar_angles, samples[:,0], 'r')
    plt.plot(almucantar_angles, samples[:,1], 'g')
    plt.plot(almucantar_angles, samples[:,2], 'b')
    
    #
    # Mark the Almucantar on the image.
    #
    plt.figure()
    img = np.ascontiguousarray(img)
    for x, y, angle in zip(almucantar_coords[0], almucantar_coords[1], almucantar_angles):
        if angle > 0:
            color = (255, 0, 0)
        else:
            color = (255, 255, 0)
            
        cv2.circle(img, (int(x), int(y)), 2, color)
        
    plt.imshow(img)
    plt.show()


def getSunphotometerImageData(camera_client, spm_df, camera_df, sunphotometer_time=SUNPHOTOMETER_TIME):
    """"""

    #
    # Get sunphotometer data for a specific datetime.
    #
    almucantar_angles, almucantar_values = spm.readSunPhotoMeter(spm_df, sunphotometer_time, sun_angles=10)
    
    #
    # Get camera  (near sunphotometer) img for the same timestamp
    #
    closest_time = spm.findClosestImageTime(camera_df, sunphotometer_time, hdr='3')
    img, img_data = camera_client.seek('102', closest_time, '3', 301)    

    #
    # Sample the data from the img
    #
    #img_samples, almucantar_angles, almucantar_coords = \
    almucantar_samples, almucantar_angles, almucantar_coords, \
        _, _, _ = \
        spm.sampleImage(img, img_data, almucantar_angles=almucantar_angles)
    
    print("Sunphotometer time: {}, Image time: {}".format(sunphotometer_time, closest_time))

    return almucantar_angles, almucantar_values, almucantar_coords, almucantar_samples, img


def main():
    #
    # Start the cameras client.
    #
    camera_client = CameraNetwork.CLIclient()
    proxy_params = CameraNetwork.retrieve_proxy_parameters(local_mode=True)
    camera_client.start(proxy_params)    

    #
    #
    #
    day = '2016-10-19'
    camera_df = camera_client.query('102', day)
    
    #
    # Wait for the cameras to connects
    #
    time.sleep(2)
    
    #
    # Load the sunphotometer data.
    #
    alm_df = spm.parseSunPhotoMeter(r'../data/aeronet/2016_oct/161001_161031_Technion_Haifa_IL.alm')
    alm_df = alm_df[day]    
    ppl_df = spm.parseSunPhotoMeter(r'../data/aeronet/2016_oct/161001_161031_Technion_Haifa_IL.ppl')
    ppl_df = ppl_df[day]
    
    #
    # Process the readings.
    #
    sunphotometer_measurements = []
    image_measurements = []
    for sunphotometer_time in alm_df[alm_df['Wavelength(um)']==0.4405].index:
        almucantar_angles, almucantar_values, almucantar_coords, img_samples, img = \
            getSunphotometerImageData(camera_client, alm_df, camera_df, sunphotometer_time=sunphotometer_time)
        sunphotometer_measurements.append(almucantar_values)
        image_measurements.append(img_samples[:,2])
        
    sm = np.array(sunphotometer_measurements)
    im = np.array(image_measurements)
    ratio = sm / im
    
    plt.figure()
    plt.plot(almucantar_angles, np.mean(ratio, axis=0))
    
    plt.figure()
    plt.plot(almucantar_angles, np.std(ratio, axis=0))
    
    plt.show()


if __name__ == '__main__':
    main()

