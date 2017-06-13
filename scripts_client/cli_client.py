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
"""

from __future__ import division
import CameraNetwork
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy as np
import traceback

try:
    import winsound
    def playsound(frequency,duration):
        winsound.Beep(frequency,duration)
except ImportError:
    def playsound(frequency,duration):
        pass
    

GAINS_MAP = {
    'pi': 100,
    'ids1': 0,
    'ids2': 0,
    'mv': 0
}

def stats_per_channel(img):
    
    if len(img.shape) == 3:
        return [img.mean()], [img.std()]
    
    means = []
    stds = []
    stds_per_pixel = []
    for ch in range(3):
        means.append(img[...,ch].mean())
        stds.append(img[...,ch].std())
        stds_per_pixel.append(img[...,ch].std(axis=0).mean())
    
    return means, stds, stds_per_pixel
    

def main():
    
    c = CameraNetwork.CLIclient()
    proxy_params = CameraNetwork.retrieve_proxy_parameters()
    c.start(proxy_params)
    
    time.sleep(3)

    EXPOSURES = range(1000, 20000, 1000)
    for server in c.client_instance.servers:
        means = []
        stds = []
        for exposure in EXPOSURES:
            arrays = []
            for i in range(20):
                try:
                    arrays.append(c.get_array(server, exposure_us=exposure, gain_db=GAINS_MAP[server]))
                except:
                    pass
            arrays = np.array(arrays)
            
            mean, std = stats_per_channel(arrays)
            means.append(mean)
            stds.append(std)
    
        stds = np.array(stds)
        means = np.array(means)
        snrs  = 20*np.log10(means/stds)
        
        plt.rc('axes', color_cycle=['r', 'g', 'b'])
        
        plt.figure()
        plt.plot(EXPOSURES, means, linewidth=3.0)
        plt.title('{server}: means'.format(server=server))
        plt.figure()
        plt.plot(EXPOSURES, stds, linewidth=3.0)
        plt.title('{server}: stds'.format(server=server))
        plt.figure()
        plt.plot(EXPOSURES, snrs, linewidth=3.0)
        plt.title('{server}: snrs'.format(server=server))

    plt.show()

def measure_DN():
    c = CameraNetwork.CLIclient()
    proxy_params = CameraNetwork.retrieve_proxy_parameters()
    c.start(proxy_params)
    
    time.sleep(3)
    
    for server in c.client_instance.servers:
        arrays = []
        for i in range(10):
            try:
                arrays.append(c.get_array(server, exposure_us=10000, gain_db=GAINS_MAP[server]))
            except Exception:
                print 'bad frame from server {server}'.format(server=server)
                print traceback.format_exc()
                
            playsound(5000, 100)
            
        arrays = np.array(arrays)
        mean, std = stats_per_channel(arrays)
    
        print 'Server: {server}, Mean: {mean}, STD: {std}'.format(server=server, mean=mean, std=std)
        playsound(1000, 500)

    
def measure_DN():
    c = CameraNetwork.CLIclient()
    proxy_params = CameraNetwork.retrieve_proxy_parameters()
    c.start(proxy_params)
    
    time.sleep(3)
    
    for server in c.client_instance.servers:
        arrays = []
        for i in range(100):
            try:
                array = c.get_array(server, exposure_us=10000, gain_db=GAINS_MAP[server])
                arrays.append(array)
            except Exception:
                print 'bad frame from server {server}'.format(server=server)
                print traceback.format_exc()
                
            playsound(5000, 100)
            
        arrays = np.array(arrays)
        mean, std, std_per_pixel = stats_per_channel(arrays)
    
        print 'Server: {server}, Mean: {mean}, STD: {std}, STD_PER_PIXEL: {std_per_pixel}'.format(server=server, mean=mean, std=std, std_per_pixel=std_per_pixel)
        playsound(1000, 500)

    
def check_bayer():
    c = CameraNetwork.CLIclient()
    proxy_params = CameraNetwork.retrieve_proxy_parameters()
    c.start(proxy_params)
    
    time.sleep(3)
    
    server = 'ids2'
    array = c.get_array(server, exposure_us=10000, gain_db=GAINS_MAP[server])
    
    #plt.subplot(221)
    #plt.imshow(array[...,0], interpolation='nearest')
    #plt.title('red')
    #plt.subplot(222)
    #plt.imshow(array[...,1], interpolation='nearest')
    #plt.title('blue')
    #plt.subplot(223)
    #plt.imshow(array[...,2], interpolation='nearest')
    #plt.title('greed')
    plt.rc('axes', color_cycle=['r', 'g', 'b'])
    plt.plot(array[100,...])
    plt.figure()
    plt.imshow(array)
    plt.show()


if __name__ == '__main__':
    measure_DN()

    
