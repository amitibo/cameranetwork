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

    
