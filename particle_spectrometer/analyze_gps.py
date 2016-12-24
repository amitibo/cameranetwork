from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import os


def main():
    file_paths = sorted(glob.glob('data/2016_07_13_10_30_34_658403/*.json'))
    file_paths = sorted(glob.glob('data/2016_07_13_12_16_33_085584/*.json'))

    data = []
    lat = []
    lon = []
    alt = []
    relative_alt = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            d = json.load(f)
            data.append(d['data'])
            if d['coords'] is not None:
                if d['coords']['lat'] == 0:
                    continue
                lat.append(d['coords']['lat']*1e-7)
                lon.append(d['coords']['lon']*1e-7)
                alt.append(d['coords']['alt']*1e-3)
                relative_alt.append(d['coords']['relative_alt']*1e-3)
                
    data = np.array(data[:-1])
    plt.figure()
    plt.plot(data)
    plt.figure()
    plt.plot(alt, label='alt')
    plt.plot(relative_alt, label='relative alt')
    plt.legend()
    plt.figure()
    plt.scatter(lon, lat)
    plt.axis('equal')
    plt.show()
    

if __name__ == '__main__':
    main()