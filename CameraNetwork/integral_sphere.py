from __future__ import division, absolute_import, print_function
from CameraNetwork.image_utils import raw2RGB
import numpy as np
import os
import re


def parseData(path):
    """Parse the data file documenting the measurement. Example file:
    
distance: 10.8 cm
temp: 2819K
Lamp: QH (Quartz Halogene)
Intensity: 808 candela/M^2  236 [Fl](Foot Lambert)
    """
    
    distance, intensity = None, None
    with open(os.path.join(path, "data.txt"), 'r') as f:
        for line in f:
            res = re.search(r'distance:\s+([\d.]+)', line.strip())
            if res is not None:
                distance = float(res.groups()[0])
                continue
            
            res = re.search(r'Intensity:\s+(\d+)\s+candela/M\^2\s+(\d+)\s+', line.strip())
            if res is not None:
                intensity, candela = [float(i) for i in res.groups()]
            
    return distance, intensity, candela


def measureRGB(raw_array):
    """Measure mean rgb values in raw array captured of integral sphere."""
    
    R, G, B = raw2RGB(raw_array)
    RGB = np.dstack((R, G, B))
    masks = [(C>0.75*C.max()) & (C<255) for C in (R, G, B)]
    r, g, b = [np.mean(C[mask]) for C, mask in zip((R, G, B), masks)]
    
    return r, g, b
    

