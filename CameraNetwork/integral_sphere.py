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
    

