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
from __future__ import division
import datetime
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d

COLUMNS = [0.25, 0.28, 0.3, 0.35, 0.4, 0.45, 0.5, 0.58, 0.65, 0.7, 0.8, 1, 1.3, 1.6, 2, 2.5, 3, 3.5, 4, 5, 6.5, 7.5, 8.5, 10, 12.5, 15, 17.5, 20, 25, 30, 32]

def main():
    file_paths = sorted(glob.glob('data/2016_07_13_10_30_34_658403/*.json'))
    file_paths = sorted(glob.glob('data/2016_07_13_12_16_33_085584/*.json'))
    file_paths = sorted(glob.glob('data/2017_04_22_09_57_54_040000/*.json'))

    data = []
    indices = []
    lat = []
    lon = []
    alt = []
    relative_alt = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            d = json.load(f)
            if len(d['data'])==0 or d['coords'] is None or d['coords']['lat'] == 0:
                #
                # ignore corrupt data.
                #
                continue

            t = datetime.datetime(*[int(i) for i in os.path.split(file_path)[-1].split('.')[0].split('_')])
            indices.append(t)
            data.append(d['data'])
            lat.append(d['coords']['lat']*1e-7)
            lon.append(d['coords']['lon']*1e-7)
            alt.append(d['coords']['alt']*1e-3)
            relative_alt.append(d['coords']['relative_alt']*1e-3)

    data = np.array(data)[..., :-1]
    df = pd.DataFrame(data=data, index=indices, columns=COLUMNS)
    ax = df.plot(figsize=(12, 12), title="Particle Distribution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Particles/Liter")

    columns = np.linspace(COLUMNS[0], COLUMNS[-1], 300)

    for ind in (-150, -100, -50):

        dist = df.iloc[ind].values
        dist_dense = interp1d(COLUMNS, dist)(columns)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        axes[0].semilogy(COLUMNS, dist)
        axes[0].set_xlabel("Sizes um")
        axes[0].set_ylabel("Particles/Liter")

        axes[1].loglog(columns, 4. / 3 * np.pi * columns**3 * dist_dense)
        axes[1].set_xlabel("Sizes um")
        axes[1].set_ylabel("dV/d(log(r))")

        plt.suptitle("Particle Distribution at: {}".format(df.index[ind]))

    alt_df = pd.DataFrame(data={"Altitude":alt, "Relative Altitude":relative_alt}, index=indices, columns=["Altitude", "Relative Altitude"])
    ax = alt_df.plot(title="Flight Altitude")
    ax.set_xlabel("Time")
    ax.set_ylabel("Altitude [m]")

    plt.figure()
    plt.scatter(lon, lat, label="Flight")
    plt.axis('equal')
    plt.title("Geo Position")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    #
    # Add the Technion
    #
    data_x = [35.024967]
    data_y = [32.775730]
    plt.plot(data_x, data_y, 'or', label="Technion")

    #
    # Add Megido
    #
    data_x = [35.234186]
    data_y = [32.597122]
    plt.plot(data_x, data_y, '*b', label="Megido")

    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()