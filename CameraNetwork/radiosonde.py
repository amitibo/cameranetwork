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
import numpy as np
import os
import pandas as pd
import pkg_resources
import requests
from CameraNetwork.utils import safe_make_dirs

BASE_ADDRESS = \
    r'http://weather.uwyo.edu/cgi-bin/sounding?region=mideast&TYPE=TEXT%3ALIST&YEAR={year}&MONTH={month}&FROM={day}11&TO={day}12&STNM=40179'
COLUMNS = ('PRES', 'HGHT', 'TEMP', 'DWPT', 'RELH', 'MIXR', 'DRCT', 'SKNT', 'THTA', 'THTE', 'THTV')
RADIOSONDES_PATH = \
    pkg_resources.resource_filename(__name__, '../data/radiosondes/')


def download_radiosonde(date, path):
    """Download the radiosonde from the online website."""

    data = []
    r = requests.get(
        BASE_ADDRESS.format(
        year=date.year, month=date.month, day=date.day))

    #
    # Parse the table
    #
    for line in r.text.split('\n')[10:]:
        if line.startswith(u'</PRE>'):
            #
            # End of table
            #
            break

        line_data = [float(s) for s in line.split()]
        if len(line_data) < len(COLUMNS):
            #
            # Ignore non full lines.
            #
            continue

        data.append(line_data)

    df = pd.DataFrame(data, columns=COLUMNS)
    df.to_csv(path)

    return df


def load_radiosonde(date):
    """Load a radiosonde according to a date."""

    folder = os.path.join(RADIOSONDES_PATH, '{}_{}'.format(date.year, date.month))
    safe_make_dirs(folder)

    path = os.path.join(folder, "{:02}.csv".format(date.day))

    if not os.path.exists(path):
        df = download_radiosonde(date, path)
    else:
        df = pd.read_csv(path)

    return df


if __name__ == '__main__':
    import datetime

    date = datetime.datetime.now() - datetime.timedelta(1)

    df = load_radiosonde(date)

    pass