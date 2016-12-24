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
BASE_PATH = \
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
    
    folder = os.path.join(BASE_PATH, '{}_{}'.format(date.year, date.month))
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