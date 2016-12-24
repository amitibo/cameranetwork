from __future__ import division, print_function, absolute_import
import cv2
import datetime
from dateutil import parser
import ephem
import numpy as np
import os
import pandas as pd

WAVELENGTHS = ['0.4405', '0.5000', '0.6744']


def calcAlmucantarPrinciplePlanes(
    latitude,
    longitude,
    capture_time,
    almucantar_angles=None,
    principleplane_angles=None,
    img_resolution=301,
    alm_resolution=72,
    pp_resolution=38):
    """Calculate the Almucantar and PrinciplePlanes for a specific
    localization, datetime.

    Args:
        almucantar_angles (array like): Sampling angles (degrees) along
            Almucantar.
        principleplane_angles (array like): Sampling angles (degrees) along
            PrinciplePlane.

    Note:    
         In practice the Almucantar and PrinciplePlane are measured in
         different times. So this function should be called twice for each
         with its own capture time.
    """
    
    #
    # Create an Sun/observer at camera position
    #
    observer = ephem.Observer()
    observer.lat, observer.long, observer.date = \
        str(latitude), str(longitude), capture_time

    sun = ephem.Sun(observer)

    #
    # Calculate Almucantar angles.
    # Almucantar samples at the altitude of the sun at differnt Azimuths. 
    start_az = -sun.az
    if almucantar_angles is None:
        #
        # Sample uniformly along Almucantar angles.
        #
        almucantar_angles = np.linspace(0, 360, alm_resolution, endpoint=False)
        alm_az = np.linspace(start_az, start_az+2*np.pi, alm_resolution, endpoint=False)
    else:
        alm_az = np.radians(almucantar_angles) + start_az
    alm_alts = sun.alt*np.ones(len(alm_az))

    #
    # Convert Almucantar angles to image coords.
    #
    alm_radius = (np.pi/2 - alm_alts)/(np.pi/2)
    alm_x = (-alm_radius * np.sin(alm_az) + 1) * img_resolution / 2
    alm_y = (alm_radius * np.cos(alm_az) + 1) * img_resolution / 2
    Almucantar_coords = np.array((alm_x, alm_y))
    
    #
    # Calculate PrinciplePlane coords.
    #
    start_alt = sun.alt
    if principleplane_angles is None:
        #
        # Sample uniformly along Almucantar angles.
        #
        principleplane_angles = np.linspace(0, 360, pp_resolution, endpoint=False)
        pp_alts = np.linspace(0, np.pi, pp_resolution, endpoint=False)
    else:
        pp_alts = np.radians(principleplane_angles) + start_alt
    pp_az = -sun.az * np.ones(len(pp_alts))    

    #
    # Convert Principal Plane angles to image coords.
    #
    pp_radius = (np.pi/2 - pp_alts)/(np.pi/2)
    pp_x = (-pp_radius * np.sin(pp_az) + 1) * img_resolution / 2
    pp_y = (pp_radius * np.cos(pp_az) + 1) * img_resolution / 2
    PrincipalPlane_coords = np.array((pp_x, pp_y))
    
    return Almucantar_coords, PrincipalPlane_coords, \
           almucantar_angles, principleplane_angles


def parseSunPhotoMeter(path):
    """Parse the sunphotometer data."""
    
    def dateparse(d, t):
        return pd.datetime.strptime(d+' '+t, '%d:%m:%Y %H:%M:%S')

    df = pd.read_csv(
        path,
        skiprows=3,
        header=0,
        parse_dates=[[0, 1]],
        date_parser=dateparse,
        index_col=0)

    return df


def findClosestImageTime(images_df, timestamp, hdr='2'):
    """Find the image taken closest to a given time stamp."""
    
    if type(timestamp) is str:
        timestamp = parser.parse(timestamp)
        

    #
    # Get a close index to time stamp.
    #
    hdr_df = images_df.xs(hdr, level='hdr')
    i = hdr_df.index.get_loc(timestamp, method='nearest')
    
    return hdr_df.iloc[i].name


def sampleImage(img, img_data, almucantar_angles=None, principleplane_angles=None):
    """Sample image RGB values along the Almucantar and PrincipalPlane."""

    latitude = img_data.latitude
    longitude = img_data.longitude
    capture_time = img_data.capture_time

    Almucantar_coords, PrincipalPlane_coords, \
        almucantar_angles, principleplane_angles = \
        calcAlmucantarPrinciplePlanes(
            latitude, longitude, capture_time, img_resolution=img.shape[0],
            almucantar_angles=almucantar_angles,
            principleplane_angles=principleplane_angles)

    img = np.ascontiguousarray(img)
    
    #
    # Sample the Almucantar angles.
    #
    BORDER_MAP_VALUE = 100000
    almucantar_samples = cv2.remap(
        img.astype(np.float),
        Almucantar_coords[0].astype(np.float32),
        Almucantar_coords[1].astype(np.float32),
        cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BORDER_MAP_VALUE
    )
    almucantar_samples = np.squeeze(almucantar_samples)
    
    # Sample the PrinciplePlane angles.
    #
    BORDER_MAP_VALUE = 100000
    principalplane_samples = cv2.remap(
        img.astype(np.float),
        PrincipalPlane_coords[0].astype(np.float32),
        PrincipalPlane_coords[1].astype(np.float32),
        cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BORDER_MAP_VALUE
    )
    principalplane_samples = np.squeeze(principalplane_samples)
    
    return almucantar_samples, almucantar_angles, Almucantar_coords, \
           principalplane_samples, principleplane_angles, PrincipalPlane_coords


def readSunPhotoMeter(df, timestamp, sun_angles=5):
    """Get a sunphotometer reading at some time."""
    
    #
    # Make sure that timestamp is of type pd.TimeStamp. For some reason, pandas
    # behaves differently when timestamp is a string, i.e. it returns a DataFrame
    # instead of a series, which causes the drop to fail (it requires an axis number).
    #
    timestamp = pd.Timestamp(timestamp)
    
    data = df.loc[timestamp].drop('Wavelength(um)').drop('SolarZenithAngle(degrees)')
    angles = np.array([float('.'.join(i.split('.')[:2])) for i in data.keys()])
    
    #
    # Remove wrong readings(?)
    #
    angles = angles[data.values != -100]
    values = data.values
    values = values[data.values != -100]
    
    #
    # Sort
    #
    angles, values = np.sort(angles), values[np.argsort(angles)]
    
    #
    # Remove angles near sun.
    #
    angles, unique_indices = np.unique(angles, return_index=True)
    values = values[unique_indices]
    
    #values = values[(angles<-sun_angles) | (angles>sun_angles)]
    #angles = angles[(angles<-sun_angles) | (angles>sun_angles)]
    values[(angles>-sun_angles) & (angles<sun_angles)] = 0
    
    return angles, values