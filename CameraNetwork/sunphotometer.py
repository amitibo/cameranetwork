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
from __future__ import division, print_function, absolute_import
import cv2
import datetime
from dateutil import parser
import ephem
import numpy as np
import os
import pandas as pd

WAVELENGTHS = ['0.4405', '0.5000', '0.6744']
SUNPHOTOMETER_WAVELENGTHS = (0.4405, 0.5000, 0.6744)


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
    #
    start_az = sun.az
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
    alm_x = (alm_radius * np.sin(alm_az) + 1) * img_resolution / 2
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
    pp_az = sun.az * np.ones(len(pp_alts))

    #
    # Convert Principal Plane angles to image coords.
    #
    pp_radius = (np.pi/2 - pp_alts)/(np.pi/2)
    pp_x = (pp_radius * np.sin(pp_az) + 1) * img_resolution / 2
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


#def sampleData(spm_df, t, camera_df, cam_id='102', resolution=301):
    #"""Sample almucantar rgb values of some camera at specific time."""

    #angles, values = readSunPhotoMeter(spm_df, t)
    #closest_time = findClosestImageTime(camera_df, t, hdr='2')
    #img, img_data = cams.seek(cam_id, closest_time, -1, resolution)
    #almucantar_samples, almucantar_angles, almucantar_coords, \
           #_, _, _ = sampleImage(img, img_data, almucantar_angles=angles)
    ##
    ## Visualize the sampling positions
    ##
    #for x, y in zip(almucantar_coords[0], almucantar_coords[1]):
        #cv2.circle(img, (int(x), int(y)), 2, (255, 255, 0))

    #return angles, values, almucantar_samples, img, closest_time


def sampleData(
    camera_client,
    spm_dfs,
    QEs,
    ch_index,
    time_index,
    camera_df,
    camera_id,
    resolution=301,
    overlay_angles=True):
    """Samples almucantar values of some camera at specific time and color channel.

    Args:
        camera_client (camera client object): Client to access the camera servers.
        spm_dfs (list of DataFrames): Sunphotometer readings (one for each visible
            in the order BGR).
        QEs (list of DataFrames): Quantum Efficiency graphs of the camera in RGB order.
        ch_index (int): Color channel to process (in order [R, G, B])
        time_index (int): Time index for the spm dataframes.
        camera_df (DataFrame): DataFrames of images captured for the specific day.
        camera_id (str): The camera to read from.
        resoluiton (int): The resolution in which to sample the image.
        overlay_angles (boolean): Overlay almucantar angles on the image.

    Returns:
        angles, values, almucantar_samples, img, closest_time: Almacuntar angles,
            sunphotometer values, image values measured at the spm angles, etc.

    Note:
        This function is supposed to be used from a notebook (it uses the camera
        clinet object).
    """

    #
    # Read the SunPhotometer values at specific time.
    #
    angles_blue, values_blue = readSunPhotoMeter(spm_dfs[0], spm_dfs[0].index[time_index])
    angles_green, values_green = readSunPhotoMeter(spm_dfs[1], spm_dfs[1].index[time_index])
    angles_red, values_red = readSunPhotoMeter(spm_dfs[2], spm_dfs[2].index[time_index])

    #
    # Join all datasets. This is important as not all datasets are sampled
    # at all angles. Therefore I use dropna() at the end.
    # Note:
    # The sun-photometer Dataframe is created in the order BGR to allow for the integration
    # along the visual spectrum.
    #
    blue_df = pd.DataFrame(data={SUNPHOTOMETER_WAVELENGTHS[0]: values_blue}, index=angles_blue)
    green_df = pd.DataFrame(data={SUNPHOTOMETER_WAVELENGTHS[1]: values_green}, index=angles_green)
    red_df = pd.DataFrame(data={SUNPHOTOMETER_WAVELENGTHS[2]: values_red}, index=angles_red)
    SPM_df = pd.concat((blue_df, green_df, red_df), axis=1).dropna()

    angles, values = integrate_QE_SP(SPM_df, QEs[ch_index])

    #
    # Get the closest image time.
    #
    t = spm_dfs[ch_index].index[time_index]
    closest_time = findClosestImageTime(camera_df, t, hdr='2')
    img, img_data = camera_client.seek(
        server_id=camera_id,
        seek_time=closest_time,
        hdr_index=-1,
        jpeg=False,
        resolution=resolution,
        correct_radiometric=False
    )
    img = img[0]
    img_data = img_data[0]

    almucantar_samples, almucantar_angles, almucantar_coords, \
           _, _, _ = sampleImage(img, img_data, almucantar_angles=angles)

    #
    # Visualize the sampling positions on the image.
    #
    if overlay_angles:
        import cv2
        for x, y in zip(almucantar_coords[0], almucantar_coords[1]):
            cv2.circle(img, (int(x), int(y)), 2, (255, 255, 0))

    return angles, values, almucantar_samples, img, closest_time, img_data


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


def calcSunphometerCoords(img_data, resolution):
    """Calculate the Almucantar and PrinciplePlanes for a specifica datetime."""

    Almucantar_coords, PrincipalPlane_coords, _, _ = \
        calcAlmucantarPrinciplePlanes(
            latitude=img_data.latitude,
            longitude=img_data.longitude,
            capture_time=img_data.capture_time,
            img_resolution=resolution)

    return Almucantar_coords.T.tolist(), PrincipalPlane_coords.T.tolist()


def calcSunCoords(img_data, resolution):
    """Calculate the Sun coords for a specifica datetime."""

    # Create an Sun/observer at camera position
    #
    observer = ephem.Observer()
    observer.lat, observer.long, observer.date = \
        str(img_data.latitude), str(img_data.longitude), img_data.capture_time

    sun = ephem.Sun(observer)

    #
    # Calculate sun angles.
    #
    sun_az = np.array([sun.az])
    sun_alts = np.array([sun.alt])

    #
    # Convert sun angles to image coords.
    #
    sun_radius = (np.pi/2 - sun_alts)/(np.pi/2)
    sun_x = (sun_radius * np.sin(sun_az) + 1) * resolution / 2
    sun_y = (sun_radius * np.cos(sun_az) + 1) * resolution / 2
    Sun_coords = np.array((sun_x, sun_y))

    return Sun_coords.T.tolist()


def integrate_QE_SP(SPM_df, QE):
    """Caclulate the argument:
        \int_{\lambda} \mathrm{QE}_{\lambda} \, \lambda \, L^{\mathrm{S-P}}_{\lambda} \, d{\lambda}

    This integral is calculated for each almacuntar angle (for specfic day time).

    Args:
        SPM_df (pandas dataframe): Dataframe of Sun Photometer readings, arranged
            in BGR order.
        QE (pandas Dataframe): Dataframe of Quantum Efficieny of a specific channel.

    Returns:
        Integration of the Sun Photometer radiances (per SP almacuntar angle)
        scaled by the Quantum Efficiency of the specific channel.
    """

    from scipy.interpolate import InterpolatedUnivariateSpline

    #
    # Limits and density of the integraion.
    #
    start, end = 0.4, 0.7
    dlambda = 0.005
    xspl = np.linspace(start, end, int((end - start) / dlambda))

    interp = []
    for angle, row, in SPM_df.iterrows():
        #
        # Interpolate the sun photometer values along the wavelengths axis.
        #
        sp_vals = row.values
        isp = InterpolatedUnivariateSpline(SUNPHOTOMETER_WAVELENGTHS, sp_vals, k=2)
        sp_ipol = isp(xspl)

        #
        # Interpolate the Quantum Efficiencies along the wavelenghts axis
        # Note:
        # The QE wavelengths are given in nm, and values are given in 100 percent.
        # So I scale these by 1/1000 and 1/100 respectively.
        #
        QEp = InterpolatedUnivariateSpline(QE["wavelength"].values/1000, QE["QE"]/100)
        QE_ipol = QEp(xspl)

        #
        # Integrate the value:
        # \int_{\lambda} \mathrm{QE}_{\lambda} \, \lambda \, L^{\mathrm{S-P}}_{\lambda} \, d{\lambda}
        #
        interp.append(np.trapz(QE_ipol * xspl * sp_ipol, xspl))

    return SPM_df.index.values, interp
