"""Utilites for handling export to solver.
"""
from __future__ import division
from CameraNetwork.utils import sun_direction
import cPickle
import cv2
from enaml.application import deferred_call, is_main_thread
import logging
import math
import os
import numpy as np
import pymap3d


def exportToShdom(
    base_path,
    array_items,
    grid,
    lat,
    lon,
    alt,
    grabcut_threshold,
    progress_callback):
    """Process export of reconstruction data on separate thread.
    The export is intended for use in SHDOM.

    Args:
        base_path (str): Path to store export data.
        array_items (list): List of array items.
        grid (list): List of grid array. This is the grid to reconstruct.
        lat, lon, lat (float): The latitude, longitude and altitude of the center
            of the grid.
        grabcut_threshold (float): Threshold for grabcut algorithm applied for
            sunshader segmentation.
        progress_callback (function): Callback function to update the (GUI) with
            the progress of the export.

    Note:
        The directions in the Qt view are as follows:
        x axis (horizontal) goes from West (left) to East (right)
        y axis (vertical) goes from South (down) to North (up).
        this makes it a EN axis system
    """

    #
    # Reset the progress indicatort.
    #
    progress_cnt = len(array_items)
    deferred_call(progress_callback, 0)

    #
    # Convert the grid from NED to ECEF
    #
    GRID = np.meshgrid(*grid)
    ecef_grid = pymap3d.ned2ecef(GRID[0], GRID[1], GRID[2], lat, lon, alt)

    #
    # Loop on all servers and collect information.
    #
    export_data = {}
    for i, (server_id, (array_model, array_view)) in enumerate(array_items):
        if not array_view.export_flag.checked:
            logging.info(
                "Reconstruction: Camera {} ignored.".format(server_id)
            )
            continue

        #
        # Store extra data like camera center, etc.
        #
        extra_data = extraReconstructionData(array_model, array_view, lat0=lat, lon0=lon, h0=alt)

        img_array = array_view.img_array

        #
        # Calculate azimuth and elevation of each pixel.
        # TODO:
        # The azimuth and elevation here are calculated assuming
        # that the cameras are near. If they are far, the azimuth
        # and elevation should take into account the earth carvature.
        # I.e. relative to the center of axis the angles are rotated.
        #
        PHI_shdom, PSI_shdom = getShdomDirections(img_array, array_model)

        #
        # Calculate Masks.
        # Note:
        # sunshader mask is calculate using grabcut. This is used for removing the
        # sunshader.
        # Manual mask is the (ROI) mask marked by the user.
        #
        sunshader_mask = calcSunshaderMask(img_array, grabcut_threshold)
        manual_mask = array_view.image_widget.getMask()
        joint_mask = (manual_mask * sunshader_mask).astype(np.uint8)

        #
        # Project the grid on the image and check viewed voxels.
        # Note:
        # This measurement is used for checking how many cameras see each voxel.
        # TODO:
        # This procedure is time expensive and can be cached.
        # This should probably be a method of the camera, and this method should
        # cache the result, or even be triggered by setting the grid.
        #
        visibility = projectGridOnCamera(ecef_grid, array_model, joint_mask)

        export_data[server_id] = dict(
            extra_data=extra_data,
            R=array_view.image_widget.getArrayRegion(img_array[..., 0]),
            G=array_view.image_widget.getArrayRegion(img_array[..., 1]),
            B=array_view.image_widget.getArrayRegion(img_array[..., 2]),
            PHI=array_view.image_widget.getArrayRegion(PHI_shdom),
            PSI=array_view.image_widget.getArrayRegion(PSI_shdom),
            MASK=array_view.image_widget.getArrayRegion(joint_mask),
            Visibility=visibility,
        )

        deferred_call(progress_callback, i / progress_cnt)

    #
    # Save the results.
    #
    with open(os.path.join(base_path, 'export_data.pkl'), 'wb') as f:
        cPickle.dump(export_data, f)

    deferred_call(progress_callback, 0)


def getShdomDirections(img_array, array_model):
    """Calculate the (SHDOM) direction of each pixel.

    Directions are calculated in SHDOM convention where the direction is
    of the photons.
    """

    Y_shdom, X_shdom = np.meshgrid(
        np.linspace(-1, 1, img_array.shape[1]),
        np.linspace(-1, 1, img_array.shape[0])
    )
    PHI_shdom = np.pi + np.arctan2(Y_shdom, X_shdom)
    PSI_shdom = -np.pi + array_model.fov * np.sqrt(X_shdom**2 + Y_shdom**2)
    return PHI_shdom, PSI_shdom


def extraReconstructionData(array_model, array_view, lat0, lon0, h0):
    """Get extra data for the reconstruction

    This includes camera position, sun angle, time etc.

    Note:
        The coordinates are given in the following conventions:
        1) Camera position is given in NEU.
        2) sun_mu, sun_az are given in the SHDOM convention
           of photons directions.
    """

    #
    # Calculate the center of the camera.
    # Note that the coords are stored as NEU (in contrast to NED)
    #
    n, e, d = pymap3d.geodetic2ned(
        array_model.latitude, array_model.longitude, array_model.altitude,
        lat0=lat0, lon0=lon0, h0=h0)

    #
    # Calculate bounding coords (useful for debug visualization)
    #
    bounding_phi, bounding_psi = calcROIbounds(array_model, array_view)

    #
    # Sun azimuth and altitude
    #
    sun_alt, sun_az = sun_direction(
        latitude=str(array_model.latitude),
        longitude=str(array_model.longitude),
        altitude=array_model.altitude,
        at_time=array_model.img_data.name_time)

    #
    # Note:
    # shdom_mu = cos(pi/2-alt-pi)=cos(-alt-pi/2)=cos(alt+pi/2)
    #
    extra_data = \
        dict(
            at_time=array_model.img_data.name_time,
            sun_mu=math.cos(float(sun_alt)+np.pi/2),
            sun_az=float(sun_az)-np.pi,
            x=n,
            y=e,
            z=-d,
            bounding_phi=bounding_phi,
            bounding_psi=bounding_psi
        )
    return extra_data


def projectGridOnCamera(ecef_grid, array_model, joint_mask):
    """Project reconstruction grid on camera.

    This is used to estimate the visibility of each voxel by the camera.
    """

    xs, ys, fov_mask = array_model.projectECEF(ecef_grid, filter_fov=False)
    xs = xs.astype(np.uint32).flatten()
    ys = ys.astype(np.uint32).flatten()

    grid_visibility = np.zeros_like(xs, dtype=np.uint8)
    grid_visibility[fov_mask] = \
        joint_mask[ys[fov_mask], xs[fov_mask]].astype(np.uint8)

    return grid_visibility.reshape(*ecef_grid[0].shape)


def calcSunshaderMask(img_array, grabcut_threshold, values_range=40):
    """Calculate a mask for the sunshader.

    Calculate a mask for the pixels covered by the sunshader.
    Uses the grabcut algorithm.

    Args:
        img_array (array): Image (float HDR).
        grabcut_threshold (float): Threshold used to set the seed for the
            background.
        values_range (float): This value is used for normalizing the image.
            It is an empirical number that works for HDR images captured
            during the day.

    Note:
        The algorithm uses some "Magic" numbers that might need to be
        adapted to different lighting levels.
    """

    sunshader_mask = np.ones(img_array.shape[:2], np.uint8)*cv2.GC_PR_FGD
    sunshader_mask[img_array.max(axis=2) < grabcut_threshold] = cv2.GC_PR_BGD
    img_u8 = (255 * np.clip(img_array, 0, values_range) / values_range).astype(np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, 0, 0)
    cv2.grabCut(img_u8, sunshader_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    sunshader_mask = np.where(
        (sunshader_mask==cv2.GC_FGD) | (sunshader_mask==cv2.GC_PR_FGD),
        1,
        0).astype('uint8')

    return sunshader_mask


def calcROIbounds(array_model, array_view):
    """Calculate bounds of ROI in array_view

    Useful for debug visualization.
    """

    #
    # Get the ROI size
    #
    roi = array_view.ROI
    size = roi.state['size']

    #
    # Get the transform from the ROI to the data.
    #
    _, tr = roi.getArraySlice(array_view.img_array, array_view.image_widget.img_item)

    #
    # Calculate the bounds.
    #
    center = float(array_view.img_array.shape[0])/2
    pts = np.array(
        [tr.map(x, y) for x, y in \
         ((0, 0), (size.x(), 0), (0, size.y()), (size.x(), size.y()))]
    )
    pts = (pts - center) / center
    X, Y = pts[:, 1], pts[:, 0]
    bounding_phi = np.arctan2(X, Y)
    bounding_psi = array_model.fov * np.sqrt(X**2 + Y**2)

    return bounding_phi, bounding_psi


