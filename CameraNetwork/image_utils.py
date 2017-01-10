"""General utilities for image processing.
"""
from __future__ import division, absolute_import, print_function
from CameraNetwork.utils import obj
import cv2
import logging
import numpy as np


def calcHDR(img_arrays, img_exposures, low_limit=20, high_limit=230):
    """Calculate an HDR image.

    Args:
        img_arrays (list of images): List of uint8 images.
        img_exposures (list of floats): Corresponding exposures (preferably in
            ms). The images are expected to be sorted from shortest to longest
            exposure.
        low_limit (float[optional]): low RGB value below which values are
            ignored except in the longest exposure.
        high_limit (float[optional]): high RGB value above which values are
            ignored except in the shortest exposure.

    Retruns:
        HDR image merged from the image arrays. There idea is that the shortest
        exposure is used for the high RGB values, and the longest for the low
        values. The other values are averaged from all images. The HDR image
        returned is float in units of [RGB/ms]
    """

    hdr_imgs = []
    for i, (img, exp) in enumerate(zip(img_arrays, img_exposures)):
        hdr_img = img.copy().astype(np.float) / exp
        if i == 0:
            mask_min, mask_max = low_limit, 255
        elif i == len(img_arrays)-1:
            mask_min, mask_max = 0, high_limit
        else:
            mask_min, mask_max = low_limit, high_limit

        mask = (img < mask_min) | (img > mask_max)
        hdr_img[mask] = np.nan
        hdr_imgs.append(hdr_img)

    return np.nanmean(hdr_imgs, axis=0)


def raw2RGB(img, dtype=None):
    """Convert a Raw image to its three RGB channels."""

    if dtype is None:
        dtype = img.dtype

    R = np.ascontiguousarray(img[::2, ::2])
    B = np.ascontiguousarray(img[1::2, 1::2])
    G1 = np.ascontiguousarray(img[1::2, 0::2]).astype(np.float)
    G2 = np.ascontiguousarray(img[0::2, 1::2]).astype(np.float)

    return R, ((G1+G2)/2).astype(R.dtype), B


def RGB2raw(R, G, B):
    """Convert RGB channels to Raw image."""

    h, w = R.shape
    raw = np.empty(shape=(2*h, 2*w), dtype=R.dtype)

    raw[::2, ::2] = R
    raw[1::2, 1::2] = B
    raw[1::2, 0::2] = G
    raw[0::2, 1::2] = G

    return raw


class FisheyeProxy(object):
    """Fisheye proxy class

    A wrapper that enables using the OcamCalib model and OpenCV fisheye model
    interchangeably.
    """
    def __init__(self, model):

        self._model = model

        #
        # Check type of fisheye model
        #
        self._ocamcalib_flag = type(model) == obj

    def __getattr__(self, name):
        """Delegate the attribute enquiry to the model"""

        return getattr(self._model, name)

    def projectPoints(self, XYZ):
        """Project 3D points on the 2D image"""

        if self._ocamcalib_flag:
            YXmap = world2cam(XYZ, self._model)
        else:
            YXmap = self._model.projectPoints(XYZ)[..., ::-1]

        return YXmap


class Normalization(object):
    """Normalized Image Class

    This class encapsulates the conversion between caputered image and
    the normalized image.
    """

    def __init__(
        self,
        resolution,
        fisheye_model,
        Rot=np.eye(3),
        fov=np.pi/2
        ):

        self._fisheye_model = fisheye_model
        self._Rot = Rot
        self.fov = fov

        self.calc_normalization_map(resolution)

    def calc_normalization_map(self, resolution):
        """Calculate normalization map

        Calc mapping from original image to normalized image.
        """

        logging.debug('Calculating normalization map.')
        self.resolution = resolution

        #
        # Create a grid of directions.
        # The coordinates create a 'linear' fisheye, where the distance
        # from the center ranges between 0-pi/2 linearily.
        #
        X, Y = np.meshgrid(
            np.linspace(-1, 1, self.resolution),
            np.linspace(-1, 1, self.resolution)
        )

        self._PHI = np.arctan2(Y, X)
        self._PSI = self.fov * np.sqrt(X**2 + Y**2)
        self.mask = self._PSI <= self.fov
        self._PSI[~self.mask] = self.fov

        z = np.cos(self._PSI)
        x = np.sin(self._PSI) * np.cos(self._PHI)
        y = np.sin(self._PSI) * np.sin(self._PHI)

        XYZ = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))

        #
        # Rot is the rotation applied to the camera coordinates to turn into world coordinates.
        # So I multiply from the right (like multiplying with the inverse of the matrix).
        #
        logging.info(self._Rot)
        XYZ_rotated = np.dot(XYZ, self._Rot)
        YXmap = self._fisheye_model.projectPoints(XYZ_rotated)

        self._Xmap = np.array(YXmap[:, 1], dtype=np.float32).reshape(X.shape)
        self._Ymap = np.array(YXmap[:, 0], dtype=np.float32).reshape(Y.shape)
        self._Xmap[~self.mask] = -1
        self._Ymap[~self.mask] = -1

    @property
    def R(self):
        return self._Rot

    @R.setter
    def R(self, R):
        self._Rot = R
        self.calc_normalization_map(self.resolution)

    def normalize(self, img):
        """Normalize Image

        Apply normalization to image.
        Note:
        The fisheye model is adapted to the size of the image, therefore
        different sized images (1200x1600 or 600x800) can be normalized.
        """

        import cv2

        logging.debug('Applying normalization to image.')
        self._tight_mask = self.mask.copy()

        if img is None:
            return np.zeros((self.resolution, self.resolution, 3))

        #
        # 2D images are assumed to be RAW, and converted to RGB
        #
        if img.ndim == 2:
            logging.debug('Converting grey image to RGB')
            img = np.dstack(raw2RGB(img))

        logging.debug('Normalizing shape: {}'.format(img.shape))

        #
        # Check if there is need to scale the images
        #
        calib_shape = self._fisheye_model.img_shape
        if img.shape[:2] != calib_shape:
            x_scale = float(img.shape[1]) / float(calib_shape[1])
            y_scale = float(img.shape[0]) / float(calib_shape[0])
            Xmap = self._Xmap * x_scale
            Ymap = self._Ymap * y_scale
        else:
            Xmap = self._Xmap
            Ymap = self._Ymap

        img_dtype = img.dtype
        BORDER_MAP_VALUE = 100000
        normalized_img = cv2.remap(
            img.astype(np.float),
            Xmap,
            Ymap,
            cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=BORDER_MAP_VALUE
        )

        if normalized_img.ndim == 2:
            self._tight_mask[normalized_img>(BORDER_MAP_VALUE-1)] = 0
        else:
            self._tight_mask[normalized_img[:,:, 0]>(BORDER_MAP_VALUE-1)] = 0

        #
        # Erod one pixel from the tight mask
        #
        self._tight_mask = cv2.erode(
            self._tight_mask.astype(np.uint8),
            kernel=np.ones((3, 3), dtype=np.uint8)
            ).astype(np.bool)

        #
        # TODO:
        # Implement radiometric correction
        #
        #normalized_img = radiometric_correction(normalized_img, self._radiometric_model).astype(np.uint8)
        normalized_img = normalized_img.astype(img_dtype)
        normalized_img[self._tight_mask==0] = 0

        return normalized_img

    def normalCoords2Direction(self, coords):
        """Convert Normal Coord to Direction Vector

        Convert a coordinate (even sub pixel) of normalized image to a
        direction vector in space.

        Args:
            coords (tuple): normalized coordinates (x, y), supports subpixels.
        """

        coords = np.array(coords).reshape(-1, 2)
        XY = coords/self.resolution*2 - 1
        #
        # Create a grid of directions.
        #
        X, Y = XY[..., 0], XY[..., 1]

        PHI = np.arctan2(Y, X)
        PSI = self.fov * np.sqrt(X**2 + Y**2)
        mask = PSI <= self.fov
        PSI[~mask] = self.fov

        z = np.cos(PSI)
        x = np.sin(PSI) * np.cos(PHI)
        y = np.sin(PSI) * np.sin(PHI)

        XYZ = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))

        #
        # Rot is the rotation applied to the camera coordinates to turn into world coordinates.
        # So I multiply from the right (like multiplying with the inverse of the matrix).
        #
        XYZ_rotated = np.dot(XYZ, self._Rot)

        return XYZ_rotated

    def distorted2Directions(self, x, y):
        """Convert distorted pixels to directions."""

        X = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        phi, theta = self._fisheye_model.undistortDirections(X)

        return phi, theta

    @property
    def normal_coords(self):
        """Normalized grid"""

        return self._Xmap, self._Ymap

    @property
    def normal_angles(self):
        """Normalized grid"""

        return self._PHI, self._PSI

    @property
    def tight_mask(self):
        """
        Return a tight mask of the normalized image. The tight mask takes into account the cropping by the sensor edges.
        """

        return self._tight_mask


