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
from traits.api import HasTraits, Range, Instance, \
     on_trait_change
from traitsui.api import View, Item, HGroup
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import \
     MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
import numpy as np
import cPickle
from skycameras import FisheyeProxy, Normalization
import pyshtools as shtools
import fisheye
import os


COLORS = ('blue', 'green', 'red')
COLOR_INDICES = {'blue': 2, 'green': 1, 'red': 0}


def estimate(z, theta, phi, THETAnorm, PHInorm, lmax):
    
    cilm, chi2 = shtools.SHExpandLSQ(z, np.degrees(theta), np.degrees(phi), lmax)
    Zimg = shtools.MakeGridArray(cilm, np.degrees(THETAnorm).ravel(), np.degrees(PHInorm).ravel(), lmax)
    Zimg.shape = PHInorm.shape
    
    return Zimg
    

class Visualization(HasTraits):
    lmax = Range(0, 30,  6)
    scene = Instance(MlabSceneModel, ())

    # the layout of the dialog created
    view = View(
        Item('scene', editor=SceneEditor(scene_class=MayaviScene),
             height=250, width=300, show_label=False),
        HGroup(
            '_', 'lmax',
            ),
    )

    def __init__(self, x, y, z, theta, phi, THETAnorm, PHInorm, mask):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        
        self.theta = theta
        self.phi = phi
        self.THETAnorm = THETAnorm
        self.PHInorm = PHInorm
        self.z = z
        self.mask = mask
        
        #
        # Visualize the results.
        #
        self.Ximg, self.Yimg = np.mgrid[0:1001, 0:1001]
        
        Zimg = estimate(self.z, self.theta, self.phi, self.THETAnorm, self.PHInorm, self.lmax)
        self.mesh = self.scene.mlab.mesh(self.Ximg, self.Yimg, Zimg, mask=~mask)
        
        x_undistort = (theta/(np.pi/2)*np.sin(phi)+1) * 1001/2
        y_undistort = (theta/(np.pi/2)*np.cos(phi)+1) * 1001/2
        self.scene.mlab.points3d(x_undistort, y_undistort, self.z, mode='sphere', scale_mode='none', scale_factor=5, color=(0, 0, 1))
    
        self.scene.mlab.outline(color=(0, 0, 0), extent=(0, 1001, 0, 1001, 0, 255))
        
    @on_trait_change('lmax')
    def update_plot(self):
        Zimg = estimate(self.z, self.theta, self.phi, self.THETAnorm, self.PHInorm, self.lmax)
        Zimg = np.clip(Zimg, 0, 255)
        self.mesh.mlab_source.set(x=self.Ximg, y=self.Yimg, z=Zimg, mask=~self.mask)


def main(base_path):
    #
    # Load the measurements
    #
    color = 'blue'
    color_index = COLOR_INDICES[color]
    
    with open(os.path.join(base_path, 'measurements_{}.pkl'.format(color)), 'rb') as f:
        measurements = cPickle.load(f)

    img_rgb = np.zeros(shape=(600, 800, 3))
    for x, y, val in measurements:
        img_rgb[int(y/2), int(x/2), ...] = val
    
    img = img_rgb[..., 2]
    y, x = np.nonzero(img)
    z = img[np.nonzero(img)]    
    
    fe = fisheye.load_model(os.path.join(base_path, 'fisheye.pkl'))

    #
    # Calculate projection of the distorted points on phi-theta coords.
    #
    X = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    phi, theta, mask = fe.undistortDirections(X)
    phi = phi[mask]
    z = z[mask]
    theta = theta[mask]

    #
    # Xnorm, Ynorm hold the distorted (image coords) coordinates of
    # the 'linear' fisheye (where the distance from the center
    # ranges between 0-pi/2 linearily)
    #
    normalization = Normalization(1001, FisheyeProxy(fe))    
    PHInorm, THETAnorm = normalization.normal_angles

    visualization = Visualization(x, y, z, theta, phi, THETAnorm, PHInorm, mask=normalization.mask)
    visualization.configure_traits()    


if __name__ == '__main__':
    base_path1 = r'radiometric_calibration\4102820374'
    main(base_path1)
    
