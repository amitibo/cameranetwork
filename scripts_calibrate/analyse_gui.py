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
from pyface.qt import QtGui, QtCore

import matplotlib
# We want matplotlib to use a QT backend
matplotlib.use('Qt4Agg')
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from traits.api import Any, HasTraits, Range, Instance, Enum, \
     on_trait_change
from traitsui.api import View, Item, HGroup, VGroup, Spring
from traitsui.qt4.editor import Editor
from traitsui.qt4.basic_editor_factory import BasicEditorFactory

from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import \
     MlabSceneModel

from mayavi import mlab
from mayavi.core.ui.mayavi_scene import MayaviScene

import numpy as np
import os
import glob
import cPickle

COLORS = ('blue', 'green', 'red')
COLOR_INDICES = {'blue': 2, 'green': 1, 'red': 0}
base_path1 = r'radiometric_calibration\4102820378'
base_path2 = r'radiometric_calibration\4102820386'


class _MPLFigureEditor(Editor):

    scrollable  = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # matplotlib commands to create a canvas
        mpl_canvas = FigureCanvas(self.value)
        return mpl_canvas

class MPLFigureEditor(BasicEditorFactory):

    klass = _MPLFigureEditor


class Visualization(HasTraits):
    meridional = Range(1, 30,  6)
    transverse = Range(0, 30, 11)
    scene_red = Instance(MlabSceneModel, ())
    scene_green = Instance(MlabSceneModel, ())
    scene_blue = Instance(MlabSceneModel, ())

    figure = Instance(Figure, ())

    colors_list = Enum(*COLORS)

    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        
        self.update_3dplot()
        self.update_plot()
        
    @on_trait_change('colors_list')
    def update_3dplot(self):
        
        color = self.colors_list
          
        with open(os.path.join(base_path1, 'measurements_{}.pkl'.format(color)), 'rb') as f:
            measurements1 = cPickle.load(f)
        with open(os.path.join(base_path2, 'measurements_{}.pkl'.format(color)), 'rb') as f:
            measurements2 = cPickle.load(f)

        x1, y1, z1 = [np.array(a) for a in zip(*measurements1)]
        x2, y2, z2 = [np.array(a) for a in zip(*measurements2)]            

        for scene, c in zip([self.scene_blue, self.scene_green, self.scene_red], COLORS):
            mlab.clf(figure=scene.mayavi_scene)
            
            mlab.points3d(x1, y1, z1[..., COLOR_INDICES[c]], mode='sphere', scale_mode='none', scale_factor=5, color=(0, 0, 1), figure=scene.mayavi_scene)
            mlab.points3d(x2, y2, z2[..., COLOR_INDICES[c]], mode='cube', scale_mode='none', scale_factor=5, color=(0, 1, 1), figure=scene.mayavi_scene)
            
            mlab.outline(color=(0, 0, 0), extent=(0, 1600, 0, 1200, 0, 255), figure=scene.mayavi_scene)
        

    @on_trait_change('colors_list')
    def update_plot(self):
        
        color = self.colors_list
          
        with open(os.path.join(base_path1, 'spec_{}.pkl'.format(color)), 'rb') as f:
            spec1 = cPickle.load(f)
        with open(os.path.join(base_path2, 'spec_{}.pkl'.format(color)), 'rb') as f:
            spec2 = cPickle.load(f)

        self.figure.clear()
        axes = self.figure.add_subplot(111)

        axes.plot(spec1[0], spec1[1], label='camera1')
        axes.plot(spec2[0], spec2[1], label='camera1')
        axes.legend()
        
        canvas = self.figure.canvas
        if canvas is not None:
            canvas.draw()

    # the layout of the dialog created
    view = View(
        VGroup(
            HGroup(
                HGroup(
                    Item('figure', editor=MPLFigureEditor(),
                         show_label=False),                    
                    label = 'Spectograms',
                    show_border = True
                ),
                HGroup(
                    Item('scene_red', editor=SceneEditor(scene_class=MayaviScene),
                         height=250, width=300, show_label=False),
                    label = 'Red',
                    show_border = True
                ),
                HGroup(
                    Item('scene_green', editor=SceneEditor(scene_class=MayaviScene),
                         height=250, width=300, show_label=False),
                    label = 'Green',
                    show_border = True
                ),
                HGroup(
                    Item('scene_blue', editor=SceneEditor(scene_class=MayaviScene),
                         height=250, width=300, show_label=False),                                
                    label = 'Blue',
                    show_border = True
                    ),
                ),
            '_',
            HGroup(
                Item('colors_list', style = 'simple'),
                Spring()
                ),
        )
    )

visualization = Visualization()
visualization.configure_traits()