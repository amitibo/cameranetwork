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
#------------------------------------------------------------------------------
#  Copyright (c) 2013, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------
from atom.api import Bool, Typed, observe, set_default
from enaml.core.declarative import d_

from traits.api import Instance, HasStrictTraits
from traitsui.api import View, Item
from tvtk.pyface.scene_editor import SceneEditor
from tvtk.pyface.scene_model import SceneModel

from .traits_view import TraitsView

from mayavi.core.ui.mayavi_scene import MayaviScene


class MayaviModel(HasStrictTraits):
    scene = Instance(SceneModel, args=())
    view = View(
        #Item('scene', editor=SceneEditor(), resizable=True, show_label=False),
        Item('scene', editor=SceneEditor(scene_class=MayaviScene), resizable=True, show_label=False),
        resizable=True)


class MayaviCanvas(TraitsView):
    """ A traits view widget that displays a mayavi scene.

    :Attributes:
        **scene** = *d_(Typed(SceneModel))*
            The mayavi scene model to be displayed.
        **show_toolbar** = *d_(Bool(True))*
            If True, show the Mayavi toolbar.

    """
    #: The mayavi scene model to be displayed.
    scene = d_(Typed(SceneModel))

    #: If True, show the Mayavi toolbar.
    show_toolbar = d_(Bool(True))

    #: The readonly instance of the model used for the traits_view.
    model = d_(Typed(MayaviModel), writable=False)

    #: The readonly instance of the view.
    view = d_(Typed(View), writable=False)

    #: Mayavi canvas expands freely in width and height by default.
    hug_width = set_default('ignore')
    hug_height = set_default('ignore')

    def create_widget(self, parent):
        control = super(MayaviCanvas, self).create_widget(parent)
        self.show_toolbar_changed({'value': self.show_toolbar})
        return control

    def _default_model(self):
        return MayaviModel(scene=self.scene)

    def _default_view(self):
        return self.model.trait_view()

    @observe('scene')
    def scene_changed(self, change):
        self.model.scene = change['value']

    @observe('show_toolbar')
    def show_toolbar_changed(self, change):
        ui = self.ui
        if ui is not None and ui.control is not None:
            editor = ui.get_editors('scene')[0]
            editor._scene._tool_bar.setVisible(change['value'])
