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
#
# (C) Copyright 2013 Enthought, Inc., Austin, TX
# All right reserved.
#
# This file is open source software distributed according to the terms in
# LICENSE.txt
#
from traits.api import HasTraits
from traitsui.api import View

from atom.api import Typed, set_default

from enaml.core.declarative import d_
from enaml.widgets.raw_widget import RawWidget


class TraitsView(RawWidget):
    """ A widget which wraps a TraitsUI View on an object.

    :Attributes:
        **model** = `d_(Typed(HasTraits))`
            The HasTraits instance that we are using.
        **view** = `d_(Typed(View))`
            The View instance that we are using.
        **ui** = `Typed(HasTraits)`
            A reference to the TraitsUI UI object.
        **hug_width** = `set_default('weak')`
            TraitsViews hug their contents' width weakly by default.

    """

    #: The HasTraits instance that we are using.
    model = d_(Typed(HasTraits))

    #: The View instance that we are using.
    view = d_(Typed(View))

    #: A reference to the TraitsUI UI object.
    ui = Typed(HasTraits)

    #: TraitsViews hug their contents' width weakly by default.
    hug_width = set_default('weak')

    def create_widget(self, parent):
        self.ui = self.model.edit_traits(
            self.view, parent=parent, kind='subpanel')
        self.ui.control.setParent(parent)
        return self.ui.control

    def destroy_widget(self):
        control = self.ui.control
        if control is not None:
            control.setParent(None)
            self.ui.dispose()

    def destroy(self):
        """ A reimplemented destructor.

        This destructor disposes the TraitsUI object before proceeding with
        the regular Enaml destruction.

        """
        self.destroy_widget()
        super(TraitsView, self).destroy()
