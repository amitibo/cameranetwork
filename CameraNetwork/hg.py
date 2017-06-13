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
"""
Utilities for handling mercurial repository.
Based on code by Troy Williams.
"""

import os
from tornado import gen
from utils import sbp_run


class MercurialException(Exception):
    pass


class Repository():

    def __init__(self, repo_path):
        self.path = repo_path
        self.name = os.path.basename(repo_path)
        self.commands = {'update':['hg', 'update']}
        self.commands['log'] = ['hg', 'log']
        self.commands['push'] = ['hg', 'push']
        self.commands['pull'] = ['hg', 'pull']
        self.commands['incoming'] = ['hg', 'incoming']
        self.commands['outgoing'] = ['hg', 'outgoing']

    def run_repo_command(self, command, *params):
        """
        Execute a command against the repository
        """

        command = command + [param for param in params]

        stdout, stderr = sbp_run(
            command,
            shell=False,
            working_directory=self.path
        )

        if stderr:
            err_msg = '{command} returned the following error:\n{err}\n'.format(
                command=command,
                err=stderr
            )
            raise MercurialException(err_msg)

        return stdout

    def log(self, *params):
        """
        Execute the hg log command on the repository
        """

        return self.run_repo_command(self.commands['log'], *params)

    @gen.coroutine
    def update(self, *params):
        """
        Execute the hg update command on the repository
        """

        return self.run_repo_command(self.commands['update'], *params)

    def push(self, *params):
        """
        Execute the hg push command on the repository
        """

        return self.run_repo_command(self.commands['push'], *params)

    @gen.coroutine
    def pull(self, *params):
        """
        Executes the hg pull command on the repository
        """

        return self.run_repo_command(self.commands['pull'], *params)

    def incoming(self, *params):
        """
        Executes the hg incoming command on the repository
        """

        return self.run_repo_command(self.commands['incoming'])

    def outgoing(self, *params):
        """
        Executes the hg outgoing command on the repository
        """

        return self.run_repo_command(self.commands['outgoing'])
