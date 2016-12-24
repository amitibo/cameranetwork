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
