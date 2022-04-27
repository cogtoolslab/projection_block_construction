# This file contains helper functions for the matter physics server

from asyncio import start_server
import subprocess
import copyreg
from random import randint
import atexit

import os
import socket


js_location = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'matter_server.js')

pid_reference_manager = {}  # stores open process IDs


class Physics_Server:
    def __init__(self, y_height=8) -> None:
        # if the height of the canvas differs, we need to subtract it to flip the y axis. 8 is default
        self.start_server()
        self.y_height = y_height

    def __del__(self):
        """Called when the object is deleted."""
        self.kill_server()

    # def __deepcopy__(self, memo):
    #     """Deep copy of the physics server—we keep the same socket and the same process"""
    #     return Physics_Server(socket=self.socket, y_height=self.y_height, _process=self._process)

    def start_server(self):
        """Starts the matter physics server."""
        # start the node process with stdin,stdout and stderr
        self._process = subprocess.Popen(
            ['node', js_location], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # add the process to the reference manager
        pid_reference_manager[self._process.pid] = 1

    def kill_server(self, force=False):
        """Kills the matter physics server."""
        if force:
            try:
                del(pid_reference_manager[self._process.pid])
                self._process.kill()
            except:
                pass
            finally:
                self._process = None
                return
        if self._process is not None:
            try:
                if pid_reference_manager[self._process.pid] == 1:
                    # we're the last user of the node server, let's kill it
                    del pid_reference_manager[self._process.pid]
                    self._process.kill()
                    self._process = None
                else:
                    # there are more references around, but we're letting this one go
                    pid_reference_manager[self._process.pid] -= 1
            except KeyError:
                # the process is already dead
                self._process = None

    def blocks_to_serializable(self, blocks):
        return [self.block_to_serializable(block) for block in blocks]

    def block_to_serializable(self, block):
        """Returns a serializable version of the block."""
        return {
            'x': float(block.x),
            'y': float(self.y_height - 1 - block.y),
            'w': float(block.width),
            'h': float(block.height),
        }

    def keep_alive(self):
        """After pickling etc. the process and the process can be lost. This function checks if we need to restart the node.js process and regenerate the process."""
        if self._process is None:
            self.start_server()
        if self._process.poll() is not None:
            self.start_server()

    def get_stability(self, blocks):
        """Returns the stability of the given blocks.
        Blocks until the result is known.
        """
        self.keep_alive()
        blocks = self.blocks_to_serializable(blocks)
        # send the request to the process via stdin
        self._process.stdin.write(
            (str(blocks).replace('\'', '"') + '\n').encode('utf-8'))
        self._process.stdin.flush()
        # read the result from the process
        result = self._process.stdout.readline().decode('utf-8')
        # return the result
        return result == 'true\n'


def pickle_physics_server(server):
    """Pickle function for physics server. A new process is started when unpickled."""
    return (Physics_Server, (server.y_height,))


# register custom pickle function for the server
copyreg.pickle(Physics_Server, pickle_physics_server)


@atexit.register
def killallprocesses():
    """Kills all the processes that are still running once we close the file (ie. are done with everything)."""
    for pid in pid_reference_manager:
        os.kill(pid, 9)
