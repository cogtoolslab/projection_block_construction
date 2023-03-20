# This file contains helper functions for the matter physics server

from asyncio import start_server
from re import A
import subprocess
import copyreg
from random import randint
import atexit

import os
import socket


js_location = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'matter_server.js')

# launch the server on import
process = subprocess.Popen(
            ['node', js_location], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# print("Started matter physics server 🧱.")

def check_process():
    """Checks if the process is still running."""
    global process
    if process.poll() is not None:
        process = subprocess.Popen(
            ['node', js_location], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print("Restarted matter physics server 🧱.")

class Physics_Server:
    def __init__(self, y_height=8) -> None:
        # if the height of the canvas differs, we need to subtract it to flip the y axis. 8 is default
        # self.start_server() # don't start the server here, rather lazily load it when physics is requested (since we often clone worlds but then not do anything with them)
        self.y_height = y_height
        self.keep_alive()

    def __del__(self):
        """Called when the object is deleted."""
        pass
        # self.kill_server()

    def start_server(self):
        """Starts the matter physics server. Does not need to be called manually—server will be started lazily."""
        check_process()
        self._process = process

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
        try:
            if self._process.poll() is not None:
                self.start_server()
        except:  # we don't have a process
            self.start_server()

    def get_stability(self, blocks):
        """Returns the stability of the given blocks.
        Blocks until the result is known.
        """
        # self.keep_alive() # check if the server process still lives
        blocks = self.blocks_to_serializable(blocks)
        # send the request to the process via stdin
        try:
            self._process.stdin.write(
                (str(blocks).replace('\'', '"') + '\n').encode('utf-8'))
            self._process.stdin.flush()
            # read the result from the process
            result = self._process.stdout.readline().decode('utf-8')
        except BrokenPipeError:
            # if the process is dead, restart it
            self.start_server()
            return self.get_stability(blocks)
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
    process.kill()
