# This file contains helper functions for the matter physics server

import subprocess
import zmq
from random import randint

import os

# socket setup
context = zmq.Context()

JS_LOCATION = 'utils/matter_server.js'

if "stimuli" in os.getcwd():
    # if our working directory in stimuli, we need to make sure to find the location of the node file in the right place
    JS_LOCATION = "../"+JS_LOCATION

class Physics_Server:
    def __init__(self, port=None, y_height=8, socket=None, _process = None) -> None:
        if socket is None:
            self.socket = self.start_server(port)
        else:
            self.socket = socket
        # if the height of the canvas differs, we need to subtract it to flip the y axis. 8 is default
        self.y_height = y_height
        # the process should only be set by the deepcopy function
        self._process = _process

    def __del__(self):
        """Called when the object is deleted."""
        self.kill_server()

    
    def __deepcopy__(self, memo):
        """Deep copy of the physics server—we keep the same socket and the same process"""
        return Physics_Server(socket=self.socket, y_height=self.y_height, _process=self._process)

    def start_server(self, port=None):
        """Starts the matter physics server and returns a socketio connection to it."""
        if port is None:
            port = randint(0, 999999999)
        # TODO if necessary add a fallback to tcp:// for Windows users
        self._process = subprocess.Popen(
            ['node', JS_LOCATION, '--port', str(port)])
        socket = context.socket(zmq.REQ)
        while True:
            try:
                # connect to ipc
                socket.connect('ipc:///tmp/matter_server_{}'.format(port))
                break
            except:
                pass
        return socket

    def kill_server(self):
        """Kills the matter physics server."""
        if self._process is not None:
            self._process.kill()

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

    def get_stability(self, blocks):
        """Returns the stability of the given blocks.
        Blocks until the result is known.
        """
        blocks = self.blocks_to_serializable(blocks)
        self.socket.send_json(blocks)
        # receive result—blocking
        result = self.socket.recv()
        # is either 'false' or 'true'
        result = result == b'true'
        return result
