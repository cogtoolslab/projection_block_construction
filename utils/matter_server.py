# This file contains helper functions for the matter physics server

import subprocess
import socketserver
import socketio
import time
from random import randint


class Physics_Server:
    def __init__(self, port=None, y_height=8, socket=None) -> None:
        if socket is None:
            self.socket = self.start_server(port)
        else:
            self.socket = socket
        self._results = {}  # stores the results of stability requests
        # if the height of the canvas differs, we need to subtract it to flip the y axis. 8 is default
        self.y_height = y_height

        # callback function for stability
        # can't believe that worked
        @self.socket.on('stability')
        def on_stability(data):
            self._results[data['id']] = data['stability']

    def __del__(self):
        """Called when the object is deleted."""
        try:
            self.kill_server()
        except: # fails if we already disconnected
            pass

    def start_server(self, port=None):
        """Starts the matter physics server and returns a socketio connection to it."""
        if port is None:
            with socketserver.TCPServer(("localhost", 0), None) as s:
                port = s.server_address[1]
        # print('Starting server on port {}'.format(port))
        subprocess.Popen(
            ['node', 'utils/matter_server.js', '--port', str(port)])
        sio = socketio.Client()
        # time.sleep(1)  # wait for server to start
        while True:
            try:
                sio.connect('http://localhost:' + str(port))
                break
            except:
                pass
        return sio

    def kill_server(self):
        """Kills the matter physics server."""
        try:
            self.socket.emit('disconnect')
        except:
            pass
        try:
            self.socket.disconnect()
        except:
            pass

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
        request_id = randint(0, 1000000)
        blocks = self.blocks_to_serializable(blocks)
        self.socket.emit('get_stability', {'id': request_id, 'blocks': blocks})
        while request_id not in self._results:
            pass
        # do wait here to reduce system load?
        result = self._results[request_id]
        del self._results[request_id]  # remove from dict
        return result
