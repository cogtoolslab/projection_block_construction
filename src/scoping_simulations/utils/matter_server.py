# This file contains helper functions for the matter physics server

import atexit
import copyreg
import os
import subprocess
import re

js_location = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "matter_server.js"
)


def launch_process():
    """Launches the matter physics server and return a handle to the process."""
    process = subprocess.Popen(
        ["node", js_location],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # wait for process to finish starting
    process.stdout.readline()  # the process prints "ready" when it's ready
    return process


# launch the server on import
process = launch_process()
# print("Started matter physics server 🧱.")


def check_process():
    """Checks if the process is still running and restarts if not."""
    global process
    if process.poll() is not None:
        process = launch_process()
        # print("Restarted matter physics server 🧱.")


class Physics_Server:
    def __init__(self, y_height=8) -> None:
        # if the height of the canvas differs, we need to subtract it to flip the y axis. 8 is default
        self.y_height = y_height
        self.start_server()

    def __del__(self):
        """Called when the object is deleted."""
        pass
        # self.kill_server()

    def start_server(self):
        """Starts the matter physics server. Does not need to be called manually—server will be started lazily."""
        global process
        check_process()
        self._process = process

    def kill_server(self):
        """Kills the matter physics server."""
        try:
            self._process.kill()
        except:
            pass

    def blocks_to_serializable(self, blocks):
        return [self.block_to_serializable(block) for block in blocks]

    def block_to_serializable(self, block):
        """Returns a serializable version of the block."""
        return {
            "x": float(block.x),
            "y": float(self.y_height - 1 - block.y),
            "w": float(block.width),
            "h": float(block.height),
        }

    def get_stability(self, blocks):
        """Returns the stability of the given blocks.
        Blocks until the result is known.
        """
        serialized_blocks = (
            str(self.blocks_to_serializable(blocks)).replace("'", '"') + "\n"
        )
        # send the request to the process via stdin
        try:
            self._process.stdin.write(serialized_blocks.encode("utf-8"))
            self._process.stdin.flush()
            # read the result from the process
            result = self._process.stdout.readline().decode("utf-8")
            # when launched from Jupyter Notebook we sometimes get ANSI codes back
            result = remove_ansi_codes(result)
        except BrokenPipeError:
            # if the process is dead, restart it
            self.kill_server()
            self.start_server()
            return self.get_stability(blocks)
        # return the result
        if result == "true\n":
            return True
        elif result == "false\n":
            return False
        elif result == "json_error\n":
            raise ValueError(
                f"Physics server reports json error: {self._process.stderr.read().decode('utf-8')} Input was: {serialized_blocks}"
            )
        else:
            raise ValueError(
                f"Unexpected output from physics server: {result}\nInput was: {serialized_blocks}\nFull output: {result}."
            )
            # print(f"Unexpected output from physics server: {result}\nInput was: {serialized_blocks}Full output: {self._process.stdout.read().decode('utf-8')}.")
            # print("Restarting physics server...")
            # self.kill_server()
            # self.start_server()
            # # wait a little while
            # time.sleep(1)
            return self.get_stability(blocks)


def remove_ansi_codes(text):
    ansi_escape_pattern = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape_pattern.sub("", text)


def pickle_physics_server(server):
    """Pickle function for physics server. A new process is started when unpickled."""
    return (Physics_Server, (server.y_height,))


# register custom pickle function for the server
copyreg.pickle(Physics_Server, pickle_physics_server)


@atexit.register
def killallprocesses():
    """Kills all the processes that are still running once we close the file (ie. are done with everything)."""
    global process
    process.kill()
