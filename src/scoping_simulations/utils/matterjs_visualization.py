"""Debug plotting code to be called from node.js matter_server.js for debugging purposes."""

import json
import sys

import matplotlib.pyplot as plt

vertices_string = sys.argv[1]

vertices = json.loads(vertices_string)

plt.figure(figsize=(6, 6))

for v in vertices[::]:
    plt.plot([v[0][0], v[1][0]], [v[0][1], v[1][1]])
    plt.plot([v[1][0], v[2][0]], [v[1][1], v[2][1]])
    plt.plot([v[2][0], v[3][0]], [v[2][1], v[3][1]])
    plt.plot([v[3][0], v[0][0]], [v[3][1], v[0][1]])
plt.xlim(120, 820)
plt.ylim(600, 0)
plt.show()
