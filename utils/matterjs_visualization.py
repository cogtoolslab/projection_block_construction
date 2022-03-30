"""Debug plotting code to be called from node.js matter_server.js for debugging purposes."""

import matplotlib.pyplot as plt
import json
import sys

vertices_string = sys.argv[1]

vertices = json.loads(vertices_string)

plt.figure(figsize=(6,6))

for v in vertices[::]:
    plt.plot([v[0][0], v[1][0]], [v[0][1], v[1][1]])
    plt.plot([v[1][0], v[2][0]], [v[1][1], v[2][1]])
    plt.plot([v[2][0], v[3][0]], [v[2][1], v[3][1]])
    plt.plot([v[3][0], v[0][0]], [v[3][1], v[0][1]])
plt.xlim(137,337)
plt.ylim(600,400)
plt.show()