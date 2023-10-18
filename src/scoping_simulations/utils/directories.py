"""This files hold paths to the default location of the various directories. 
By default, the directories are in the user directory under the name "scoping_simulations". This can be changed by setting the environment variable "SCOPING_SIMULATIONS_DIR" to the desired location.
The directories are then created in the specified location. The default location is then used as a fallback if the environment variable is not set.
"""

# do we have an environment variable set?
import os
from pathlib import Path

if "SCOPING_SIMULATIONS_DIR" in os.environ:
    # yes, use that
    PROJ_DIR = Path(os.environ["SCOPING_SIMULATIONS_DIR"])
else:
    PROJ_DIR = Path.home() / "tools_block_construction"

STIM_DIR = PROJ_DIR / "stimuli"

EXPERIMENTS_DIR = PROJ_DIR / "experiments"

RESULTS_DIR = PROJ_DIR / "results"

DF_DIR = RESULTS_DIR / "dataframes"

# create the directories if they don't exist
for d in [PROJ_DIR, STIM_DIR, EXPERIMENTS_DIR, RESULTS_DIR, DF_DIR]:
    if not d.exists():
        print(f"Creating directory {d}")
        d.mkdir()
