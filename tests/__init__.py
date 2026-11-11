import os

# conda-forge proj turns network grids on; tests must not depend on a grid download.
os.environ["PROJ_NETWORK"] = "OFF"
