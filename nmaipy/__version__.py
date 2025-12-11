"""Version information for nmaipy."""

import re

__version__ = "4.0.0a3"
__version_info__ = tuple(int(re.match(r"\d+", part).group()) for part in __version__.split("."))
