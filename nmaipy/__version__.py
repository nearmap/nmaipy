"""Version information for nmaipy."""

__version__ = "4.2.0a6"
__version_info__ = tuple(int(i) for i in __version__.replace("a", ".").replace("b", ".").replace("rc", ".").split(".")[:3])
