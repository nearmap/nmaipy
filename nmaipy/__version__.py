"""Version information for nmaipy."""

__version__ = "4.0.5a3"
__version_info__ = tuple(int(i) for i in __version__.replace("a", ".").replace("b", ".").replace("rc", ".").split(".")[:3])
