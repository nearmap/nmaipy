"""Version information for nmaipy."""

__version__ = "5.0.0a1"
__version_info__ = tuple(
    int(i) for i in __version__.replace("a", ".").replace("b", ".").replace("rc", ".").split(".")[:3]
)
