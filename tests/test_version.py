"""Tests for version management."""

import re
import pytest
from nmaipy import __version__, __version_info__


class TestVersion:
    """Test version formatting and consistency."""

    def test_version_format(self):
        """Test that version follows PEP 440 versioning (X.Y.Z with optional pre-release suffix)."""
        # Check version string format - allows X.Y.Z with optional pre-release suffix
        # e.g., 4.0.0, 4.0.0a1, 4.0.0b2, 4.0.0rc1
        pattern = r'^\d+\.\d+\.\d+(a|b|rc)?\d*$'
        assert re.match(pattern, __version__), f"Version {__version__} doesn't match PEP 440 versioning"

    def test_version_info_tuple(self):
        """Test that version_info is a tuple of integers."""
        assert isinstance(__version_info__, tuple), "version_info should be a tuple"
        assert len(__version_info__) == 3, "version_info should have 3 components (major, minor, patch)"

        for component in __version_info__:
            assert isinstance(component, int), f"Version component {component} should be an integer"
            assert component >= 0, f"Version component {component} should be non-negative"

    def test_version_consistency(self):
        """Test that version string base and version_info tuple match."""
        # Extract just the X.Y.Z part (without pre-release suffix) for comparison
        base_version = re.match(r'^(\d+\.\d+\.\d+)', __version__).group(1)
        version_from_tuple = '.'.join(str(x) for x in __version_info__)
        assert version_from_tuple == base_version, \
            f"Version mismatch: base={base_version}, tuple={version_from_tuple}"
