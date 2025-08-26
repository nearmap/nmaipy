"""Tests for version management."""

import re
import pytest
from nmaipy import __version__, __version_info__


class TestVersion:
    """Test version formatting and consistency."""
    
    def test_version_format(self):
        """Test that version follows semantic versioning (X.Y.Z)."""
        # Check version string format
        pattern = r'^\d+\.\d+\.\d+$'
        assert re.match(pattern, __version__), f"Version {__version__} doesn't match semantic versioning"
    
    def test_version_info_tuple(self):
        """Test that version_info is a tuple of integers."""
        assert isinstance(__version_info__, tuple), "version_info should be a tuple"
        assert len(__version_info__) == 3, "version_info should have 3 components (major, minor, patch)"
        
        for component in __version_info__:
            assert isinstance(component, int), f"Version component {component} should be an integer"
            assert component >= 0, f"Version component {component} should be non-negative"
    
    def test_version_consistency(self):
        """Test that version string and version_info tuple match."""
        version_from_tuple = '.'.join(str(x) for x in __version_info__)
        assert version_from_tuple == __version__, \
            f"Version mismatch: string={__version__}, tuple={version_from_tuple}"