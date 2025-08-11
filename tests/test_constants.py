"""Tests for constants validation."""

import pytest
import uuid
from nmaipy.constants import (
    MAX_AOI_AREA_SQM_BEFORE_GRIDDING,
    API_CRS,
    AREA_CRS,
    DEFAULT_URL_ROOT,
    AOI_ID_COLUMN_NAME,
    GRID_SIZE_DEGREES,
    MAX_RETRIES,
    METERS_TO_FEET,
    SQUARED_METERS_TO_SQUARED_FEET,
)


class TestConstants:
    """Test that constants are properly defined and valid."""
    
    def test_gridding_threshold(self):
        """Test that gridding threshold is reasonable."""
        assert MAX_AOI_AREA_SQM_BEFORE_GRIDDING > 0, "Gridding threshold should be positive"
        assert MAX_AOI_AREA_SQM_BEFORE_GRIDDING == 1_000_000, "Should be 1 sq km as per requirements"
        # 1 sq km is reasonable - not too small (causing excessive gridding) nor too large
        assert 100_000 <= MAX_AOI_AREA_SQM_BEFORE_GRIDDING <= 10_000_000, \
            "Gridding threshold should be between 0.1 and 10 sq km"
    
    def test_crs_definitions(self):
        """Test that CRS definitions are valid."""
        # API CRS should be WGS84
        assert API_CRS.upper() == "EPSG:4326", "API should use WGS84 coordinate system"
        
        # Area CRS should be defined for each region
        assert isinstance(AREA_CRS, dict), "AREA_CRS should be a dictionary"
        
        # Check required regions
        required_regions = ['au', 'us', 'nz', 'ca']
        for region in required_regions:
            assert region in AREA_CRS, f"Missing CRS for region: {region}"
            
            # Check CRS format
            crs = AREA_CRS[region]
            assert isinstance(crs, str), f"CRS for {region} should be a string"
            # CRS can be EPSG or ESRI format
            assert crs.upper().startswith(("EPSG:", "ESRI:")), \
                f"CRS for {region} should be EPSG or ESRI format"
    
    def test_processing_defaults(self):
        """Test that processing defaults are reasonable."""
        # Grid size should be positive
        assert GRID_SIZE_DEGREES > 0, "Grid size should be positive"
        assert GRID_SIZE_DEGREES < 1, "Grid size should be less than 1 degree"
        
        # Max retries should be reasonable
        assert MAX_RETRIES > 0, "Max retries should be positive"
        assert MAX_RETRIES <= 1000, "Too many retries might cause issues"
    
    def test_url_root(self):
        """Test that API URL is properly formatted."""
        # DEFAULT_URL_ROOT doesn't include https:// prefix
        assert "api.nearmap.com" in DEFAULT_URL_ROOT, "Should point to Nearmap API"
        assert not DEFAULT_URL_ROOT.endswith("/"), "URL root should not end with slash"
    
    def test_column_names(self):
        """Test that column names are defined."""
        assert AOI_ID_COLUMN_NAME, "AOI ID column name should be defined"
        assert isinstance(AOI_ID_COLUMN_NAME, str), "Column name should be a string"
        assert AOI_ID_COLUMN_NAME == "aoi_id", "Standard AOI ID column name"
    
    def test_conversion_factors(self):
        """Test that conversion factors are correct."""
        # Check meters to feet conversion
        assert METERS_TO_FEET == pytest.approx(3.28084, rel=1e-5), "Meters to feet conversion incorrect"
        
        # Check squared conversion
        assert SQUARED_METERS_TO_SQUARED_FEET == pytest.approx(METERS_TO_FEET * METERS_TO_FEET, rel=1e-5), \
            "Squared meters to squared feet conversion incorrect"
    
    def test_no_negative_values(self):
        """Test that numeric constants are not negative where inappropriate."""
        # These should all be positive
        positive_constants = [
            MAX_AOI_AREA_SQM_BEFORE_GRIDDING,
            GRID_SIZE_DEGREES,
            MAX_RETRIES,
            METERS_TO_FEET,
            SQUARED_METERS_TO_SQUARED_FEET,
        ]
        
        for const in positive_constants:
            assert const > 0, f"Constant should be positive: {const}"