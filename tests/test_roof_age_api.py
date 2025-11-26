"""
Tests for the Roof Age API client.

These tests verify:
- API response parsing
- Bulk query functionality
- Error handling
- Caching behavior
"""
import json
from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from nmaipy.roof_age_api import RoofAgeApi, RoofAgeAPIError
from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    ROOF_AGE_INSTALLATION_DATE_FIELD,
    ROOF_AGE_TRUST_SCORE_FIELD,
    ROOF_AGE_AREA_FIELD,
    ROOF_AGE_RESOURCE_ID_FIELD,
)


@pytest.fixture
def test_aoi_nj():
    """Test AOI from New Jersey (from test_parcels_2.csv)"""
    return Polygon([
        [-74.27516463201826, 40.64218565282876],
        [-74.27542320356542, 40.64233089427355],
        [-74.27515898040713, 40.64251014788475],
        [-74.27499790708188, 40.64236582723893],
        [-74.27516554166887, 40.64218907322704],
        [-74.27516463201826, 40.64218565282876]
    ])


@pytest.fixture
def test_roof_age_response(data_directory):
    """Load the test roof age API response from fixture file"""
    fixture_path = data_directory / "test_roof_age_nj_response.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def roof_age_api(cache_directory):
    """Create a RoofAgeApi instance for testing with cache enabled"""
    return RoofAgeApi(
        api_key="test_key",
        cache_dir=cache_directory,
        threads=2,
    )


def test_roof_age_api_initialization():
    """Test that RoofAgeApi initializes correctly"""
    api = RoofAgeApi(api_key="test_key")

    assert api.api_key == "test_key"
    assert "roofage" in api.base_url
    assert "latest" in api.base_url


def test_roof_age_api_missing_key():
    """Test that RoofAgeApi raises error when no API key is provided"""
    # Clear environment variable temporarily
    import os
    old_key = os.environ.get("API_KEY")
    if "API_KEY" in os.environ:
        del os.environ["API_KEY"]

    try:
        with pytest.raises(ValueError, match="No API KEY provided"):
            RoofAgeApi()
    finally:
        # Restore environment variable
        if old_key is not None:
            os.environ["API_KEY"] = old_key


def test_build_request_payload_aoi(roof_age_api, test_aoi_nj):
    """Test building request payload with AOI"""
    payload = roof_age_api._build_request_payload(aoi=test_aoi_nj)

    assert "aoi" in payload
    assert payload["aoi"]["type"] == "Polygon"
    assert len(payload["aoi"]["coordinates"][0]) == 6  # 5 points + closing


def test_build_request_payload_address(roof_age_api):
    """Test building request payload with address"""
    address = {
        "streetAddress": "123 Main St",
        "city": "Austin",
        "state": "TX",
        "zip": "78701"
    }
    payload = roof_age_api._build_request_payload(address=address)

    assert "address" in payload
    assert payload["address"] == address


def test_build_request_payload_both_raises(roof_age_api, test_aoi_nj):
    """Test that providing both AOI and address raises error"""
    address = {"streetAddress": "123 Main St", "city": "Austin", "state": "TX", "zip": "78701"}

    with pytest.raises(ValueError, match="Cannot specify both"):
        roof_age_api._build_request_payload(aoi=test_aoi_nj, address=address)


def test_build_request_payload_neither_raises(roof_age_api):
    """Test that providing neither AOI nor address raises error"""
    with pytest.raises(ValueError, match="Must specify either"):
        roof_age_api._build_request_payload()


def test_build_request_payload_incomplete_address(roof_age_api):
    """Test that incomplete address raises error"""
    incomplete_address = {"streetAddress": "123 Main St", "city": "Austin"}

    with pytest.raises(ValueError, match="Address missing required field"):
        roof_age_api._build_request_payload(address=incomplete_address)


def test_parse_response(roof_age_api, test_roof_age_response):
    """Test parsing API response into GeoDataFrame"""
    aoi_id = "test_aoi_1"
    gdf = roof_age_api._parse_response(test_roof_age_response, aoi_id)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    assert gdf.crs == API_CRS

    # Check that aoi_id is added
    assert AOI_ID_COLUMN_NAME in gdf.columns
    assert gdf[AOI_ID_COLUMN_NAME].iloc[0] == aoi_id

    # Check key fields
    assert ROOF_AGE_INSTALLATION_DATE_FIELD in gdf.columns
    assert gdf[ROOF_AGE_INSTALLATION_DATE_FIELD].iloc[0] == "2001-07-09"

    assert ROOF_AGE_TRUST_SCORE_FIELD in gdf.columns
    assert gdf[ROOF_AGE_TRUST_SCORE_FIELD].iloc[0] == 51.5

    assert ROOF_AGE_AREA_FIELD in gdf.columns
    assert gdf[ROOF_AGE_AREA_FIELD].iloc[0] == 107.66206

    # Check geometry
    assert not gdf.geometry.is_empty.any()
    assert all(gdf.geometry.geom_type == "Polygon")

    # Check that timeline was serialized to JSON string
    assert "timeline" in gdf.columns
    assert isinstance(gdf["timeline"].iloc[0], str)
    timeline_data = json.loads(gdf["timeline"].iloc[0])
    assert isinstance(timeline_data, list)
    assert len(timeline_data) == 2  # We have 2 timeline entries in the fixture


def test_parse_empty_response(roof_age_api):
    """Test parsing response with no features"""
    empty_response = {
        "resourceId": "test-resource",
        "type": "FeatureCollection",
        "features": [],
        "nextCursor": None
    }

    gdf = roof_age_api._parse_response(empty_response, "test_aoi")

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 0
    assert AOI_ID_COLUMN_NAME in gdf.columns


@pytest.mark.integration
def test_get_roof_age_by_aoi_real_api(test_aoi_nj):
    """
    Integration test with real API.

    This test requires:
    - Valid API_KEY environment variable
    - US location AOI
    - Internet connection
    """
    api = RoofAgeApi()  # Use API key from environment

    gdf = api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="nj_test_property")

    # Should get at least one roof (there's a house in this AOI)
    assert len(gdf) >= 1
    assert ROOF_AGE_INSTALLATION_DATE_FIELD in gdf.columns
    assert ROOF_AGE_TRUST_SCORE_FIELD in gdf.columns


def test_get_roof_age_by_aoi_mocked(roof_age_api, test_aoi_nj, test_roof_age_response):
    """Test get_roof_age_by_aoi with mocked API response"""
    # Mock the session.post method
    with patch.object(roof_age_api, '_session_scope') as mock_session_scope:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = test_roof_age_response
        mock_session.post.return_value = mock_response
        mock_session._timeout = (120, 90)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        gdf = roof_age_api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="test_aoi")

        assert len(gdf) == 1
        assert gdf[ROOF_AGE_INSTALLATION_DATE_FIELD].iloc[0] == "2001-07-09"


def test_get_roof_age_by_aoi_error(roof_age_api, test_aoi_nj):
    """Test that API errors are handled correctly"""
    with patch.object(roof_age_api, '_session_scope') as mock_session_scope:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_response.json.return_value = {"message": "No data available"}
        mock_session.post.return_value = mock_response
        mock_session._timeout = (120, 90)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with pytest.raises(RoofAgeAPIError):
            roof_age_api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="test_aoi")


def test_caching(roof_age_api, test_aoi_nj, test_roof_age_response):
    """Test that caching works correctly"""
    with patch.object(roof_age_api, '_session_scope') as mock_session_scope:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = test_roof_age_response
        mock_session.post.return_value = mock_response
        mock_session._timeout = (120, 90)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # First call should hit the API
        gdf1 = roof_age_api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="test_aoi")
        assert mock_session.post.call_count == 1

        # Second call should use cache
        gdf2 = roof_age_api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="test_aoi")
        assert mock_session.post.call_count == 1  # Should not increase

        # Results should be identical
        assert len(gdf1) == len(gdf2)
        assert gdf1[ROOF_AGE_INSTALLATION_DATE_FIELD].iloc[0] == gdf2[ROOF_AGE_INSTALLATION_DATE_FIELD].iloc[0]


def test_bulk_query(roof_age_api, test_roof_age_response):
    """Test bulk querying multiple AOIs"""
    # Create a small GeoDataFrame with 3 AOIs
    aois = [
        Polygon([[-74.275, 40.642], [-74.274, 40.642], [-74.274, 40.641], [-74.275, 40.641], [-74.275, 40.642]]),
        Polygon([[-74.276, 40.643], [-74.275, 40.643], [-74.275, 40.642], [-74.276, 40.642], [-74.276, 40.643]]),
        Polygon([[-74.277, 40.644], [-74.276, 40.644], [-74.276, 40.643], [-74.277, 40.643], [-74.277, 40.644]]),
    ]
    aoi_gdf = gpd.GeoDataFrame(
        geometry=aois,
        crs=API_CRS,
        index=pd.Index([0, 1, 2], name=AOI_ID_COLUMN_NAME)
    )

    # Mock the API responses
    with patch.object(roof_age_api, '_session_scope') as mock_session_scope:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = test_roof_age_response
        mock_session.post.return_value = mock_response
        mock_session._timeout = (120, 90)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        roofs_gdf, metadata_df, errors_df = roof_age_api.get_roof_age_bulk(aoi_gdf)

        assert len(roofs_gdf) == 3  # One roof per AOI
        assert len(metadata_df) == 3
        assert len(errors_df) == 0
        assert ROOF_AGE_RESOURCE_ID_FIELD in metadata_df.columns


def test_bulk_query_with_errors(roof_age_api):
    """Test bulk query error handling"""
    # Create a small GeoDataFrame
    aois = [
        Polygon([[-74.275, 40.642], [-74.274, 40.642], [-74.274, 40.641], [-74.275, 40.641], [-74.275, 40.642]]),
        Polygon([[-74.276, 40.643], [-74.275, 40.643], [-74.275, 40.642], [-74.276, 40.642], [-74.276, 40.643]]),
    ]
    aoi_gdf = gpd.GeoDataFrame(
        geometry=aois,
        crs=API_CRS,
        index=pd.Index([0, 1], name=AOI_ID_COLUMN_NAME)
    )

    # Mock one success and one failure
    with patch.object(roof_age_api, 'get_roof_age_by_aoi') as mock_get:
        def side_effect(aoi, aoi_id):
            if aoi_id == 0:
                # Return a valid GeoDataFrame
                return gpd.GeoDataFrame(
                    [{AOI_ID_COLUMN_NAME: aoi_id, ROOF_AGE_INSTALLATION_DATE_FIELD: "2020-01-01"}],
                    geometry=[aoi],
                    crs=API_CRS
                )
            else:
                # Raise an error
                raise RoofAgeAPIError(None, "test_url", message="Test error")

        mock_get.side_effect = side_effect

        roofs_gdf, metadata_df, errors_df = roof_age_api.get_roof_age_bulk(aoi_gdf, max_allowed_error_pct=50)

        assert len(roofs_gdf) == 1  # One success
        assert len(errors_df) == 1  # One error


def test_bulk_query_too_many_errors(roof_age_api):
    """Test that bulk query raises error if too many requests fail"""
    aois = [Polygon([[-74.275, 40.642], [-74.274, 40.642], [-74.274, 40.641], [-74.275, 40.641], [-74.275, 40.642]])]
    aoi_gdf = gpd.GeoDataFrame(
        geometry=aois,
        crs=API_CRS,
        index=pd.Index([0], name=AOI_ID_COLUMN_NAME)
    )

    # Mock all requests to fail
    with patch.object(roof_age_api, 'get_roof_age_by_aoi') as mock_get:
        mock_get.side_effect = RoofAgeAPIError(None, "test_url", message="Test error")

        with pytest.raises(ValueError, match="Error rate .* exceeds maximum"):
            roof_age_api.get_roof_age_bulk(aoi_gdf, max_allowed_error_pct=10)
