"""
Tests for the Roof Age API client.

These tests verify:
- API response parsing
- Bulk query functionality
- Error handling
- Caching behavior
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    FEATURE_CLASS_DESCRIPTIONS,
    ROOF_AGE_AREA_FIELD,
    ROOF_AGE_INSTALLATION_DATE_FIELD,
    ROOF_AGE_MAPBROWSER_URL_FIELD,
    ROOF_AGE_MAPBROWSER_URL_OUTPUT_FIELD,
    ROOF_AGE_NEXT_CURSOR_FIELD,
    ROOF_AGE_RESOURCE_ID_FIELD,
    ROOF_AGE_TRUST_SCORE_FIELD,
    ROOF_INSTANCE_CLASS_ID,
)
from nmaipy.exporter import export_feature_class
from nmaipy.roof_age_api import RoofAgeApi, RoofAgeAPIError


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


def test_roof_age_api_bulk_mode_default():
    """Test that RoofAgeApi defaults to bulk_mode=True"""
    api = RoofAgeApi(api_key="test_key")
    assert api.bulk_mode is True


def test_roof_age_api_bulk_mode_disabled():
    """Test that RoofAgeApi can disable bulk_mode"""
    api = RoofAgeApi(api_key="test_key", bulk_mode=False)
    assert api.bulk_mode is False


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
    # The API adds a "country" field to the address
    expected_address = {**address, "country": "US"}
    assert payload["address"] == expected_address


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

    # Check that mapbrowserURL was renamed and has ?locationMarker appended
    assert ROOF_AGE_MAPBROWSER_URL_FIELD not in gdf.columns, "Original mapbrowserURL should be removed"
    assert ROOF_AGE_MAPBROWSER_URL_OUTPUT_FIELD in gdf.columns, "roof_age_mapbrowser_url should be present"
    url = gdf[ROOF_AGE_MAPBROWSER_URL_OUTPUT_FIELD].iloc[0]
    assert url.endswith("?locationMarker"), f"URL should end with ?locationMarker, got: {url}"


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
    """Test bulk query error handling - errors are returned in errors_df, not raised"""
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

        # The bulk method returns errors in errors_df rather than raising exceptions
        roofs_gdf, metadata_df, errors_df = roof_age_api.get_roof_age_bulk(aoi_gdf)

        assert len(roofs_gdf) == 1  # One success
        assert len(errors_df) == 1  # One error
        assert errors_df.index[0] == 1  # The failed AOI


def test_bulk_query_all_errors(roof_age_api):
    """Test that bulk query returns all errors when all requests fail"""
    aois = [Polygon([[-74.275, 40.642], [-74.274, 40.642], [-74.274, 40.641], [-74.275, 40.641], [-74.275, 40.642]])]
    aoi_gdf = gpd.GeoDataFrame(
        geometry=aois,
        crs=API_CRS,
        index=pd.Index([0], name=AOI_ID_COLUMN_NAME)
    )

    # Mock all requests to fail
    with patch.object(roof_age_api, 'get_roof_age_by_aoi') as mock_get:
        mock_get.side_effect = RoofAgeAPIError(None, "test_url", message="Test error")

        # All errors are returned in errors_df
        roofs_gdf, metadata_df, errors_df = roof_age_api.get_roof_age_bulk(aoi_gdf)

        assert len(roofs_gdf) == 0  # No successes
        assert len(errors_df) == 1  # One error
        assert "message" in errors_df.columns


def test_pagination(roof_age_api, test_aoi_nj):
    """Test that pagination correctly fetches all pages and merges results"""
    # Create mock responses for multiple pages
    page1_response = {
        "type": "FeatureCollection",
        "resourceId": "test-resource-123",
        ROOF_AGE_NEXT_CURSOR_FIELD: "cursor_page_2",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[-74.275, 40.642], [-74.274, 40.642], [-74.274, 40.641], [-74.275, 40.641], [-74.275, 40.642]]]},
                "properties": {ROOF_AGE_INSTALLATION_DATE_FIELD: "2020-01-01", ROOF_AGE_TRUST_SCORE_FIELD: 80.0, ROOF_AGE_AREA_FIELD: 100.0, "kind": "roof"}
            },
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[-74.276, 40.643], [-74.275, 40.643], [-74.275, 40.642], [-74.276, 40.642], [-74.276, 40.643]]]},
                "properties": {ROOF_AGE_INSTALLATION_DATE_FIELD: "2019-05-15", ROOF_AGE_TRUST_SCORE_FIELD: 75.0, ROOF_AGE_AREA_FIELD: 150.0, "kind": "roof"}
            }
        ]
    }
    page2_response = {
        "type": "FeatureCollection",
        "resourceId": "test-resource-123",
        # No nextCursor - this is the last page
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[-74.277, 40.644], [-74.276, 40.644], [-74.276, 40.643], [-74.277, 40.643], [-74.277, 40.644]]]},
                "properties": {ROOF_AGE_INSTALLATION_DATE_FIELD: "2018-08-20", ROOF_AGE_TRUST_SCORE_FIELD: 90.0, ROOF_AGE_AREA_FIELD: 200.0, "kind": "roof"}
            }
        ]
    }

    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_response = Mock()
        mock_response.ok = True
        if call_count == 1:
            mock_response.json.return_value = page1_response
        else:
            mock_response.json.return_value = page2_response
        return mock_response

    with patch.object(roof_age_api, '_session_scope') as mock_session_scope:
        mock_session = Mock()
        mock_session.post.side_effect = mock_post
        mock_session._timeout = (120, 90)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        gdf = roof_age_api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="test_pagination")

        # Should have made 2 API calls (2 pages)
        assert call_count == 2

        # Should have all 3 features (2 from page 1 + 1 from page 2)
        assert len(gdf) == 3

        # Verify features from both pages are present
        dates = gdf[ROOF_AGE_INSTALLATION_DATE_FIELD].tolist()
        assert "2020-01-01" in dates
        assert "2019-05-15" in dates
        assert "2018-08-20" in dates

        # Verify resource ID was captured from first page
        assert ROOF_AGE_RESOURCE_ID_FIELD in gdf.columns
        assert gdf[ROOF_AGE_RESOURCE_ID_FIELD].iloc[0] == "test-resource-123"


def test_pagination_with_limit(roof_age_api, test_aoi_nj):
    """Test that limit parameter is passed correctly to API"""
    with patch.object(roof_age_api, '_session_scope') as mock_session_scope:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "type": "FeatureCollection",
            "resourceId": "test-resource",
            "features": []
        }
        mock_session.post.return_value = mock_response
        mock_session._timeout = (120, 90)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        roof_age_api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="test_limit", limit=500)

        # Check that limit was included in the request payload
        call_args = mock_session.post.call_args
        payload = call_args.kwargs.get('json', call_args[1].get('json', {}))
        assert payload.get("limit") == 500


def test_build_request_payload_with_pagination(roof_age_api, test_aoi_nj):
    """Test that pagination parameters are included in payload"""
    # Without pagination params
    payload_basic = roof_age_api._build_request_payload(aoi=test_aoi_nj)
    assert "cursor" not in payload_basic
    assert "limit" not in payload_basic

    # With cursor only
    payload_cursor = roof_age_api._build_request_payload(aoi=test_aoi_nj, cursor="test_cursor_abc")
    assert payload_cursor["cursor"] == "test_cursor_abc"
    assert "limit" not in payload_cursor

    # With limit only
    payload_limit = roof_age_api._build_request_payload(aoi=test_aoi_nj, limit=100)
    assert "cursor" not in payload_limit
    assert payload_limit["limit"] == 100

    # With both
    payload_both = roof_age_api._build_request_payload(aoi=test_aoi_nj, cursor="cursor_xyz", limit=250)
    assert payload_both["cursor"] == "cursor_xyz"
    assert payload_both["limit"] == 250


@pytest.mark.integration
def test_pagination_real_api():
    """
    Integration test for pagination with real API.

    Uses Breezy Point, NY which has >1000 roof instances, requiring pagination.

    This test requires:
    - Valid API_KEY environment variable
    - Internet connection
    """
    api = RoofAgeApi()

    address = {
        "streetAddress": "21702 BREEZY POINT BLVD",
        "city": "BREEZY POINT",
        "state": "NY",
        "zip": "11697",
    }

    gdf = api.get_roof_age_by_address(address, aoi_id="breezy_point_pagination_test")

    # Breezy Point has >1000 roof instances, so pagination must have occurred
    assert len(gdf) > 1000, f"Expected >1000 features (pagination required), got {len(gdf)}"
    assert ROOF_AGE_INSTALLATION_DATE_FIELD in gdf.columns
    assert ROOF_AGE_RESOURCE_ID_FIELD in gdf.columns


def test_bulk_mode_parameter_included_in_request(cache_directory, test_aoi_nj):
    """Test that bulk=true parameter is included in request when bulk_mode=True"""
    api = RoofAgeApi(api_key="test_key", cache_dir=cache_directory, bulk_mode=True)

    with patch.object(api, '_session_scope') as mock_session_scope:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "type": "FeatureCollection",
            "resourceId": "test-resource",
            "features": []
        }
        mock_session.post.return_value = mock_response
        mock_session._timeout = (120, 90)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="test_bulk")

        # Check that bulk=true was included in params
        call_args = mock_session.post.call_args
        params = call_args.kwargs.get('params', call_args[1].get('params', {}))
        assert params.get("bulk") == "true", f"bulk param should be 'true', got {params}"


def test_bulk_mode_parameter_not_included_when_disabled(cache_directory, test_aoi_nj):
    """Test that bulk parameter is not included when bulk_mode=False"""
    api = RoofAgeApi(api_key="test_key", cache_dir=cache_directory, bulk_mode=False)

    with patch.object(api, '_session_scope') as mock_session_scope:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "type": "FeatureCollection",
            "resourceId": "test-resource",
            "features": []
        }
        mock_session.post.return_value = mock_response
        mock_session._timeout = (120, 90)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="test_no_bulk")

        # Check that bulk param is NOT included
        call_args = mock_session.post.call_args
        params = call_args.kwargs.get('params', call_args[1].get('params', {}))
        assert "bulk" not in params, f"bulk param should not be present, got {params}"


@pytest.mark.integration
def test_bulk_export_with_parcels_2(parcels_2_gdf):
    """
    Integration test for bulk roof age query and export with 100 NJ parcels.

    Fetches roof instances from the Roof Age API and exports them via
    export_feature_class, verifying the output contains expected columns.
    """
    if not os.environ.get("API_KEY"):
        pytest.skip("API_KEY not set")

    api = RoofAgeApi()
    roofs_gdf, metadata_df, errors_df = api.get_roof_age_bulk(parcels_2_gdf)

    if len(roofs_gdf) == 0:
        pytest.skip("No roof instances returned from API")

    roofs_gdf = roofs_gdf.reset_index()

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path, _ = export_feature_class(
            features_gdf=roofs_gdf,
            class_id=ROOF_INSTANCE_CLASS_ID,
            class_description=FEATURE_CLASS_DESCRIPTIONS[ROOF_INSTANCE_CLASS_ID],
            output_stem=f"{tmpdir}/test",
            export_csv=True,
            export_parquet=False,
            country="us",
        )

        assert csv_path is not None
        assert csv_path.exists()

        result_df = pd.read_csv(csv_path)

        assert len(result_df) == len(roofs_gdf)

        expected_cols = ["roof_age_installation_date", "roof_age_trust_score", "roof_age_evidence_type"]
        for col in expected_cols:
            assert col in result_df.columns, f"Missing column: {col}"

        # Most roof instances should have installation dates (>90%)
        install_date_coverage = result_df["roof_age_installation_date"].notna().mean()
        assert install_date_coverage > 0.9, f"Only {install_date_coverage:.0%} have installation dates"

        # Check evidence types are present
        assert result_df["roof_age_evidence_type"].notna().any(), "No evidence types present"
