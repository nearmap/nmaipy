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
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from nmaipy import parcels
from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    FEATURE_CLASS_DESCRIPTIONS,
    ROOF_AGE_AREA_FIELD,
    ROOF_AGE_NEXT_CURSOR_FIELD,
    ROOF_AGE_PREFIX_COLUMNS,
    ROOF_ID,
    ROOF_INSTANCE_CLASS_ID,
)
from nmaipy.exporter import export_feature_class
from nmaipy.roof_age_api import RoofAgeApi, RoofAgeAPIError


@pytest.mark.skip("Comment out this line if you wish to regen the test data")
def test_gen_roof_age_data(parcels_2_gdf, data_directory: Path, cache_directory: Path):
    """Generate cached roof age data for NJ parcels (3 AOIs)."""
    small_gdf = parcels_2_gdf.head(3)
    api = RoofAgeApi(cache_dir=cache_directory)
    roofs_gdf, metadata_df, errors_df = api.get_roof_age_bulk(small_gdf)
    roofs_gdf.to_csv(data_directory / "test_roof_age_bulk_nj.csv", index=False)
    metadata_df.to_csv(data_directory / "test_roof_age_metadata_nj.csv")


@pytest.fixture
def test_aoi_nj():
    """Test AOI from New Jersey (from test_parcels_2.csv)"""
    return Polygon(
        [
            [-74.27516463201826, 40.64218565282876],
            [-74.27542320356542, 40.64233089427355],
            [-74.27515898040713, 40.64251014788475],
            [-74.27499790708188, 40.64236582723893],
            [-74.27516554166887, 40.64218907322704],
            [-74.27516463201826, 40.64218565282876],
        ]
    )


@pytest.fixture
def test_roof_age_response(data_directory):
    """Load the test roof age API response from fixture file"""
    fixture_path = data_directory / "test_roof_age_nj_response.json"
    with open(fixture_path, "r") as f:
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
        "zip": "78701",
    }
    payload = roof_age_api._build_request_payload(address=address)

    assert "address" in payload
    # The API adds a "country" field to the address
    expected_address = {**address, "country": "US"}
    assert payload["address"] == expected_address


def test_build_request_payload_both_raises(roof_age_api, test_aoi_nj):
    """Test that providing both AOI and address raises error"""
    address = {
        "streetAddress": "123 Main St",
        "city": "Austin",
        "state": "TX",
        "zip": "78701",
    }

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

    # Check key fields (snake_case after parse, matching feature_api.py pattern)
    assert "installation_date" in gdf.columns
    assert gdf["installation_date"].iloc[0] == "2001-07-09"

    assert "trust_score" in gdf.columns
    assert gdf["trust_score"].iloc[0] == 51.5

    # area is dropped (mapped to area_sqm)
    assert "area" not in gdf.columns
    assert "area_sqm" in gdf.columns
    assert gdf["area_sqm"].iloc[0] == 107.66206

    # Check geometry
    assert not gdf.geometry.is_empty.any()
    assert all(gdf.geometry.geom_type == "Polygon")

    # timeline is dropped after serialization (internal field)
    assert "timeline" not in gdf.columns

    # hilbert_id is dropped (mapped to feature_id)
    assert "hilbert_id" not in gdf.columns
    assert "feature_id" in gdf.columns

    # mapBrowserUrl → map_browser_url with ?locationMarker appended
    assert "map_browser_url" in gdf.columns
    url = gdf["map_browser_url"].iloc[0]
    assert url.endswith("?locationMarker"), f"URL should end with ?locationMarker, got: {url}"


def test_parse_empty_response(roof_age_api):
    """Test parsing response with no features"""
    empty_response = {
        "resourceId": "test-resource",
        "type": "FeatureCollection",
        "features": [],
        "nextCursor": None,
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
    assert "installation_date" in gdf.columns
    assert "trust_score" in gdf.columns


def test_get_roof_age_by_aoi_mocked(roof_age_api, test_aoi_nj, test_roof_age_response):
    """Test get_roof_age_by_aoi with mocked API response"""
    # Mock the session.post method
    with patch.object(roof_age_api, "_session_scope") as mock_session_scope:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = test_roof_age_response
        mock_session.post.return_value = mock_response
        mock_session._timeout = (120, 90)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        gdf = roof_age_api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="test_aoi")

        assert len(gdf) == 1
        assert gdf["installation_date"].iloc[0] == "2001-07-09"


def test_get_roof_age_by_aoi_error(roof_age_api, test_aoi_nj):
    """Test that API errors are handled correctly"""
    with patch.object(roof_age_api, "_session_scope") as mock_session_scope:
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
    with patch.object(roof_age_api, "_session_scope") as mock_session_scope:
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
        assert gdf1["installation_date"].iloc[0] == gdf2["installation_date"].iloc[0]


def test_bulk_query(roof_age_api, test_roof_age_response):
    """Test bulk querying multiple AOIs"""
    # Create a small GeoDataFrame with 3 AOIs
    aois = [
        Polygon(
            [
                [-74.275, 40.642],
                [-74.274, 40.642],
                [-74.274, 40.641],
                [-74.275, 40.641],
                [-74.275, 40.642],
            ]
        ),
        Polygon(
            [
                [-74.276, 40.643],
                [-74.275, 40.643],
                [-74.275, 40.642],
                [-74.276, 40.642],
                [-74.276, 40.643],
            ]
        ),
        Polygon(
            [
                [-74.277, 40.644],
                [-74.276, 40.644],
                [-74.276, 40.643],
                [-74.277, 40.643],
                [-74.277, 40.644],
            ]
        ),
    ]
    aoi_gdf = gpd.GeoDataFrame(geometry=aois, crs=API_CRS, index=pd.Index([0, 1, 2], name=AOI_ID_COLUMN_NAME))

    # Mock the API responses
    with patch.object(roof_age_api, "_session_scope") as mock_session_scope:
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
        assert "resource_id" in metadata_df.columns


def test_bulk_query_with_errors(roof_age_api):
    """Test bulk query error handling - errors are returned in errors_df, not raised"""
    # Create a small GeoDataFrame
    aois = [
        Polygon(
            [
                [-74.275, 40.642],
                [-74.274, 40.642],
                [-74.274, 40.641],
                [-74.275, 40.641],
                [-74.275, 40.642],
            ]
        ),
        Polygon(
            [
                [-74.276, 40.643],
                [-74.275, 40.643],
                [-74.275, 40.642],
                [-74.276, 40.642],
                [-74.276, 40.643],
            ]
        ),
    ]
    aoi_gdf = gpd.GeoDataFrame(geometry=aois, crs=API_CRS, index=pd.Index([0, 1], name=AOI_ID_COLUMN_NAME))

    # Mock one success and one failure
    with patch.object(roof_age_api, "get_roof_age_by_aoi") as mock_get:

        def side_effect(aoi, aoi_id):
            if aoi_id == 0:
                # Return a valid GeoDataFrame
                return gpd.GeoDataFrame(
                    [{AOI_ID_COLUMN_NAME: aoi_id, "installation_date": "2020-01-01"}],
                    geometry=[aoi],
                    crs=API_CRS,
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
    aois = [
        Polygon(
            [
                [-74.275, 40.642],
                [-74.274, 40.642],
                [-74.274, 40.641],
                [-74.275, 40.641],
                [-74.275, 40.642],
            ]
        )
    ]
    aoi_gdf = gpd.GeoDataFrame(geometry=aois, crs=API_CRS, index=pd.Index([0], name=AOI_ID_COLUMN_NAME))

    # Mock all requests to fail
    with patch.object(roof_age_api, "get_roof_age_by_aoi") as mock_get:
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
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-74.275, 40.642],
                            [-74.274, 40.642],
                            [-74.274, 40.641],
                            [-74.275, 40.641],
                            [-74.275, 40.642],
                        ]
                    ],
                },
                "properties": {
                    "installationDate": "2020-01-01",
                    "trustScore": 80.0,
                    ROOF_AGE_AREA_FIELD: 100.0,
                    "kind": "roof",
                },
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-74.276, 40.643],
                            [-74.275, 40.643],
                            [-74.275, 40.642],
                            [-74.276, 40.642],
                            [-74.276, 40.643],
                        ]
                    ],
                },
                "properties": {
                    "installationDate": "2019-05-15",
                    "trustScore": 75.0,
                    ROOF_AGE_AREA_FIELD: 150.0,
                    "kind": "roof",
                },
            },
        ],
    }
    page2_response = {
        "type": "FeatureCollection",
        "resourceId": "test-resource-123",
        # No nextCursor - this is the last page
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-74.277, 40.644],
                            [-74.276, 40.644],
                            [-74.276, 40.643],
                            [-74.277, 40.643],
                            [-74.277, 40.644],
                        ]
                    ],
                },
                "properties": {
                    "installationDate": "2018-08-20",
                    "trustScore": 90.0,
                    ROOF_AGE_AREA_FIELD: 200.0,
                    "kind": "roof",
                },
            }
        ],
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

    with patch.object(roof_age_api, "_session_scope") as mock_session_scope:
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
        dates = gdf["installation_date"].tolist()
        assert "2020-01-01" in dates
        assert "2019-05-15" in dates
        assert "2018-08-20" in dates

        # Verify resource ID was captured from first page
        assert "resource_id" in gdf.columns
        assert gdf["resource_id"].iloc[0] == "test-resource-123"


def test_pagination_with_limit(roof_age_api, test_aoi_nj):
    """Test that limit parameter is passed correctly to API"""
    with patch.object(roof_age_api, "_session_scope") as mock_session_scope:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "type": "FeatureCollection",
            "resourceId": "test-resource",
            "features": [],
        }
        mock_session.post.return_value = mock_response
        mock_session._timeout = (120, 90)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        roof_age_api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="test_limit", limit=500)

        # Check that limit was included in the request payload
        call_args = mock_session.post.call_args
        payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
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
    assert "installation_date" in gdf.columns
    assert "resource_id" in gdf.columns


def test_bulk_mode_parameter_included_in_request(cache_directory, test_aoi_nj):
    """Test that bulk=true parameter is included in request when bulk_mode=True"""
    api = RoofAgeApi(api_key="test_key", cache_dir=cache_directory, bulk_mode=True)

    with patch.object(api, "_session_scope") as mock_session_scope:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "type": "FeatureCollection",
            "resourceId": "test-resource",
            "features": [],
        }
        mock_session.post.return_value = mock_response
        mock_session._timeout = (120, 90)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="test_bulk")

        # Check that bulk=true was included in params
        call_args = mock_session.post.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
        assert params.get("bulk") == "true", f"bulk param should be 'true', got {params}"


def test_bulk_mode_parameter_not_included_when_disabled(cache_directory, test_aoi_nj):
    """Test that bulk parameter is not included when bulk_mode=False"""
    api = RoofAgeApi(api_key="test_key", cache_dir=cache_directory, bulk_mode=False)

    with patch.object(api, "_session_scope") as mock_session_scope:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "type": "FeatureCollection",
            "resourceId": "test-resource",
            "features": [],
        }
        mock_session.post.return_value = mock_response
        mock_session._timeout = (120, 90)
        mock_session_scope.return_value.__enter__.return_value = mock_session

        api.get_roof_age_by_aoi(test_aoi_nj, aoi_id="test_no_bulk")

        # Check that bulk param is NOT included
        call_args = mock_session.post.call_args
        params = call_args.kwargs.get("params", call_args[1].get("params", {}))
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
            output_dir=tmpdir,
            tabular_file_format="csv",
            export_geo_parquet=False,
            country="us",
        )

        assert csv_path is not None
        assert Path(csv_path).exists()

        result_df = pd.read_csv(csv_path)

        assert len(result_df) == len(roofs_gdf)

        expected_cols = [
            "roof_age_installation_date",
            "roof_age_trust_score",
            "roof_age_evidence_type",
        ]
        for col in expected_cols:
            assert col in result_df.columns, f"Missing column: {col}"

        # Most roof instances should have installation dates (>90%)
        install_date_coverage = result_df["roof_age_installation_date"].notna().mean()
        assert install_date_coverage > 0.9, f"Only {install_date_coverage:.0%} have installation dates"

        # Check evidence types are present
        assert result_df["roof_age_evidence_type"].notna().any(), "No evidence types present"


def test_all_roof_age_fields_accounted_for(roof_age_api, test_roof_age_response):
    """Every column from _parse_response must be in ROOF_AGE_PREFIX_COLUMNS or known structural columns.

    This guards against new API fields being silently dropped by the whitelist.
    If a new field appears in the Roof Age API response, this test will fail and
    prompt you to add it to ROOF_AGE_PREFIX_COLUMNS in constants.py.
    """
    gdf = roof_age_api._parse_response(test_roof_age_response, "test_aoi")
    structural = {
        AOI_ID_COLUMN_NAME,
        "feature_id",
        "class_id",
        "geometry",
        "area_sqm",
        "resource_id",
        "description",
    }
    for col in gdf.columns:
        assert col in ROOF_AGE_PREFIX_COLUMNS or col in structural, (
            f"Column '{col}' from Roof Age API not in ROOF_AGE_PREFIX_COLUMNS or structural columns — "
            f"add it to the whitelist in constants.py"
        )


class TestRoofAgeFieldMapping:
    """
    Tests that export_feature_class correctly adds roof_age_ prefix to
    Roof Age API columns and rejects Feature API columns that leak through
    after pd.concat(feature_api_features, roof_age_features).

    Uses real cached data from the Roof Age API (test_roof_age_bulk_nj.parquet)
    combined with real Feature API data (test_features_2.csv) to replicate the
    actual data flow in the exporter.
    """

    @pytest.fixture
    def combined_features_gdf(self, roof_age_gdf, features_2_gdf):
        """Simulate the pd.concat that happens in the exporter's process_chunk.

        This is the critical test setup: after concat, roof instance rows inherit
        ALL Feature API columns (link, system_version, aoi_geometry, etc.) with NaN values.
        The whitelist in export_feature_class must filter these out.
        """
        # Prepare roof age data with aoi_id as index (matching exporter behaviour)
        ra = roof_age_gdf.copy()
        if AOI_ID_COLUMN_NAME in ra.columns:
            ra = ra.set_index(AOI_ID_COLUMN_NAME)

        # Take a small slice of Feature API features for the same AOIs
        matching_aoi_ids = ra.index.unique()
        fa = features_2_gdf[features_2_gdf.index.isin(matching_aoi_ids)].copy()

        # Concat just like exporter.py line ~2203
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*concatenation with empty or all-NA.*")
            combined = gpd.GeoDataFrame(
                pd.concat([fa, ra], ignore_index=False),
                crs=API_CRS,
            )
        return combined

    def test_roof_instance_export_whitelists_columns(self, combined_features_gdf):
        """Only ROOF_AGE_PREFIX_COLUMNS get roof_age_ prefix; Feature API columns are excluded."""
        # Extract just the roof instance rows (as the exporter does)
        ri_features = combined_features_gdf[combined_features_gdf["class_id"] == ROOF_INSTANCE_CLASS_ID]
        assert len(ri_features) > 0, "Should have roof instance features in combined data"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, _ = export_feature_class(
                features_gdf=ri_features,
                class_id=ROOF_INSTANCE_CLASS_ID,
                class_description=FEATURE_CLASS_DESCRIPTIONS[ROOF_INSTANCE_CLASS_ID],
                output_dir=tmpdir,
                tabular_file_format="csv",
                export_geo_parquet=False,
                country="us",
            )

            result_df = pd.read_csv(csv_path)

            # Core roof age columns must be present (these exist for all properties)
            for col in [
                "roof_age_installation_date",
                "roof_age_trust_score",
                "roof_age_kind",
                "roof_age_map_browser_url",
                "roof_age_model_version",
                "roof_age_evidence_type",
                "roof_age_as_of_date",
            ]:
                assert col in result_df.columns, f"Missing expected column: {col}"

            # Every roof_age_ prefixed column must come from the whitelist
            for col in result_df.columns:
                if col.startswith("roof_age_"):
                    base = col[len("roof_age_") :]
                    assert (
                        base in ROOF_AGE_PREFIX_COLUMNS or col == "roof_age_years_as_of_date"
                    ), f"Unexpected roof_age_ column: {col} (base '{base}' not in ROOF_AGE_PREFIX_COLUMNS)"

            # Boolean fields converted to Y/N
            assert set(result_df["roof_age_relevant_permits"].dropna().unique()) <= {
                "Y",
                "N",
            }, "Boolean relevant_permits should be Y/N"
            assert set(result_df["roof_age_assessor_data"].dropna().unique()) <= {
                "Y",
                "N",
            }, "Boolean assessor_data should be Y/N"

            # CRITICAL: Feature API columns must NOT leak through with roof_age_ prefix
            # These are the exact columns that leaked in v4.4.0
            forbidden_columns = [
                "roof_age_link",
                "roof_age_system_version",
                "roof_age_survey_id",
                "roof_age_aoi_geometry",
                "roof_age_perspective",
                "roof_age_survey_date",
                "roof_age_mesh_date",
                "roof_age_confidence",
                "roof_age_fidelity",
                "roof_age_attributes",
                "roof_age_parent_id",
                "roof_age_belongs_to_parcel",
                "roof_age_multiparcel_feature",
                "roof_age_is_footprint",
                "roof_age_damage",
            ]
            for col in forbidden_columns:
                assert col not in result_df.columns, f"Feature API column leaked into roof instance export: {col}"

    def test_roof_export_includes_primary_child_columns(self, combined_features_gdf, features_2_gdf):
        """Roof age columns propagate as primary_child_roof_age_ on parent roofs."""
        ri_features = combined_features_gdf[combined_features_gdf["class_id"] == ROOF_INSTANCE_CLASS_ID]
        roof_features = combined_features_gdf[combined_features_gdf["class_id"] == ROOF_ID]

        if len(ri_features) == 0 or len(roof_features) == 0:
            pytest.skip("Need both roof instances and roofs in combined data")

        # Link roof instances to roofs (replicating exporter flow)
        ri_linked, roofs_linked = parcels.link_roof_instances_to_roofs(ri_features, roof_features)

        # Only test roofs that have a linked primary child
        roofs_with_link = roofs_linked[roofs_linked["primary_child_roof_age_feature_id"].notna()]
        if len(roofs_with_link) == 0:
            pytest.skip("No roofs linked to roof instances in test data")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, _ = export_feature_class(
                features_gdf=None,
                class_id=ROOF_ID,
                class_description="Roof",
                output_dir=tmpdir,
                tabular_file_format="csv",
                export_geo_parquet=False,
                country="us",
                class_features=roofs_with_link,
                roof_instance_features=ri_features,
            )

            result_df = pd.read_csv(csv_path)

            # Primary child roof age columns should be present
            for col in [
                "installation_date",
                "trust_score",
                "map_browser_url",
                "model_version",
            ]:
                prefixed = f"primary_child_roof_age_{col}"
                assert prefixed in result_df.columns, f"Missing expected column: {prefixed}"

            # Every primary_child_roof_age_ column must come from the whitelist
            for col in result_df.columns:
                if col.startswith("primary_child_roof_age_"):
                    base = col[len("primary_child_roof_age_") :]
                    allowed = (
                        base in ROOF_AGE_PREFIX_COLUMNS
                        or base in {"feature_id", "iou"}  # linkage columns
                        or col.startswith("primary_child_roof_age_years_")  # calculated
                    )
                    assert allowed, f"Unexpected primary_child_roof_age_ column: {col} (base '{base}')"

            # Boolean primary child columns should be Y/N
            if "primary_child_roof_age_relevant_permits" in result_df.columns:
                vals = result_df["primary_child_roof_age_relevant_permits"].dropna().unique()
                assert set(vals) <= {"Y", "N"}, "Boolean should be Y/N"

            # Feature API columns must NOT leak through with primary_child_roof_age_ prefix
            forbidden_columns = [
                "primary_child_roof_age_link",
                "primary_child_roof_age_system_version",
                "primary_child_roof_age_aoi_geometry",
                "primary_child_roof_age_survey_date",
                "primary_child_roof_age_perspective",
                "primary_child_roof_age_confidence",
                "primary_child_roof_age_fidelity",
            ]
            for col in forbidden_columns:
                assert col not in result_df.columns, f"Feature API column leaked into roof export: {col}"
