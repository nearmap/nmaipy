"""
Tests for the Damage Conflation API client (DamageConflationApi).

Verifies request payload construction, response parsing, pagination, caching, error
handling, and bulk querying. Uses the real Hurricane Milton response fixture — never
mock data — for parsing assertions.
"""

import copy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS
from nmaipy.damage_conflation_api import DamageConflationApi, DamageConflationAPIError

EVENT_ID = "2f510853-5d55-50f4-9102-2c02de08190e"  # Hurricane Milton


@pytest.fixture
def milton_response(data_directory: Path):
    fixture_path = data_directory / "test_damage_conflation_milton_response.json"
    with open(fixture_path, "r") as f:
        return json.load(f)


@pytest.fixture
def test_aoi():
    """~100m AOI near St Pete Beach, FL (inside the Milton event footprint)."""
    lon, lat = -82.754961, 27.742889
    half_lat = 50 / 111320
    half_lon = 50 / (111320 * 0.8853)
    return box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


@pytest.fixture
def damage_api(cache_directory):
    return DamageConflationApi(event_id=EVENT_ID, api_key="test_key", cache_dir=cache_directory, threads=2)


def _mock_session(mock_session_scope, json_payload, ok=True, status_code=200):
    mock_session = Mock()
    mock_response = Mock()
    mock_response.ok = ok
    mock_response.status_code = status_code
    mock_response.json.return_value = json_payload
    mock_session.post.return_value = mock_response
    mock_session._timeout = (120, 90)
    mock_session_scope.return_value.__enter__.return_value = mock_session
    return mock_session


# ----------------------------------------------------------------- construction
def test_init_requires_event_id():
    with pytest.raises(ValueError, match="event_id is required"):
        DamageConflationApi(event_id="", api_key="t")


def test_init_base_url():
    api = DamageConflationApi(event_id=EVENT_ID, api_key="t")
    assert "ai/damage/v2" in api.base_url
    assert api.base_url.endswith(f"/events/{EVENT_ID}/latest")


# ------------------------------------------------------------- payload building
def test_build_request_payload_aoi(damage_api, test_aoi):
    payload = damage_api._build_request_payload(aoi=test_aoi)
    assert payload["aoi"]["type"] == "Polygon"
    assert "address" not in payload
    assert "cursor" not in payload and "limit" not in payload

    paged = damage_api._build_request_payload(aoi=test_aoi, cursor="abc", limit=250)
    assert paged["cursor"] == "abc"
    assert paged["limit"] == 250


def test_build_request_payload_address(damage_api):
    address = {"streetAddress": "650 78th Ave", "city": "St Pete Beach", "state": "FL", "zip": "33706"}
    payload = damage_api._build_request_payload(address=address)
    assert payload["address"]["country"] == "US"  # client country, appended
    assert payload["address"]["streetAddress"] == "650 78th Ave"
    assert "aoi" not in payload


def test_build_request_payload_xor(damage_api, test_aoi):
    with pytest.raises(ValueError):
        damage_api._build_request_payload()  # neither
    with pytest.raises(ValueError):
        damage_api._build_request_payload(aoi=test_aoi, address={"streetAddress": "x"})  # both


# --------------------------------------------------------------- parse response
def test_parse_response(damage_api, milton_response):
    gdf = damage_api._parse_response(milton_response, aoi_id="aoi_1")

    assert len(gdf) == len(milton_response["features"]) == 14
    for col in ("feature_id", "damage_event_rating", "area_sqm", "event_uuid", "event_name"):
        assert col in gdf.columns
    # aoi_id tagged on every row
    assert (gdf[AOI_ID_COLUMN_NAME] == "aoi_1").all()
    # feature_id is the stable top-level GeoJSON id
    assert gdf["feature_id"].iloc[0] == milton_response["features"][0]["id"]
    # event metadata copied from the response top level
    assert gdf["event_uuid"].iloc[0] == milton_response["eventUuid"]
    assert gdf["event_name"].iloc[0] == milton_response["eventName"]
    # no nulls in the rating for these real features (all have an event block)
    assert gdf["damage_event_rating"].notna().all()


def test_parse_empty_response(damage_api):
    gdf = damage_api._parse_response({"type": "FeatureCollection", "features": []}, aoi_id="empty")
    assert len(gdf) == 0
    assert AOI_ID_COLUMN_NAME in gdf.columns
    assert "geometry" in gdf.columns


# -------------------------------------------------------------------- pagination
def test_pagination_merges_pages(damage_api, test_aoi, milton_response):
    """Two pages (page1 carries nextCursor) merge into one parsed result."""
    page1 = copy.deepcopy(milton_response)
    page2 = copy.deepcopy(milton_response)
    page1["features"] = milton_response["features"][:8]
    page1["nextCursor"] = "cursor-abc"
    page2["features"] = milton_response["features"][8:]
    page2.pop("nextCursor", None)

    with (
        patch.object(damage_api, "_session_scope") as scope,
        patch.object(damage_api, "_load_from_cache", return_value=None),
        patch.object(damage_api, "_save_to_cache"),
    ):
        mock_session = Mock()
        r1, r2 = Mock(), Mock()
        for r, payload in ((r1, page1), (r2, page2)):
            r.ok = True
            r.json.return_value = payload
        mock_session.post.side_effect = [r1, r2]
        mock_session._timeout = (120, 90)
        scope.return_value.__enter__.return_value = mock_session

        gdf = damage_api.get_damage_by_aoi(test_aoi, aoi_id="paged")

    assert mock_session.post.call_count == 2  # two pages fetched
    assert len(gdf) == len(milton_response["features"])  # all features merged
    # the cursor from page1 is forwarded in page2's request body
    assert mock_session.post.call_args_list[1].kwargs["json"]["cursor"] == "cursor-abc"


# ----------------------------------------------------------------------- caching
def test_caching(damage_api, test_aoi, milton_response):
    with patch.object(damage_api, "_session_scope") as scope:
        _mock_session(scope, milton_response)
        damage_api.get_damage_by_aoi(test_aoi, aoi_id="cache_test")
        first_calls = scope.return_value.__enter__.return_value.post.call_count
        assert first_calls == 1
        # second call should hit the on-disk cache, not the network
        damage_api.get_damage_by_aoi(test_aoi, aoi_id="cache_test")
        assert scope.return_value.__enter__.return_value.post.call_count == first_calls


def test_cache_path_includes_event_id(damage_api, test_aoi):
    cache_key = damage_api._build_cache_key(aoi=test_aoi)
    path = damage_api._get_cache_path(cache_key)
    assert "/damageconflation/" in path
    assert f"/{EVENT_ID}/" in path


def test_cache_separated_by_event_id(test_aoi, cache_directory):
    """Same AOI, different events -> different cache paths (no cross-event collision)."""
    a = DamageConflationApi(event_id="event-a", api_key="t", cache_dir=cache_directory)
    b = DamageConflationApi(event_id="event-b", api_key="t", cache_dir=cache_directory)
    assert a._get_cache_path(a._build_cache_key(aoi=test_aoi)) != b._get_cache_path(b._build_cache_key(aoi=test_aoi))


# ------------------------------------------------------------------ error paths
def test_http_error_raises(damage_api, test_aoi):
    with (
        patch.object(damage_api, "_session_scope") as scope,
        patch.object(damage_api, "_load_from_cache", return_value=None),
    ):
        _mock_session(scope, {"error": "nope"}, ok=False, status_code=403)
        with pytest.raises(DamageConflationAPIError):
            damage_api.get_damage_by_aoi(test_aoi, aoi_id="err")


def test_error_body_on_200_raises(damage_api, test_aoi):
    """A 200 response carrying an {error, code} body is still surfaced as an error."""
    with (
        patch.object(damage_api, "_session_scope") as scope,
        patch.object(damage_api, "_load_from_cache", return_value=None),
    ):
        _mock_session(scope, {"error": "Out of coverage", "code": "NO_ACCESS"}, ok=True, status_code=200)
        with pytest.raises(DamageConflationAPIError, match="Out of coverage"):
            damage_api.get_damage_by_aoi(test_aoi, aoi_id="err200")


# -------------------------------------------------------------------------- bulk
def test_get_damage_bulk(damage_api, test_aoi, milton_response):
    """Bulk fans out per AOI; one AOI errors, the other succeeds."""
    parsed = damage_api._parse_response(milton_response, aoi_id="good")
    aoi_gdf = gpd.GeoDataFrame(
        geometry=[test_aoi, test_aoi],
        crs=API_CRS,
        index=pd.Index(["good", "bad"], name=AOI_ID_COLUMN_NAME),
    )

    def fake_get(aoi, aoi_id, *args, **kwargs):
        if aoi_id == "bad":
            raise DamageConflationAPIError(None, "url", message="boom")
        return parsed

    with patch.object(damage_api, "get_damage_by_aoi", side_effect=fake_get):
        features_gdf, metadata_df, errors_df = damage_api.get_damage_bulk(aoi_gdf)

    assert len(features_gdf) == len(parsed)
    assert "good" in metadata_df.index
    assert "bad" in errors_df.index
    assert metadata_df.loc["good", "event_uuid"] == milton_response["eventUuid"]


def test_get_damage_bulk_requires_aoi_id_index(damage_api, test_aoi):
    bad = gpd.GeoDataFrame(geometry=[test_aoi], crs=API_CRS)  # default RangeIndex
    with pytest.raises(ValueError, match=AOI_ID_COLUMN_NAME):
        damage_api.get_damage_bulk(bad)
