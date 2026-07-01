"""Tests for nmaipy.coverage_utils (standard coverage path).

Network is mocked at get_payload / get_surveys_from_point so URL building,
response→dataframe shaping, the global-vs-per-row window logic, tolerant column
projection, and timeout plumbing are all exercised without HTTP.
"""

import copy
import json
import os
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.ops import unary_union

from nmaipy import coverage_utils as cu

MILTON_EVENT_ID = "2f510853-5d55-50f4-9102-2c02de08190e"

# A standard /coverage/v2/point response: s1 has tiles+ai+3d+tags; s2 has tiles
# only (no aifeatures, no 3d, no tags key) — exercises tolerant shaping.
_STD_RESPONSE = {
    "surveys": [
        {
            "id": "s1",
            "captureDate": "2020-06-01",
            "resources": {"tiles": [{"id": "t1"}], "aifeatures": [{"id": "a1"}], "3d": [{"id": "d1"}]},
            "tags": [],
        },
        {
            "id": "s2",
            "captureDate": "2019-06-01",
            "resources": {"tiles": [{"id": "t2"}]},
        },
    ]
}


def test_std_coverage_response_to_dataframe_adds_resource_flags():
    df = cu.std_coverage_response_to_dataframe(_STD_RESPONSE)
    assert list(df["id"]) == ["s1", "s2"]
    assert df.loc[0, "aifeatures"] and not df.loc[1, "aifeatures"]
    assert df.loc[0, "3d"] and not df.loc[1, "3d"]
    assert df["tiles"].all()


def test_std_coverage_empty():
    assert len(cu.std_coverage_response_to_dataframe({"surveys": []})) == 0


def test_survey_resource_id_from_standard_coverage():
    assert cu.get_survey_resource_id_from_standard_coverage({"aifeatures": [{"id": "a1"}, {"id": "a2"}]}) == "a2"
    assert cu.get_survey_resource_id_from_standard_coverage({"tiles": [{"id": "t1"}]}) is None


def test_get_payload_passes_timeout():
    with patch.object(cu.s, "get") as mock_get:
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {"surveys": []}
        cu.get_payload("https://example/x", timeout=(3, 7))
    assert mock_get.call_args.kwargs["timeout"] == (3, 7)


def test_get_surveys_from_point_standard():
    with patch.object(cu, "get_payload", return_value=_STD_RESPONSE):
        df, resp = cu.get_surveys_from_point(151.0, -33.0, "2019-01-01", "2021-01-01", "KEY", cu.STANDARD_COVERAGE)
    assert resp is _STD_RESPONSE
    assert set(df["id"]) == {"s1", "s2"}


def test_threaded_uses_global_window_when_no_columns():
    """A points frame with NO since/until columns must not KeyError — the global
    since/until is applied to every row."""
    df = pd.DataFrame({"longitude": [151.0, 151.1], "latitude": [-33.0, -33.1]})
    seen = []

    def _record(lon, lat, since, until, *a, **k):
        seen.append((since, until))
        return pd.DataFrame(), None

    with patch.object(cu, "get_surveys_from_point", side_effect=_record):
        cu.threaded_get_coverage_from_point_results(df, apikey="KEY", since="2020-01-01", until="2020-12-31", threads=2)
    assert seen == [("2020-01-01", "2020-12-31"), ("2020-01-01", "2020-12-31")]


def test_threaded_per_row_columns_override_global():
    df = pd.DataFrame(
        {
            "longitude": [151.0],
            "latitude": [-33.0],
            "since": ["2018-01-01"],
            "until": ["2018-12-31"],
        }
    )
    seen = []

    def _record(lon, lat, since, until, *a, **k):
        seen.append((since, until))
        return pd.DataFrame(), None

    with patch.object(cu, "get_surveys_from_point", side_effect=_record):
        cu.threaded_get_coverage_from_point_results(df, apikey="KEY", since="2099-01-01", until="2099-12-31", threads=1)
    assert seen == [("2018-01-01", "2018-12-31")]  # per-row wins


def test_get_coverage_from_points_rejects_missing_id_col(tmp_path):
    """id_col is required — and explicitly must not be 'id' (collides with the
    per-survey 'id' field). Both omission and the 'id' literal must fail loud."""
    points = gpd.GeoDataFrame(
        {"aoi_id": ["u1"], "longitude": [151.0], "latitude": [-33.0]},
        geometry=[Point(151.0, -33.0)],
        crs="EPSG:4326",
    )
    import pytest

    with pytest.raises(ValueError, match="id_col is required"):
        cu.get_coverage_from_points(points, api_key="KEY", coverage_chunk_cache_dir=str(tmp_path / "cov1"))
    with pytest.raises(ValueError, match="collides"):
        cu.get_coverage_from_points(points, api_key="KEY", id_col="id", coverage_chunk_cache_dir=str(tmp_path / "cov2"))


def test_get_coverage_from_points_standard_tolerant(tmp_path):
    # id_col="aoi_id" mirrors real callers (must NOT be "id" — that collides with
    # the survey's own "id" field and is rejected at the boundary above).
    points = gpd.GeoDataFrame(
        {"aoi_id": ["u1", "u2"], "longitude": [151.0, 151.1], "latitude": [-33.0, -33.1]},
        geometry=[Point(151.0, -33.0), Point(151.1, -33.1)],
        crs="EPSG:4326",
    )
    with patch.object(cu, "get_payload", return_value=_STD_RESPONSE):
        out = cu.get_coverage_from_points(
            points,
            api_key="KEY",
            coverage_type=cu.STANDARD_COVERAGE,
            since="2019-01-01",
            until="2021-01-01",
            id_col="aoi_id",
            threads=2,
            coverage_chunk_cache_dir=str(tmp_path / "cov"),
        )
    # one row per (unit × survey): 2 units × 2 surveys
    assert len(out) == 4
    assert set(out.index.unique()) == {"u1", "u2"}
    assert "survey_resource_id" in out.columns and "tags" in out.columns  # tolerant projection kept tags
    assert pd.api.types.is_datetime64_any_dtype(out["captureDate"])
    # s1 has an aifeatures resource id; s2 does not
    s1 = out[out["survey_id"] == "s1"].iloc[0]
    assert s1["survey_resource_id"] == "a1"


# ---------------------------------------------------------------------------
# Post-catastrophe event discovery (mocked at get_payload; fixtures are real
# captured Coverage API responses from tests/data/).
# ---------------------------------------------------------------------------
@pytest.fixture
def coverage_point_response(data_directory):
    with open(data_directory / "test_coverage_point_milton.json") as f:
        return json.load(f)


@pytest.fixture
def coverage_boundary_response(data_directory):
    with open(data_directory / "test_coverage_boundary_milton.json") as f:
        return json.load(f)


def _tagged_union(response, event_id):
    """Independent reference union of the boundaries of surveys tagged with event_id."""
    polys = [
        shape(tile["boundary"])
        for s in response["surveys"]
        if (s.get("tags") or {}).get("postCatEventId") == event_id
        for tile in s["resources"]["tiles"]
        if tile.get("boundary")
    ]
    return unary_union(polys)


def test_latest_event_id_at_point(coverage_point_response):
    with patch.object(cu, "get_payload", return_value=coverage_point_response):
        assert cu.latest_event_id_at_point(27.742889, -82.754961, apikey="k") == MILTON_EVENT_ID


def test_latest_event_id_at_point_none_raises():
    with patch.object(cu, "get_payload", return_value={"surveys": []}):
        with pytest.raises(ValueError, match="No post-catastrophe event"):
            cu.latest_event_id_at_point(0.0, 0.0, apikey="k")


def test_latest_event_id_at_point_picks_newest(coverage_point_response):
    """With >1 distinct event at the point, the one with the newest postCatEventDate wins."""
    resp = copy.deepcopy(coverage_point_response)
    older = copy.deepcopy(resp["surveys"][0])
    older["tags"] = dict(older["tags"])
    older["tags"]["postCatEventId"] = "older-event-id"
    older["tags"]["postCatEventDate"] = "2019-01-01"  # older than the real Milton date
    resp["surveys"].append(older)
    with patch.object(cu, "get_payload", return_value=resp):
        assert cu.latest_event_id_at_point(27.742889, -82.754961, apikey="k") == MILTON_EVENT_ID


def test_event_boundary_unions_only_tagged_surveys(coverage_boundary_response):
    expected = _tagged_union(coverage_boundary_response, MILTON_EVENT_ID)
    with_untagged = unary_union(
        [
            shape(tile["boundary"])
            for s in coverage_boundary_response["surveys"]
            for tile in s["resources"]["tiles"]
            if tile.get("boundary")
        ]
    )
    with patch.object(cu, "get_payload", return_value=coverage_boundary_response):
        boundary = cu.event_boundary(MILTON_EVENT_ID, since="2024-10-08", until="2024-10-25", apikey="k")
    assert isinstance(boundary, (Polygon, MultiPolygon)) and boundary.is_valid and boundary.area > 0
    assert boundary.equals(expected)  # only tagged surveys contributed
    # the fixture's non-event survey adds area -> proves the defensive client-side tag filter excludes it
    assert with_untagged.area > boundary.area


def test_event_boundary_unknown_event_raises(coverage_boundary_response):
    with patch.object(cu, "get_payload", return_value=coverage_boundary_response):
        with pytest.raises(ValueError, match="No surveys found for event"):
            cu.event_boundary("no-such-event-id", apikey="k")


def test_discover_event_returns_id_and_boundary(coverage_point_response, coverage_boundary_response):
    def fake_get_payload(url, timeout=cu.DEFAULT_TIMEOUT):
        if "/point/" in url:
            return coverage_point_response
        if "/surveys" in url:  # event_boundary uses the /surveys?include= tag filter
            return coverage_boundary_response
        raise AssertionError(f"unexpected URL: {url}")

    with patch.object(cu, "get_payload", side_effect=fake_get_payload):
        event_id, boundary = cu.discover_event(27.742889, -82.754961, apikey="k")
    assert event_id == MILTON_EVENT_ID
    assert isinstance(boundary, (Polygon, MultiPolygon)) and boundary.area > 0


@pytest.mark.live_api
def test_discover_event_live():
    """Live: a point in the Hurricane Milton footprint resolves to the Milton event id and a
    non-empty boundary. Requires API_KEY."""
    if not os.environ.get("API_KEY"):
        pytest.skip("API_KEY not set")
    event_id, boundary = cu.discover_event(lat=27.742889, lon=-82.754961)
    assert event_id == MILTON_EVENT_ID
    assert isinstance(boundary, (Polygon, MultiPolygon)) and boundary.area > 0


def test_event_boundary_paginates_offset(coverage_boundary_response):
    """event_boundary pages through offset/limit until `total` is reached, so a large event
    isn't truncated at one page. Two mocked pages must both contribute to the union."""
    tagged = [
        s
        for s in coverage_boundary_response["surveys"]
        if (s.get("tags") or {}).get("postCatEventId") == MILTON_EVENT_ID
    ]
    assert len(tagged) >= 2, "fixture needs >=2 tagged surveys to split across pages"
    total = len(tagged)
    pages = [
        {"surveys": tagged[:1], "total": total, "limit": 1, "offset": 0},
        {"surveys": tagged[1:], "total": total, "limit": 1, "offset": 1},
    ]
    calls = {"n": 0}

    def fake(url, timeout=cu.DEFAULT_TIMEOUT):
        i = calls["n"]
        calls["n"] += 1
        return pages[i] if i < len(pages) else {"surveys": [], "total": total}

    expected = _tagged_union(coverage_boundary_response, MILTON_EVENT_ID)
    with patch.object(cu, "get_payload", side_effect=fake):
        boundary = cu.event_boundary(MILTON_EVENT_ID, apikey="k")
    assert calls["n"] >= 2, "should have paged (>=2 requests)"
    assert boundary.equals(expected)  # boundaries from BOTH pages were unioned
