"""Tests for nmaipy.coverage_utils (standard coverage path).

Network is mocked at get_payload / get_surveys_from_point so URL building,
response→dataframe shaping, the global-vs-per-row window logic, tolerant column
projection, and timeout plumbing are all exercised without HTTP.
"""

from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from nmaipy import coverage_utils as cu

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


def test_get_coverage_from_points_standard_tolerant(tmp_path):
    # id_col="aoi_id" mirrors real callers; the default "id" collides with the
    # survey's own "id" field (a pre-existing footgun, out of scope here).
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
