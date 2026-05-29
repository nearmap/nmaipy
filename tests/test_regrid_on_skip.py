"""Tests for reactive gridding when the Feature API marks any include as skipped.

The Feature API returns ``{"skipped": true}`` for include sections when a
parcel exceeds the per-parcel CPU budget. ``FeatureApi.regrid_on_skip``
(default True) detects this and reactively re-issues the AOI as a gridded
query so the sub-AOIs come back as mini-parcels with populated scores.
"""

import geopandas as gpd
from shapely.geometry import Polygon

from nmaipy.feature_api import FeatureApi, _response_has_include_skips


SMALL_AOI = Polygon(
    [
        [144.9, -37.8],
        [144.9 + 0.0045, -37.8],
        [144.9 + 0.0045, -37.8 + 0.0045],
        [144.9, -37.8 + 0.0045],
        [144.9, -37.8],
    ]
)


def _features_gdf_with_skip(col):
    return gpd.GeoDataFrame({col: [True], "geometry": [SMALL_AOI]}, geometry="geometry")


class TestResponseHasIncludeSkips:
    def test_no_skips_returns_false(self):
        features_gdf = gpd.GeoDataFrame(
            {"roof_spotlight_index_skipped": [False, False], "geometry": [SMALL_AOI, SMALL_AOI]},
            geometry="geometry",
        )
        any_skipped, skipped = _response_has_include_skips(features_gdf)
        assert any_skipped is False
        assert skipped == []

    def test_per_roof_skip_detected(self):
        features_gdf = _features_gdf_with_skip("hurricane_score_skipped")
        any_skipped, skipped = _response_has_include_skips(features_gdf)
        assert any_skipped is True
        assert skipped == ["hurricane_score_skipped"]

    def test_none_input_returns_false(self):
        any_skipped, skipped = _response_has_include_skips(None)
        assert any_skipped is False
        assert skipped == []

    def test_multiple_skip_columns_all_reported(self):
        features_gdf = gpd.GeoDataFrame(
            {
                "roof_spotlight_index_skipped": [True],
                "defensible_space_skipped": [True],
                "wind_score_skipped": [False],
                "geometry": [SMALL_AOI],
            },
            geometry="geometry",
        )
        any_skipped, skipped = _response_has_include_skips(features_gdf)
        assert any_skipped is True
        assert set(skipped) == {"roof_spotlight_index_skipped", "defensible_space_skipped"}

    def test_nan_skip_treated_as_false(self):
        features_gdf = gpd.GeoDataFrame(
            {"roof_spotlight_index_skipped": [None, None], "geometry": [SMALL_AOI, SMALL_AOI]},
            geometry="geometry",
        )
        any_skipped, _ = _response_has_include_skips(features_gdf)
        assert any_skipped is False


def _install_stubs(api, monkeypatch, *, skipped: bool):
    """Stub `get_features`, `payload_gdf` and `_attempt_gridding` on `api`.

    The stubbed payload reports `roof_spotlight_index_skipped = skipped`.
    Returns a dict that the test can inspect; `dict["count"]` is the number
    of times `_attempt_gridding` was invoked.
    """
    gridding_called = {"count": 0}

    def stub_get_features(*args, **kwargs):
        return {}

    def stub_payload_gdf(payload, aoi_id, parcel_mode):
        features_gdf = gpd.GeoDataFrame(
            {"roof_spotlight_index_skipped": [skipped], "geometry": [SMALL_AOI]},
            geometry="geometry",
        )
        return features_gdf, {"aoi_id": aoi_id}

    def stub_attempt_gridding(*args, **kwargs):
        gridding_called["count"] += 1
        return (
            gpd.GeoDataFrame({"geometry": [SMALL_AOI]}, geometry="geometry"),
            {"aoi_id": kwargs.get("aoi_id", "x"), "system_version": "v"},
            None,
            None,
        )

    monkeypatch.setattr(api, "get_features", stub_get_features)
    monkeypatch.setattr(FeatureApi, "payload_gdf", staticmethod(stub_payload_gdf))
    monkeypatch.setattr(api, "_attempt_gridding", stub_attempt_gridding)
    return gridding_called


class TestRegridOnSkip:
    """End-to-end behaviour of FeatureApi.get_features_gdf when skips arrive."""

    def _build_api(self, tmp_path, regrid_on_skip=True):
        return FeatureApi(api_key="test-key", cache_dir=tmp_path, regrid_on_skip=regrid_on_skip)

    def test_default_regrid_on_skip_is_true(self, tmp_path):
        api = FeatureApi(api_key="test-key", cache_dir=tmp_path)
        assert api.regrid_on_skip is True

    def test_skip_triggers_gridding(self, tmp_path, monkeypatch):
        api = self._build_api(tmp_path, regrid_on_skip=True)
        gridding_called = _install_stubs(api, monkeypatch, skipped=True)

        api.get_features_gdf(geometry=SMALL_AOI, region="au", aoi_id="skip-aoi")
        assert gridding_called["count"] == 1

    def test_skip_does_not_trigger_when_flag_off(self, tmp_path, monkeypatch):
        api = self._build_api(tmp_path, regrid_on_skip=False)
        gridding_called = _install_stubs(api, monkeypatch, skipped=True)

        api.get_features_gdf(geometry=SMALL_AOI, region="au", aoi_id="skip-aoi")
        assert gridding_called["count"] == 0

    def test_no_skips_does_not_trigger_gridding(self, tmp_path, monkeypatch):
        api = self._build_api(tmp_path, regrid_on_skip=True)
        gridding_called = _install_stubs(api, monkeypatch, skipped=False)

        api.get_features_gdf(geometry=SMALL_AOI, region="au", aoi_id="ok-aoi")
        assert gridding_called["count"] == 0

    def test_gridding_mode_suppresses_skip_trigger(self, tmp_path, monkeypatch):
        api = self._build_api(tmp_path, regrid_on_skip=True)
        gridding_called = _install_stubs(api, monkeypatch, skipped=True)

        api.get_features_gdf(
            geometry=SMALL_AOI,
            region="au",
            aoi_id="sub-aoi",
            in_gridding_mode=True,
        )
        assert gridding_called["count"] == 0
