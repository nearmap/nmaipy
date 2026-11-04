"""Tests for reactive gridding when the Feature API marks any include as skipped.

The Feature API returns ``{"skipped": true}`` for include sections when a
parcel exceeds the per-parcel CPU budget. ``FeatureApi.regrid_on_skip``
(default True) detects this and reactively re-issues the AOI as a gridded
query so the sub-AOIs come back as mini-parcels with populated scores.
"""

import geopandas as gpd
from shapely.geometry import Polygon, mapping

from nmaipy.constants import ROOF_ID
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


def _roof(feature_id, **includes):
    """A raw Feature API roof feature, with includes keyed by their camelCase API names."""
    return {
        "id": feature_id,
        "classId": ROOF_ID,
        "description": "Roof",
        "confidence": 0.9,
        "parentId": None,
        "geometry": mapping(SMALL_AOI),
        "areaSqm": 100.0,
        "clippedAreaSqm": 100.0,
        "unclippedAreaSqm": 100.0,
        "areaSqft": 1076.4,
        "clippedAreaSqft": 1076.4,
        "unclippedAreaSqft": 1076.4,
        "attributes": [],
        "surveyDate": "2026-01-01",
        "meshDate": "",
        **includes,
    }


def _payload(*features):
    return {
        "systemVersion": "gen6-test",
        "link": "https://apps.nearmap.com/maps/#/@-37.8,144.9,20.00z,0d/V/20260101",
        "surveyDate": "2026-01-01",
        "surveyId": "survey",
        "resourceId": "resource",
        "perspective": "Vert",
        "postcat": False,
        "features": list(features),
    }


def _skip_check(*features):
    """Run the skip check on payload_gdf output, the form get_features_gdf passes it."""
    features_gdf, _ = FeatureApi.payload_gdf(_payload(*features), aoi_id="aoi", parcel_mode=True)
    return _response_has_include_skips(features_gdf)


class TestResponseHasIncludeSkips:
    def test_no_skips_returns_false(self):
        any_skipped, skipped = _skip_check(
            _roof("a", roofSpotlightIndex={"value": 50, "confidence": 0.9}), _roof("b", hurricaneScore={})
        )
        assert any_skipped is False
        assert skipped == []

    def test_per_roof_skip_detected(self):
        any_skipped, skipped = _skip_check(_roof("a", hurricaneScore={"skipped": True}), _roof("b"))
        assert any_skipped is True
        assert skipped == ["hurricane_score_skipped"]

    def test_none_input_returns_false(self):
        any_skipped, skipped = _response_has_include_skips(None)
        assert any_skipped is False
        assert skipped == []

    def test_multiple_skips_all_reported(self):
        any_skipped, skipped = _skip_check(
            _roof("a", roofSpotlightIndex={"skipped": True}, windScore={"vulnerabilityScore": 3}),
            _roof("b", defensibleSpace={"skipped": True}),
        )
        assert any_skipped is True
        assert set(skipped) == {"roof_spotlight_index_skipped", "defensible_space_skipped"}

    def test_roofs_without_the_include_are_not_skips(self):
        """Roofs lacking an include another roof carries come back NaN in that column, which is not a skip."""
        any_skipped, _ = _skip_check(_roof("a", roofSpotlightIndex={"value": 50}), _roof("b"))
        assert any_skipped is False

    def test_skipped_false_is_not_a_skip(self):
        any_skipped, _ = _skip_check(_roof("a", roofSpotlightIndex={"skipped": False, "value": 50}))
        assert any_skipped is False


def _install_stubs(api, monkeypatch, *, skipped: bool):
    """Stub `get_features` and `_attempt_gridding` on `api`; the real `payload_gdf` parses the response.

    The response's roof carries `roofSpotlightIndex = {"skipped": true}` when `skipped`, else a score.
    Returns a dict that the test can inspect; `dict["count"]` is the number
    of times `_attempt_gridding` was invoked.
    """
    gridding_called = {"count": 0}
    include = {"skipped": True} if skipped else {"value": 50, "confidence": 0.9}

    def stub_get_features(*args, **kwargs):
        return _payload(_roof("roof", roofSpotlightIndex=include))

    def stub_attempt_gridding(*args, **kwargs):
        gridding_called["count"] += 1
        return (
            gpd.GeoDataFrame({"geometry": [SMALL_AOI]}, geometry="geometry"),
            {"aoi_id": kwargs.get("aoi_id", "x"), "system_version": "v"},
            None,
            None,
        )

    monkeypatch.setattr(api, "get_features", stub_get_features)
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
