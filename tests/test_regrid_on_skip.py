"""Tests for reactive gridding when the Feature API marks any include as skipped.

The Feature API returns ``{"skipped": true}`` for include sections (RSI, peril
scores, defensible space) when a parcel exceeds the per-parcel CPU budget —
currently ~100 buildings. ``FeatureApi.regrid_on_skip`` (default True) detects
this and reactively re-issues the AOI as a gridded query.
"""

import geopandas as gpd
import pandas as pd
import pytest
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


def _empty_features_gdf():
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry")


def _features_gdf_with_skip(col):
    return gpd.GeoDataFrame({col: [True], "geometry": [SMALL_AOI]}, geometry="geometry")


class TestResponseHasIncludeSkips:
    def test_no_skips_returns_false(self):
        features_gdf = gpd.GeoDataFrame(
            {"roof_spotlight_index_skipped": [False, False], "geometry": [SMALL_AOI, SMALL_AOI]},
            geometry="geometry",
        )
        metadata = {"aggregate": None}
        any_skipped, skipped = _response_has_include_skips(features_gdf, metadata)
        assert any_skipped is False
        assert skipped == []

    def test_per_roof_skip_detected(self):
        features_gdf = _features_gdf_with_skip("hurricane_score_skipped")
        any_skipped, skipped = _response_has_include_skips(features_gdf, None)
        assert any_skipped is True
        assert skipped == ["hurricane_score_skipped"]

    def test_aggregate_ds_skip_detected(self):
        metadata = {"aggregate": {"defensibleSpace": {"skipped": True}}}
        any_skipped, skipped = _response_has_include_skips(_empty_features_gdf(), metadata)
        assert any_skipped is True
        assert skipped == ["aggregate_defensible_space"]

    def test_aggregate_ds_not_skipped(self):
        metadata = {"aggregate": {"defensibleSpace": {"skipped": False, "zones": []}}}
        any_skipped, skipped = _response_has_include_skips(_empty_features_gdf(), metadata)
        assert any_skipped is False

    def test_missing_aggregate_returns_false(self):
        any_skipped, skipped = _response_has_include_skips(_empty_features_gdf(), {})
        assert any_skipped is False

    def test_none_inputs_return_false(self):
        any_skipped, skipped = _response_has_include_skips(None, None)
        assert any_skipped is False

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
        any_skipped, skipped = _response_has_include_skips(features_gdf, None)
        assert any_skipped is True
        assert set(skipped) == {"roof_spotlight_index_skipped", "defensible_space_skipped"}

    def test_nan_skip_treated_as_false(self):
        features_gdf = gpd.GeoDataFrame(
            {"roof_spotlight_index_skipped": [None, None], "geometry": [SMALL_AOI, SMALL_AOI]},
            geometry="geometry",
        )
        any_skipped, _ = _response_has_include_skips(features_gdf, None)
        assert any_skipped is False


class TestRegridOnSkip:
    """End-to-end behaviour of FeatureApi.get_features_gdf when skips arrive."""

    def _build_api(self, tmp_path, regrid_on_skip=True):
        return FeatureApi(api_key="test-key", cache_dir=tmp_path, regrid_on_skip=regrid_on_skip)

    def test_default_regrid_on_skip_is_true(self, tmp_path):
        api = FeatureApi(api_key="test-key", cache_dir=tmp_path)
        assert api.regrid_on_skip is True

    def test_skip_triggers_gridding(self, tmp_path, monkeypatch):
        api = self._build_api(tmp_path, regrid_on_skip=True)

        gridding_called = {"count": 0}

        def mock_get_features(*args, **kwargs):
            return {}

        def mock_payload_gdf(payload, aoi_id, parcel_mode):
            features_gdf = gpd.GeoDataFrame(
                {"roof_spotlight_index_skipped": [True], "geometry": [SMALL_AOI]},
                geometry="geometry",
            )
            metadata = {"aoi_id": aoi_id, "aggregate": None}
            return features_gdf, metadata

        def mock_attempt_gridding(*args, **kwargs):
            gridding_called["count"] += 1
            return (
                gpd.GeoDataFrame({"geometry": [SMALL_AOI]}, geometry="geometry"),
                {"aoi_id": kwargs.get("aoi_id", "x"), "system_version": "v"},
                None,
                None,
            )

        monkeypatch.setattr(api, "get_features", mock_get_features)
        monkeypatch.setattr(FeatureApi, "payload_gdf", staticmethod(mock_payload_gdf))
        monkeypatch.setattr(api, "_attempt_gridding", mock_attempt_gridding)

        api.get_features_gdf(geometry=SMALL_AOI, region="au", aoi_id="skip-aoi")
        assert gridding_called["count"] == 1

    def test_skip_does_not_trigger_when_flag_off(self, tmp_path, monkeypatch):
        api = self._build_api(tmp_path, regrid_on_skip=False)

        gridding_called = {"count": 0}

        def mock_get_features(*args, **kwargs):
            return {}

        def mock_payload_gdf(payload, aoi_id, parcel_mode):
            features_gdf = gpd.GeoDataFrame(
                {"roof_spotlight_index_skipped": [True], "geometry": [SMALL_AOI]},
                geometry="geometry",
            )
            return features_gdf, {"aoi_id": aoi_id, "aggregate": None}

        def mock_attempt_gridding(*args, **kwargs):
            gridding_called["count"] += 1
            return (None, None, None, None)

        monkeypatch.setattr(api, "get_features", mock_get_features)
        monkeypatch.setattr(FeatureApi, "payload_gdf", staticmethod(mock_payload_gdf))
        monkeypatch.setattr(api, "_attempt_gridding", mock_attempt_gridding)

        api.get_features_gdf(geometry=SMALL_AOI, region="au", aoi_id="skip-aoi")
        assert gridding_called["count"] == 0

    def test_no_skips_does_not_trigger_gridding(self, tmp_path, monkeypatch):
        api = self._build_api(tmp_path, regrid_on_skip=True)

        gridding_called = {"count": 0}

        def mock_get_features(*args, **kwargs):
            return {}

        def mock_payload_gdf(payload, aoi_id, parcel_mode):
            features_gdf = gpd.GeoDataFrame(
                {"roof_spotlight_index_skipped": [False], "geometry": [SMALL_AOI]},
                geometry="geometry",
            )
            return features_gdf, {"aoi_id": aoi_id, "aggregate": None}

        def mock_attempt_gridding(*args, **kwargs):
            gridding_called["count"] += 1
            return (None, None, None, None)

        monkeypatch.setattr(api, "get_features", mock_get_features)
        monkeypatch.setattr(FeatureApi, "payload_gdf", staticmethod(mock_payload_gdf))
        monkeypatch.setattr(api, "_attempt_gridding", mock_attempt_gridding)

        api.get_features_gdf(geometry=SMALL_AOI, region="au", aoi_id="ok-aoi")
        assert gridding_called["count"] == 0

    def test_gridding_mode_suppresses_skip_trigger(self, tmp_path, monkeypatch):
        api = self._build_api(tmp_path, regrid_on_skip=True)

        gridding_called = {"count": 0}

        def mock_get_features(*args, **kwargs):
            return {}

        def mock_payload_gdf(payload, aoi_id, parcel_mode):
            features_gdf = gpd.GeoDataFrame(
                {"roof_spotlight_index_skipped": [True], "geometry": [SMALL_AOI]},
                geometry="geometry",
            )
            return features_gdf, {"aoi_id": aoi_id, "aggregate": None}

        def mock_attempt_gridding(*args, **kwargs):
            gridding_called["count"] += 1
            return (None, None, None, None)

        monkeypatch.setattr(api, "get_features", mock_get_features)
        monkeypatch.setattr(FeatureApi, "payload_gdf", staticmethod(mock_payload_gdf))
        monkeypatch.setattr(api, "_attempt_gridding", mock_attempt_gridding)

        api.get_features_gdf(
            geometry=SMALL_AOI,
            region="au",
            aoi_id="sub-aoi",
            in_gridding_mode=True,
        )
        assert gridding_called["count"] == 0

    def test_aggregate_ds_skip_synthesized_after_grid(self, tmp_path, monkeypatch):
        """When DS was requested and skip triggered gridding, synthesize aggregate skip flag.

        Aggregate DS is parcel-mode-only and the gridded path disables parcel mode, so
        the API never returns aggregate values for gridded sub-queries. We synthesize
        ``metadata["aggregate"]["defensibleSpace"]["skipped"] = True`` so
        parcels.parcel_rollup populates aggregate_defensible_space_skipped=True.
        """
        api = self._build_api(tmp_path, regrid_on_skip=True)

        def mock_get_features(*args, **kwargs):
            return {}

        def mock_payload_gdf(payload, aoi_id, parcel_mode):
            features_gdf = gpd.GeoDataFrame(
                {"defensible_space_skipped": [True], "geometry": [SMALL_AOI]},
                geometry="geometry",
            )
            return features_gdf, {"aoi_id": aoi_id, "aggregate": {"defensibleSpace": {"skipped": True}}}

        def mock_attempt_gridding(*args, **kwargs):
            metadata = {"aoi_id": kwargs.get("aoi_id", "x"), "system_version": "v"}
            return (
                gpd.GeoDataFrame({"geometry": [SMALL_AOI]}, geometry="geometry"),
                metadata,
                None,
                None,
            )

        monkeypatch.setattr(api, "get_features", mock_get_features)
        monkeypatch.setattr(FeatureApi, "payload_gdf", staticmethod(mock_payload_gdf))
        monkeypatch.setattr(api, "_attempt_gridding", mock_attempt_gridding)

        _, metadata, _, _ = api.get_features_gdf(
            geometry=SMALL_AOI,
            region="au",
            include=["defensibleSpace"],
            aoi_id="ds-aoi",
        )
        assert metadata is not None
        assert metadata["aggregate"]["defensibleSpace"]["skipped"] is True

    def test_aggregate_ds_skip_not_synthesized_when_ds_not_requested(self, tmp_path, monkeypatch):
        """If the caller never asked for defensibleSpace, don't fabricate an aggregate dict."""
        api = self._build_api(tmp_path, regrid_on_skip=True)

        def mock_get_features(*args, **kwargs):
            return {}

        def mock_payload_gdf(payload, aoi_id, parcel_mode):
            features_gdf = gpd.GeoDataFrame(
                {"roof_spotlight_index_skipped": [True], "geometry": [SMALL_AOI]},
                geometry="geometry",
            )
            return features_gdf, {"aoi_id": aoi_id, "aggregate": None}

        def mock_attempt_gridding(*args, **kwargs):
            metadata = {"aoi_id": kwargs.get("aoi_id", "x"), "system_version": "v"}
            return (
                gpd.GeoDataFrame({"geometry": [SMALL_AOI]}, geometry="geometry"),
                metadata,
                None,
                None,
            )

        monkeypatch.setattr(api, "get_features", mock_get_features)
        monkeypatch.setattr(FeatureApi, "payload_gdf", staticmethod(mock_payload_gdf))
        monkeypatch.setattr(api, "_attempt_gridding", mock_attempt_gridding)

        _, metadata, _, _ = api.get_features_gdf(
            geometry=SMALL_AOI,
            region="au",
            include=["roofSpotlightIndex"],
            aoi_id="rsi-aoi",
        )
        assert metadata is not None
        assert "aggregate" not in metadata or metadata["aggregate"] is None or "defensibleSpace" not in metadata.get("aggregate", {})
