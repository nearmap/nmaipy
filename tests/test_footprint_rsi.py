"""Tests for resolving footprint RSI from building lifecycle."""

import json

import geopandas as gpd
import pandas as pd
import pytest

from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    BUILDING_ID,
    BUILDING_LIFECYCLE_ID,
    BUILDING_NEW_ID,
    LAWN_GRASS_ID,
    ROOF_ID,
    VEG_MEDHIGH_ID,
)
from nmaipy.parcels import (
    _build_parent_lookup_by_aoi,
    build_parent_lookup,
    build_parent_lookup_by_aoi,
    extract_rsi_from_feature,
    resolve_footprint_rsi,
)


class TestExtractRsiFromFeature:
    def test_rsi_dict_from_real_roof_row(self, features_rsi_gdf):
        """RSI dict on a real roof row is extracted correctly."""
        roof = features_rsi_gdf[features_rsi_gdf["class_id"] == ROOF_ID].iloc[0]
        result = extract_rsi_from_feature(roof)
        assert result["roof_spotlight_index"] == roof["roof_spotlight_index"]["value"]
        assert result["roof_spotlight_index_confidence"] == roof["roof_spotlight_index"]["confidence"]

    def test_no_rsi_from_real_bn_row(self, features_rsi_gdf):
        """Building(New) rows have no RSI — returns empty dict."""
        bn = features_rsi_gdf[features_rsi_gdf["class_id"] == BUILDING_NEW_ID].iloc[0]
        assert extract_rsi_from_feature(bn) == {}

    def test_nan_rsi(self, features_rsi_gdf):
        """NaN RSI value (e.g. after DataFrame merge) — returns empty dict."""
        bn = features_rsi_gdf[features_rsi_gdf["class_id"] == BUILDING_NEW_ID].iloc[0]
        feature = bn.copy()
        feature["roof_spotlight_index"] = float("nan")
        assert extract_rsi_from_feature(feature) == {}

    def test_none_rsi(self, features_rsi_gdf):
        """None RSI value — returns empty dict."""
        bn = features_rsi_gdf[features_rsi_gdf["class_id"] == BUILDING_NEW_ID].iloc[0]
        feature = bn.copy()
        feature["roof_spotlight_index"] = None
        assert extract_rsi_from_feature(feature) == {}

    def test_json_string_rsi(self, features_rsi_gdf):
        """RSI as JSON string (Parquet roundtrip) — parsed and extracted correctly."""
        roof = features_rsi_gdf[features_rsi_gdf["class_id"] == ROOF_ID].iloc[0]
        feature = roof.copy()
        expected_value = roof["roof_spotlight_index"]["value"]
        feature["roof_spotlight_index"] = json.dumps({"value": expected_value, "confidence": 0.6})
        result = extract_rsi_from_feature(feature)
        assert result["roof_spotlight_index"] == expected_value

    def test_model_version_propagated(self, features_rsi_gdf):
        """modelVersion in RSI dict is propagated to roof_spotlight_index_model_version."""
        roof = features_rsi_gdf[features_rsi_gdf["class_id"] == ROOF_ID].iloc[0]
        feature = roof.copy()
        feature["roof_spotlight_index"] = {"value": 85, "confidence": 0.7, "modelVersion": "v2"}
        result = extract_rsi_from_feature(feature)
        assert result["roof_spotlight_index"] == 85, "value not extracted"
        assert result["roof_spotlight_index_confidence"] == 0.7, "confidence not extracted"
        assert result["roof_spotlight_index_model_version"] == "v2", "modelVersion not propagated"

    def test_model_version_absent(self, features_rsi_gdf):
        """modelVersion absent from RSI dict — key not present in result."""
        roof = features_rsi_gdf[features_rsi_gdf["class_id"] == ROOF_ID].iloc[0]
        result = extract_rsi_from_feature(roof)
        assert "roof_spotlight_index_model_version" not in result, "key should not be present when modelVersion absent"


class TestResolveFootprintRsi:
    def test_roof_has_rsi_returns_directly(self, features_rsi_gdf):
        """Roof row with RSI — returned without any parent traversal."""
        roof = features_rsi_gdf[features_rsi_gdf["class_id"] == ROOF_ID].iloc[0]
        expected = roof["roof_spotlight_index"]["value"]
        result = resolve_footprint_rsi(roof, parent_lookup={})
        assert result["roof_spotlight_index"] == expected, "roof's own RSI should be returned"

    def test_bn_to_bl_one_hop(self, features_rsi_gdf):
        """Building(New) with no RSI resolves 1-hop to Building Lifecycle."""
        bn = features_rsi_gdf[features_rsi_gdf["class_id"] == BUILDING_NEW_ID].iloc[0]
        bdep = features_rsi_gdf[features_rsi_gdf["class_id"] == BUILDING_ID].iloc[0]
        # Use BDep row as BL template: real column structure, correct class_id/feature_id
        bl = bdep.copy()
        bl["class_id"] = BUILDING_LIFECYCLE_ID
        bl["feature_id"] = bn["parent_id"]  # BN's parent_id points to BL
        bl["parent_id"] = None
        bl["roof_spotlight_index"] = {"value": 42, "confidence": 0.8}
        parent_lookup = {bl["feature_id"]: bl}
        result = resolve_footprint_rsi(bn, parent_lookup=parent_lookup)
        assert result["roof_spotlight_index"] == 42, "BL RSI should be returned via 1-hop"
        assert result["roof_spotlight_index_confidence"] == 0.8

    def test_bn_no_bl_in_lookup(self, features_rsi_gdf):
        """Building(New) with no RSI and no BL in lookup — returns empty dict."""
        bn = features_rsi_gdf[features_rsi_gdf["class_id"] == BUILDING_NEW_ID].iloc[0]
        result = resolve_footprint_rsi(bn, parent_lookup={})
        assert result == {}, "no BL in lookup should yield empty dict"

    def test_broken_parent_chain(self, features_rsi_gdf):
        """parent_id references a feature not in lookup — returns empty dict."""
        bn = features_rsi_gdf[features_rsi_gdf["class_id"] == BUILDING_NEW_ID].iloc[0]
        feature = bn.copy()
        feature["parent_id"] = "nonexistent-feature-id"
        result = resolve_footprint_rsi(feature, parent_lookup={})
        assert result == {}, "missing parent_id target should yield empty dict"

    def test_empty_features_gdf(self, features_rsi_gdf):
        """Empty features GDF — returns empty dict."""
        bn = features_rsi_gdf[features_rsi_gdf["class_id"] == BUILDING_NEW_ID].iloc[0]
        result = resolve_footprint_rsi(bn, gpd.GeoDataFrame())
        assert result == {}

    def test_none_features_gdf(self, features_rsi_gdf):
        """None features GDF — returns empty dict."""
        bn = features_rsi_gdf[features_rsi_gdf["class_id"] == BUILDING_NEW_ID].iloc[0]
        result = resolve_footprint_rsi(bn, None)
        assert result == {}

    def test_prebuilt_lookup_used_for_bl_fallback(self, features_rsi_gdf):
        """Pre-built parent_lookup reused across calls — correct BL RSI resolved."""
        bn = features_rsi_gdf[features_rsi_gdf["class_id"] == BUILDING_NEW_ID].iloc[0]
        bdep = features_rsi_gdf[features_rsi_gdf["class_id"] == BUILDING_ID].iloc[0]
        bl = bdep.copy()
        bl["class_id"] = BUILDING_LIFECYCLE_ID
        bl["feature_id"] = bn["parent_id"]
        bl["parent_id"] = None
        bl["roof_spotlight_index"] = {"value": 77, "confidence": 0.55}
        lookup = {bl["feature_id"]: bl}
        result = resolve_footprint_rsi(bn, parent_lookup=lookup)
        assert result["roof_spotlight_index"] == 77


class TestParentLookupRows:
    """The chunk lookup holds only Roof, Building(New) and parent rows; the single-parcel lookup holds every row."""

    ROWS = [
        ("roof", ROOF_ID, None),
        ("building", BUILDING_NEW_ID, "lifecycle"),
        ("lifecycle", BUILDING_LIFECYCLE_ID, None),
        ("lawn", LAWN_GRASS_ID, None),
        ("tree", VEG_MEDHIGH_ID, "lawn"),
        ("other_tree", VEG_MEDHIGH_ID, None),
    ]

    def _frame(self, with_parent_id=True):
        rows = [{AOI_ID_COLUMN_NAME: "aoi_1", "feature_id": f, "class_id": c, "parent_id": p} for f, c, p in self.ROWS]
        frame = pd.DataFrame(rows)
        return frame if with_parent_id else frame.drop(columns="parent_id")

    def test_single_parcel_lookup_keeps_every_row(self):
        assert set(build_parent_lookup(self._frame())) == {f for f, _, _ in self.ROWS}

    def test_chunk_lookup_keeps_roofs_buildings_and_parents(self):
        lookup = _build_parent_lookup_by_aoi(self._frame())["aoi_1"]
        expected = {"roof", "building", "lifecycle", "lawn"}
        assert set(lookup) == expected, "lookup should hold Roof and Building(New) rows and every row's parent"

    def test_chunk_lookup_without_parent_id_keeps_roofs_and_buildings(self):
        assert set(_build_parent_lookup_by_aoi(self._frame(with_parent_id=False))["aoi_1"]) == {"roof", "building"}

    def test_chunk_lookup_still_resolves_lifecycle_rsi(self):
        frame = self._frame()
        frame["roof_spotlight_index"] = None
        frame.loc[frame["feature_id"] == "lifecycle", "roof_spotlight_index"] = [{"value": 42, "confidence": 0.8}]
        lookup = _build_parent_lookup_by_aoi(frame)["aoi_1"]
        assert resolve_footprint_rsi(lookup["building"], parent_lookup=lookup)["roof_spotlight_index"] == 42

    def test_deprecated_public_name_warns_and_returns_the_chunk_lookup(self):
        with pytest.warns(DeprecationWarning, match="build_parent_lookup_by_aoi"):
            lookup = build_parent_lookup_by_aoi(self._frame())
        assert lookup == _build_parent_lookup_by_aoi(self._frame())
