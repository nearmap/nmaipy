"""Tests for INDS-2030: Resolve footprint RSI from building lifecycle."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    BUILDING_LIFECYCLE_ID,
    BUILDING_NEW_ID,
    ROOF_ID,
)
from nmaipy.parcels import (
    _extract_rsi_from_feature,
    build_parent_lookup,
    resolve_footprint_rsi,
)

# Building (Deprecated) class ID
BUILDING_DEPRECATED_ID = "a2e4ae39-8a61-5515-9d18-8900aa6e6072"


def _make_features_gdf(features: list) -> gpd.GeoDataFrame:
    """Helper to create a GeoDataFrame from a list of feature dicts."""
    df = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")
    if AOI_ID_COLUMN_NAME in df.columns:
        df = df.set_index(AOI_ID_COLUMN_NAME)
    return df


def _make_roof(feature_id="roof-1", parent_id="bldg-dep-1", rsi=None, aoi_id="aoi-1"):
    """Create a roof feature dict."""
    f = {
        AOI_ID_COLUMN_NAME: aoi_id,
        "feature_id": feature_id,
        "class_id": ROOF_ID,
        "description": "Roof",
        "parent_id": parent_id,
        "confidence": 0.9,
        "fidelity": 0.8,
        "area_sqm": 100.0,
        "clipped_area_sqm": 100.0,
        "unclipped_area_sqm": 100.0,
        "area_sqft": 1076.0,
        "clipped_area_sqft": 1076.0,
        "unclipped_area_sqft": 1076.0,
        "geometry": box(-74.275, 40.642, -74.274, 40.643),
        "roof_spotlight_index": rsi,
    }
    return f


def _make_building_deprecated(
    feature_id="bldg-dep-1", parent_id="bl-1", aoi_id="aoi-1"
):
    """Create a Building (Deprecated) feature dict."""
    return {
        AOI_ID_COLUMN_NAME: aoi_id,
        "feature_id": feature_id,
        "class_id": BUILDING_DEPRECATED_ID,
        "description": "Building (Deprecated)",
        "parent_id": parent_id,
        "confidence": 0.9,
        "fidelity": 0.8,
        "area_sqm": 100.0,
        "clipped_area_sqm": 100.0,
        "unclipped_area_sqm": 100.0,
        "area_sqft": 1076.0,
        "clipped_area_sqft": 1076.0,
        "unclipped_area_sqft": 1076.0,
        "geometry": box(-74.275, 40.642, -74.274, 40.643),
        "roof_spotlight_index": None,
    }


def _make_building_lifecycle(feature_id="bl-1", rsi=None, aoi_id="aoi-1"):
    """Create a Building Lifecycle feature dict."""
    return {
        AOI_ID_COLUMN_NAME: aoi_id,
        "feature_id": feature_id,
        "class_id": BUILDING_LIFECYCLE_ID,
        "description": "Building lifecycle",
        "parent_id": None,
        "confidence": 0.9,
        "fidelity": 0.8,
        "area_sqm": 110.0,
        "clipped_area_sqm": 110.0,
        "unclipped_area_sqm": 110.0,
        "area_sqft": 1184.0,
        "clipped_area_sqft": 1184.0,
        "unclipped_area_sqft": 1184.0,
        "geometry": box(-74.2755, 40.6415, -74.2735, 40.6435),
        "roof_spotlight_index": rsi,
    }


def _make_building_new(feature_id="bldg-new-1", parent_id="bl-1", aoi_id="aoi-1"):
    """Create a Building (New) feature dict."""
    return {
        AOI_ID_COLUMN_NAME: aoi_id,
        "feature_id": feature_id,
        "class_id": BUILDING_NEW_ID,
        "description": "Building",
        "parent_id": parent_id,
        "confidence": 0.9,
        "fidelity": 0.8,
        "area_sqm": 100.0,
        "clipped_area_sqm": 100.0,
        "unclipped_area_sqm": 100.0,
        "area_sqft": 1076.0,
        "clipped_area_sqft": 1076.0,
        "unclipped_area_sqft": 1076.0,
        "geometry": box(-74.275, 40.642, -74.274, 40.643),
        "roof_spotlight_index": None,
    }


# --- Unit tests for _extract_rsi_from_feature ---


class TestExtractRsiFromFeature:
    def test_dict_format(self):
        """RSI as a dict (fresh from API)."""
        feature = {"roof_spotlight_index": {"value": 85, "confidence": 0.7}}
        result = _extract_rsi_from_feature(feature)
        assert result["footprint_roof_spotlight_index"] == 85
        assert result["footprint_roof_spotlight_index_confidence"] == 0.7

    def test_none(self):
        """No RSI on feature."""
        feature = {"roof_spotlight_index": None}
        assert _extract_rsi_from_feature(feature) == {}

    def test_nan(self):
        """NaN RSI (from DataFrame)."""
        feature = {"roof_spotlight_index": float("nan")}
        assert _extract_rsi_from_feature(feature) == {}

    def test_json_string(self):
        """RSI as JSON string (from Parquet roundtrip)."""
        import json

        feature = {"roof_spotlight_index": json.dumps({"value": 90, "confidence": 0.6})}
        result = _extract_rsi_from_feature(feature)
        assert result["footprint_roof_spotlight_index"] == 90

    def test_series(self):
        """RSI from a pandas Series (as in GeoDataFrame row)."""
        s = pd.Series({"roof_spotlight_index": {"value": 75, "confidence": 0.5}})
        result = _extract_rsi_from_feature(s)
        assert result["footprint_roof_spotlight_index"] == 75


# --- Unit tests for resolve_footprint_rsi ---


class TestResolveFootprintRsi:
    def test_roof_has_rsi(self):
        """Roof has RSI directly — returns it without traversing."""
        rsi = {"value": 88, "confidence": 0.6}
        roof = _make_roof(rsi=rsi)
        features = _make_features_gdf([roof])
        result = resolve_footprint_rsi(pd.Series(roof), features)
        assert result["footprint_roof_spotlight_index"] == 88
        assert result["footprint_roof_spotlight_index_confidence"] == 0.6

    def test_rsi_on_building_lifecycle(self):
        """Roof has no RSI, BL grandparent does — returns BL's RSI."""
        rsi = {"value": 42, "confidence": 0.8}
        roof = _make_roof(rsi=None)
        bldg_dep = _make_building_deprecated()
        bl = _make_building_lifecycle(rsi=rsi)
        features = _make_features_gdf([roof, bldg_dep, bl])
        result = resolve_footprint_rsi(pd.Series(roof), features)
        assert result["footprint_roof_spotlight_index"] == 42
        assert result["footprint_roof_spotlight_index_confidence"] == 0.8

    def test_neither_has_rsi(self):
        """Neither roof nor BL has RSI."""
        roof = _make_roof(rsi=None)
        bldg_dep = _make_building_deprecated()
        bl = _make_building_lifecycle(rsi=None)
        features = _make_features_gdf([roof, bldg_dep, bl])
        result = resolve_footprint_rsi(pd.Series(roof), features)
        assert result == {}

    def test_no_bl_in_features(self):
        """Only roof, no BL features — returns empty."""
        roof = _make_roof(rsi=None)
        features = _make_features_gdf([roof])
        result = resolve_footprint_rsi(pd.Series(roof), features)
        assert result == {}

    def test_broken_parent_chain(self):
        """Parent_id points to feature not in GDF — returns empty."""
        roof = _make_roof(rsi=None, parent_id="missing-id")
        bl = _make_building_lifecycle(rsi={"value": 50, "confidence": 0.9})
        features = _make_features_gdf([roof, bl])
        result = resolve_footprint_rsi(pd.Series(roof), features)
        assert result == {}

    def test_building_new_one_hop(self):
        """Building (New) → parent_id → BL (1 hop)."""
        rsi = {"value": 60, "confidence": 0.7}
        bldg = _make_building_new(parent_id="bl-1")
        bl = _make_building_lifecycle(rsi=rsi)
        features = _make_features_gdf([bldg, bl])
        result = resolve_footprint_rsi(pd.Series(bldg), features)
        assert result["footprint_roof_spotlight_index"] == 60

    def test_with_parent_lookup(self):
        """Using pre-built parent_lookup for performance."""
        rsi = {"value": 77, "confidence": 0.55}
        roof = _make_roof(rsi=None)
        bldg_dep = _make_building_deprecated()
        bl = _make_building_lifecycle(rsi=rsi)
        features = _make_features_gdf([roof, bldg_dep, bl])
        lookup = build_parent_lookup(features)
        result = resolve_footprint_rsi(pd.Series(roof), parent_lookup=lookup)
        assert result["footprint_roof_spotlight_index"] == 77

    def test_empty_features_gdf(self):
        """Empty features GDF — returns empty."""
        roof = _make_roof(rsi=None)
        features = gpd.GeoDataFrame()
        result = resolve_footprint_rsi(pd.Series(roof), features)
        assert result == {}

    def test_none_features_gdf(self):
        """None features GDF — returns empty."""
        roof = _make_roof(rsi=None)
        result = resolve_footprint_rsi(pd.Series(roof), None)
        assert result == {}
