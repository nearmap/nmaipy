"""
Unit tests for the per-class export column gating logic.

`is_primary` is gated on `class_id in PRIMARY_FEATURE_CLASS_IDS` and `fidelity`
is gated on `class_id in CLASSES_WITH_FIDELITY` in `_compute_feature_class_data`.
Without these gates, the chunk-level `pd.concat` of features across classes
would leak NaN-filled `is_primary` / `fidelity` columns into per-class exports
where they don't apply (e.g. solar, pools).

These tests construct a minimal cross-class chunk_gdf and verify the gating
deterministically, independent of API content.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    BUILDING_NEW_ID,
    CLASSES_WITH_FIDELITY,
    POOL_ID,
    PRIMARY_FEATURE_CLASS_IDS,
    ROOF_ID,
    SOLAR_ID,
)
from nmaipy.exporter import _add_is_primary_column, _compute_all_per_class_data


def _row(*, feature_id, class_id, description, x, y, fidelity=None, confidence=0.9):
    return {
        AOI_ID_COLUMN_NAME: "aoi-1",
        "feature_id": feature_id,
        "class_id": class_id,
        "description": description,
        "confidence": confidence,
        "fidelity": fidelity,
        "geometry": box(x, y, x + 0.0005, y + 0.0005),
        "area_sqm": 100.0,
        "clipped_area_sqm": 100.0,
        "unclipped_area_sqm": 100.0,
        "area_sqft": 1076.0,
        "clipped_area_sqft": 1076.0,
        "unclipped_area_sqft": 1076.0,
        "parent_id": None,
        "survey_date": "2024-01-01",
        "mesh_date": "2024-01-01",
    }


@pytest.fixture
def cross_class_chunk_gdf():
    """Synthetic chunk-level GDF with one feature per class.

    Mirrors the shape of `final_features_df` after upstream pd.concat: every
    row carries every column (with None where the API didn't populate the
    field for that class). Solar/pool rows have None in `fidelity`, roof and
    building rows have a real value.
    """
    rows = [
        _row(
            feature_id="roof-1",
            class_id=ROOF_ID,
            description="Roof",
            x=-111.8905,
            y=40.7610,
            fidelity=0.85,
        ),
        _row(
            feature_id="building-1",
            class_id=BUILDING_NEW_ID,
            description="Building",
            x=-111.8905,
            y=40.7611,
            fidelity=0.95,
        ),
        _row(
            feature_id="pool-1",
            class_id=POOL_ID,
            description="Swimming Pool",
            x=-111.8905,
            y=40.7612,
            fidelity=None,
        ),
        _row(
            feature_id="solar-1",
            class_id=SOLAR_ID,
            description="Solar Panel",
            x=-111.8905,
            y=40.7613,
            fidelity=None,
        ),
    ]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=API_CRS)
    return gdf


@pytest.fixture
def primary_ids_df():
    """Synthetic rollup primary IDs covering all PRIMARY_FEATURE_CLASS_IDS members."""
    return pd.DataFrame(
        {
            "primary_roof_feature_id": ["roof-1"],
            "primary_building_feature_id": ["building-1"],
            "primary_swimming_pool_feature_id": ["pool-1"],
        },
        index=pd.Index(["aoi-1"], name=AOI_ID_COLUMN_NAME),
    )


def test_solar_per_class_export_omits_is_primary_and_fidelity(cross_class_chunk_gdf, primary_ids_df):
    chunk_gdf = _add_is_primary_column(cross_class_chunk_gdf.copy(), primary_ids_df)
    assert "is_primary" in chunk_gdf.columns

    results = _compute_all_per_class_data(
        chunk_gdf=chunk_gdf,
        country="us",
        aoi_input_columns=[],
        whitelisted_classes=[SOLAR_ID],
    )
    assert SOLAR_ID in results

    tabular_cols = set(results[SOLAR_ID]["tabular"].column_names)
    geo_cols = set(results[SOLAR_ID]["geo"].column_names)

    assert SOLAR_ID not in PRIMARY_FEATURE_CLASS_IDS
    assert SOLAR_ID not in CLASSES_WITH_FIDELITY
    assert "is_primary" not in tabular_cols
    assert "is_primary" not in geo_cols
    assert "fidelity" not in tabular_cols
    assert "fidelity" not in geo_cols


def test_pool_per_class_export_has_is_primary_but_no_fidelity(cross_class_chunk_gdf, primary_ids_df):
    chunk_gdf = _add_is_primary_column(cross_class_chunk_gdf.copy(), primary_ids_df)

    results = _compute_all_per_class_data(
        chunk_gdf=chunk_gdf,
        country="us",
        aoi_input_columns=[],
        whitelisted_classes=[POOL_ID],
    )
    assert POOL_ID in results

    tabular_cols = set(results[POOL_ID]["tabular"].column_names)
    geo_cols = set(results[POOL_ID]["geo"].column_names)

    assert POOL_ID in PRIMARY_FEATURE_CLASS_IDS
    assert POOL_ID not in CLASSES_WITH_FIDELITY
    assert "is_primary" in tabular_cols
    assert "is_primary" in geo_cols
    assert "fidelity" not in tabular_cols
    assert "fidelity" not in geo_cols


def test_roof_per_class_export_has_both_is_primary_and_fidelity(cross_class_chunk_gdf, primary_ids_df):
    chunk_gdf = _add_is_primary_column(cross_class_chunk_gdf.copy(), primary_ids_df)

    results = _compute_all_per_class_data(
        chunk_gdf=chunk_gdf,
        country="us",
        aoi_input_columns=[],
        whitelisted_classes=[ROOF_ID],
    )
    assert ROOF_ID in results

    tabular_cols = set(results[ROOF_ID]["tabular"].column_names)
    geo_cols = set(results[ROOF_ID]["geo"].column_names)

    assert ROOF_ID in PRIMARY_FEATURE_CLASS_IDS
    assert ROOF_ID in CLASSES_WITH_FIDELITY
    assert "is_primary" in tabular_cols
    assert "is_primary" in geo_cols
    assert "fidelity" in tabular_cols
    assert "fidelity" in geo_cols


def test_building_per_class_export_has_both_columns(cross_class_chunk_gdf, primary_ids_df):
    chunk_gdf = _add_is_primary_column(cross_class_chunk_gdf.copy(), primary_ids_df)

    results = _compute_all_per_class_data(
        chunk_gdf=chunk_gdf,
        country="us",
        aoi_input_columns=[],
        whitelisted_classes=[BUILDING_NEW_ID],
    )
    assert BUILDING_NEW_ID in results

    tabular_cols = set(results[BUILDING_NEW_ID]["tabular"].column_names)

    assert BUILDING_NEW_ID in PRIMARY_FEATURE_CLASS_IDS
    assert BUILDING_NEW_ID in CLASSES_WITH_FIDELITY
    assert "is_primary" in tabular_cols
    assert "fidelity" in tabular_cols
