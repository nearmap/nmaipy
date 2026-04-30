"""
Tests for country-aware unit emission across per-feature exports.

The Feature API returns both metric and imperial area columns; per-feature exports
should keep only the country-correct family (matching rollup behaviour). These tests
verify the four sites that the fix touches:

1. constants helpers — `country_area_suffix`, `wrong_unit_area_columns`.
2. Per-class assembler in `_compute_feature_class_data` (US: sqft only, AU: sqm only).
3. `extract_building_features` populates only the country-correct columns.
4. `flatten_roof_attributes` clipping detection works for US (regression test for the
   previously latent metric-only hardcoding bug).
"""

from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    BUILDING_NEW_ID,
    POOL_ID,
    ROOF_ID,
    ROOF_INSTANCE_CLASS_ID,
    SOLAR_ID,
    SQUARED_METERS_TO_SQUARED_FEET,
    country_area_suffix,
    wrong_unit_area_columns,
)
from nmaipy.exporter import _add_is_primary_column, _compute_all_per_class_data
from nmaipy.feature_attributes import flatten_roof_attributes
from nmaipy.parcels import extract_building_features

# ---------------------------------------------------------------------------
# 1. Constants helpers
# ---------------------------------------------------------------------------


def test_country_area_suffix_imperial():
    assert country_area_suffix("us") == "sqft"
    assert country_area_suffix("US") == "sqft"


def test_country_area_suffix_metric():
    assert country_area_suffix("au") == "sqm"
    assert country_area_suffix("ca") == "sqm"
    assert country_area_suffix("nz") == "sqm"


def test_wrong_unit_area_columns_us():
    assert wrong_unit_area_columns("us") == ["area_sqm", "clipped_area_sqm", "unclipped_area_sqm"]


def test_wrong_unit_area_columns_au():
    assert wrong_unit_area_columns("au") == ["area_sqft", "clipped_area_sqft", "unclipped_area_sqft"]


# ---------------------------------------------------------------------------
# 2. Per-class assembler
# ---------------------------------------------------------------------------


def _row(*, feature_id, class_id, description, x, y, fidelity=None):
    """Synthetic row with both unit families populated (mirrors API response post-flatten)."""
    return {
        AOI_ID_COLUMN_NAME: "aoi-1",
        "feature_id": feature_id,
        "class_id": class_id,
        "description": description,
        "confidence": 0.9,
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
    rows = [
        _row(feature_id="roof-1", class_id=ROOF_ID, description="Roof", x=-111.89, y=40.76, fidelity=0.85),
        _row(feature_id="bld-1", class_id=BUILDING_NEW_ID, description="Building", x=-111.89, y=40.7611, fidelity=0.95),
        _row(feature_id="pool-1", class_id=POOL_ID, description="Swimming Pool", x=-111.89, y=40.7612),
        _row(feature_id="solar-1", class_id=SOLAR_ID, description="Solar Panel", x=-111.89, y=40.7613),
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=API_CRS)


@pytest.fixture
def primary_ids_df():
    return pd.DataFrame(
        {
            "primary_roof_feature_id": ["roof-1"],
            "primary_building_feature_id": ["bld-1"],
            "primary_swimming_pool_feature_id": ["pool-1"],
        },
        index=pd.Index(["aoi-1"], name=AOI_ID_COLUMN_NAME),
    )


@pytest.mark.parametrize(
    "country,kept,dropped",
    [
        ("us", "sqft", "sqm"),
        ("au", "sqm", "sqft"),
    ],
)
def test_per_class_assembler_emits_only_country_unit(cross_class_chunk_gdf, primary_ids_df, country, kept, dropped):
    chunk_gdf = _add_is_primary_column(cross_class_chunk_gdf.copy(), primary_ids_df)
    results = _compute_all_per_class_data(
        chunk_gdf=chunk_gdf,
        country=country,
        aoi_input_columns=[],
        whitelisted_classes=[ROOF_ID],
    )
    cols = set(results[ROOF_ID]["tabular"].column_names)
    assert f"area_{kept}" in cols
    assert f"clipped_area_{kept}" in cols
    assert f"unclipped_area_{kept}" in cols
    assert f"area_{dropped}" not in cols
    assert f"clipped_area_{dropped}" not in cols
    assert f"unclipped_area_{dropped}" not in cols


@pytest.mark.parametrize("country,kept,dropped", [("us", "sqft", "sqm"), ("au", "sqm", "sqft")])
def test_roof_instance_per_class_export_emits_only_country_unit(country, kept, dropped):
    """Roof instances have a single area (no clipped/unclipped distinction)."""
    rows = [
        {
            AOI_ID_COLUMN_NAME: "aoi-1",
            "feature_id": "ri-1",
            "class_id": ROOF_INSTANCE_CLASS_ID,
            "description": "Roof Instance",
            "geometry": box(-111.89, 40.76, -111.8895, 40.7605),
            "area_sqm": 100.0,
            "area_sqft": 1076.0,
        }
    ]
    chunk_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=API_CRS)
    chunk_gdf["is_primary"] = False  # downstream logic expects this column

    results = _compute_all_per_class_data(
        chunk_gdf=chunk_gdf,
        country=country,
        aoi_input_columns=[],
        whitelisted_classes=[ROOF_INSTANCE_CLASS_ID],
    )
    cols = set(results[ROOF_INSTANCE_CLASS_ID]["tabular"].column_names)
    assert f"area_{kept}" in cols
    assert f"area_{dropped}" not in cols


# ---------------------------------------------------------------------------
# 3. Buildings extract
# ---------------------------------------------------------------------------


def _building_features_and_parcels():
    """Minimal features_gdf + parcels_gdf for extract_building_features."""
    polygon = box(-111.89, 40.76, -111.8895, 40.7605)
    features = pd.DataFrame(
        [
            {
                "feature_id": "bld-1",
                "class_id": BUILDING_NEW_ID,
                "description": "Building",
                "confidence": 0.9,
                "fidelity": 0.95,
                "area_sqm": 200.0,
                "clipped_area_sqm": 180.0,
                "unclipped_area_sqm": 200.0,
                "area_sqft": 2153.0,
                "clipped_area_sqft": 1937.0,
                "unclipped_area_sqft": 2153.0,
                "survey_date": "2024-01-01",
                "mesh_date": "2024-01-01",
                "geometry": polygon,
                "is_primary": True,
            }
        ],
        index=pd.Index(["aoi-1"], name=AOI_ID_COLUMN_NAME),
    )
    features_gdf = gpd.GeoDataFrame(features, geometry="geometry", crs=API_CRS)
    parcels_gdf = gpd.GeoDataFrame(
        {"geometry": [polygon]},
        index=pd.Index(["aoi-1"], name=AOI_ID_COLUMN_NAME),
        crs=API_CRS,
    )
    return features_gdf, parcels_gdf


@pytest.mark.parametrize("country,kept,dropped", [("us", "sqft", "sqm"), ("au", "sqm", "sqft")])
def test_extract_building_features_emits_only_country_unit(country, kept, dropped):
    features_gdf, parcels_gdf = _building_features_and_parcels()
    out = extract_building_features(parcels_gdf=parcels_gdf, features_gdf=features_gdf, country=country)
    cols = set(out.columns)
    assert f"area_{kept}" in cols
    assert f"clipped_area_{kept}" in cols
    assert f"unclipped_area_{kept}" in cols
    assert f"area_{dropped}" not in cols
    assert f"clipped_area_{dropped}" not in cols
    assert f"unclipped_area_{dropped}" not in cols


# ---------------------------------------------------------------------------
# 4. Clipping detection regression test (real bug fix)
# ---------------------------------------------------------------------------


def _clipped_roof(country: str):
    """A single roof dict with area in the country-correct unit, clipped to 50%.

    A clipped roof has clipped_area < unclipped_area (here 50%). The fix verifies that
    `flatten_roof_attributes` uses the country-correct column suffix to detect this and
    triggers component recalculation.
    """
    suffix = country_area_suffix(country)
    return {
        "feature_id": "roof-1",
        "class_id": ROOF_ID,
        f"area_{suffix}": 50.0,
        f"clipped_area_{suffix}": 50.0,
        f"unclipped_area_{suffix}": 100.0,  # 50% clipped → triggers recalc path
        "geometry": box(0, 0, 0.0005, 0.0005),
        "attributes": [],  # no components → recalc path is invoked but emits nothing
    }


def _roof_with_clipped_components(suffix: str) -> dict:
    """Roof row with clipped < unclipped areas and a single material component.

    Provides the structure needed for the recalculation branch: API-format `attributes`
    list with a `components` entry, and a child class_id so calculate_child_feature_attributes
    has work to do.
    """
    component_class_id = "test-tile-class-id"
    return {
        "feature_id": "roof-1",
        "class_id": ROOF_ID,
        f"area_{suffix}": 50.0,
        f"clipped_area_{suffix}": 50.0,
        f"unclipped_area_{suffix}": 100.0,  # 50% clipped
        "geometry": box(0, 0, 0.0005, 0.0005),
        "attributes": [
            {
                "description": "Roof material",
                "components": [
                    {
                        "classId": component_class_id,
                        "description": "Tile",
                        # Both unit families always present in the API response.
                        "areaSqm": 25.0,
                        "areaSqft": 269.0,
                        "ratio": 0.5,
                        "confidence": 0.9,
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize("country", ["us", "au"])
def test_flatten_roof_attributes_invokes_recalc_when_clipped(country):
    """Clipping detection drives the recalc path; this is the real regression test.

    Before the fix, US exports hardcoded `clipped_area_sqm` / `unclipped_area_sqm`. When the
    export-time unit filter dropped those columns for US, both lookups returned None, the
    `is_clipped` flag stayed False, and `calculate_child_feature_attributes` was never called
    on a clipped roof — silently dropping recalculated component areas from US output.

    After the fix, the function uses the country-correct suffix and the recalc path fires.
    We assert that by spying on `calculate_child_feature_attributes`.
    """
    suffix = country_area_suffix(country)
    roof = _roof_with_clipped_components(suffix)

    # Provide a non-None child_features so the recalc gate (`is_clipped and child_features
    # is not None`) can be reached. An empty GeoDataFrame is enough to exercise the call.
    child_features = gpd.GeoDataFrame(columns=["class_id", "geometry"], geometry="geometry", crs=API_CRS)

    with patch("nmaipy.parcels.calculate_child_feature_attributes") as mock_recalc:
        mock_recalc.return_value = {f"tile_area_{suffix}": 25.0, "tile_ratio": 0.5}
        flatten_roof_attributes([roof], country=country, child_features=child_features)

    assert mock_recalc.called, (
        f"Expected calculate_child_feature_attributes to be invoked for a clipped roof "
        f"with country={country!r} and area_{suffix} columns, but it was not. "
        f"This indicates the clipping-detection bug has regressed."
    )


@pytest.mark.parametrize("country", ["us", "au"])
def test_flatten_roof_attributes_skips_recalc_when_not_clipped(country):
    """Inverse of the regression test: no recalc when clipped == unclipped (full coverage)."""
    suffix = country_area_suffix(country)
    roof = _roof_with_clipped_components(suffix)
    # Full coverage: no clipping → no recalc.
    roof[f"clipped_area_{suffix}"] = roof[f"unclipped_area_{suffix}"]

    child_features = gpd.GeoDataFrame(columns=["class_id", "geometry"], geometry="geometry", crs=API_CRS)

    with patch("nmaipy.parcels.calculate_child_feature_attributes") as mock_recalc:
        flatten_roof_attributes([roof], country=country, child_features=child_features)

    assert not mock_recalc.called, "Recalc should NOT fire on an unclipped roof"
