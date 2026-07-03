"""Regression tests for multiparcel roof attribute cache keying.

In ``parcelMode`` a roof that overlaps more than one parcel is returned once per
parcel, each row carrying attributes (RSI, component areas, ...) recomputed on
*that* parcel's clipped geometry. The per-class attribute cache in
``_compute_all_per_class_data`` must therefore key on ``(aoi_id, feature_id)`` —
keying by ``feature_id`` alone lets one parcel's clip overwrite another's, so
every duplicate row inherits whichever parcel was flattened last.

See the roof-only bug: same roof, two parcels, RSI 15/0.57 (this parcel) vs
96/0.92 (neighbour) — both rows used to export 96/0.92.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from nmaipy.constants import AOI_ID_COLUMN_NAME, BUILDING_NEW_ID, ROOF_ID
from nmaipy.exporter import _compute_all_per_class_data

SHARED_ROOF_FID = "2aa2818b-50ba-5470-9c34-59ed24fb7fa4"

# Square geometries per parcel — distinct locations so each AOI's roof/building
# pair matches by IoU without cross-AOI overlap.
_SOUTH_GEOM = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
_NORTH_GEOM = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])


def _roof_row(aoi_id, geom, rsi_value, rsi_conf, clipped_sqft):
    return {
        AOI_ID_COLUMN_NAME: aoi_id,
        "feature_id": SHARED_ROOF_FID,
        "class_id": ROOF_ID,
        "description": "Roof",
        "confidence": 0.9,
        "fidelity": 0.9,
        "parent_id": None,
        "is_primary": True,
        "geometry": geom,
        "area_sqft": clipped_sqft,
        "clipped_area_sqft": clipped_sqft,
        "unclipped_area_sqft": 1147.0,
        "area_sqm": round(clipped_sqft / 10.7639, 2),
        "clipped_area_sqm": round(clipped_sqft / 10.7639, 2),
        "unclipped_area_sqm": 106.6,
        "attributes": [],
        "roof_spotlight_index": {"value": rsi_value, "confidence": rsi_conf, "modelVersion": "B.0"},
        "survey_date": "2026-03-28",
        "mesh_date": "",
        "belongs_to_parcel": True,
    }


def _building_row(aoi_id, feature_id, geom):
    return {
        AOI_ID_COLUMN_NAME: aoi_id,
        "feature_id": feature_id,
        "class_id": BUILDING_NEW_ID,
        "description": "Building",
        "confidence": 0.95,
        "fidelity": 0.95,
        "parent_id": None,
        "is_primary": True,
        "geometry": geom,
        "area_sqft": 900.0,
        "clipped_area_sqft": 900.0,
        "unclipped_area_sqft": 900.0,
        "area_sqm": 83.6,
        "clipped_area_sqm": 83.6,
        "unclipped_area_sqm": 83.6,
        "attributes": [],
        "survey_date": "2026-03-28",
        "mesh_date": "",
        "belongs_to_parcel": True,
    }


def _make_chunk(rows, indexed=True):
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    # Standalone layout indexes by aoi_id; streaming layout keeps it as a column
    # with a RangeIndex (after the on=aoi_id merges). Both must resolve the same.
    return gdf.set_index(AOI_ID_COLUMN_NAME) if indexed else gdf


@pytest.mark.parametrize("indexed", [True, False], ids=["aoi_index", "aoi_column"])
def test_multiparcel_roof_gets_own_clip_rsi(indexed):
    """Roof-only export: each parcel's row keeps its own clipped RSI."""
    chunk = _make_chunk(
        [
            _roof_row("parcel_south", _SOUTH_GEOM, 15, 0.57, 691.0),
            _roof_row("parcel_north", _NORTH_GEOM, 96, 0.92, 455.0),
        ],
        indexed=indexed,
    )
    res = _compute_all_per_class_data(
        chunk, country="us", aoi_input_columns=[], whitelisted_classes=[ROOF_ID], threads=1
    )
    roof = res[ROOF_ID]["tabular"].to_pandas().set_index(AOI_ID_COLUMN_NAME)

    assert roof.loc["parcel_south", "roof_spotlight_index"] == 15
    assert roof.loc["parcel_south", "roof_spotlight_index_confidence"] == 0.57
    assert roof.loc["parcel_north", "roof_spotlight_index"] == 96
    assert roof.loc["parcel_north", "roof_spotlight_index_confidence"] == 0.92


def test_multiparcel_building_min_max_rsi_uses_own_parcel_clip():
    """Building(New) export: per-building child-roof RSI min/max resolve to the
    building's own parcel clip of a shared multiparcel roof (Section D layer-1)."""
    chunk = _make_chunk(
        [
            _roof_row("parcel_south", _SOUTH_GEOM, 15, 0.57, 691.0),
            _roof_row("parcel_north", _NORTH_GEOM, 96, 0.92, 455.0),
            _building_row("parcel_south", "bldg-south", _SOUTH_GEOM),
            _building_row("parcel_north", "bldg-north", _NORTH_GEOM),
        ]
    )
    res = _compute_all_per_class_data(
        chunk,
        country="us",
        aoi_input_columns=[],
        whitelisted_classes=[ROOF_ID, BUILDING_NEW_ID],
        threads=1,
    )
    bldg = res[BUILDING_NEW_ID]["tabular"].to_pandas().set_index(AOI_ID_COLUMN_NAME)

    # Single child roof per building → min == max == that parcel's own clip RSI.
    assert bldg.loc["parcel_south", "roof_spotlight_index_min"] == 15
    assert bldg.loc["parcel_south", "roof_spotlight_index_max"] == 15
    assert bldg.loc["parcel_north", "roof_spotlight_index_min"] == 96
    assert bldg.loc["parcel_north", "roof_spotlight_index_max"] == 96
