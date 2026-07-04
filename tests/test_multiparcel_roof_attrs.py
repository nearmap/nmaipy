"""Regression tests for multiparcel per-parcel row identity in parcelMode.

In ``parcelMode`` ANY feature (roof, building, building lifecycle) that overlaps
more than one parcel is returned once per parcel, each row carrying attributes
(RSI, component areas, ...) recomputed on *that* parcel's clipped geometry. The
row identity is therefore ``(aoi_id, feature_id)`` — every cross-row structure
keyed by ``feature_id`` alone (attribute caches, parent-chain lookups,
roof→building→BL linkage) lets one parcel's clip overwrite another's, so
duplicate rows inherit whichever parcel came last.

Covered here:
- roof attribute cache (roof class, both frame layouts)
- Building(New): child-roof RSI min/max AND headline resolved RSI
- Building Lifecycle: primary-child-roof linkage and resolved RSI
- roof-class RSI fallback via BL when the roof has no RSI of its own
- parcel_rollup: per-parcel RSI min/max via the chunk-hoisted parent lookup

See the original roof-only bug: same roof, two parcels, RSI 15/0.57 (this
parcel) vs 96/0.92 (neighbour) — both rows used to export 96/0.92.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from nmaipy.constants import AOI_ID_COLUMN_NAME, BUILDING_LIFECYCLE_ID, BUILDING_NEW_ID, ROOF_ID
from nmaipy.exporter import _compute_all_per_class_data
from nmaipy.parcels import parcel_rollup

SHARED_ROOF_FID = "2aa2818b-50ba-5470-9c34-59ed24fb7fa4"
SHARED_BUILDING_FID = "7f3d9c01-1b2a-5e4d-8f6c-0123456789ab"
SHARED_BL_FID = "9e8d7c65-4b3a-52f1-8e0d-fedcba987654"

# Square geometries per parcel — distinct locations so each AOI's roof/building
# pair matches by IoU without cross-AOI overlap.
_SOUTH_GEOM = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
_NORTH_GEOM = Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])


def _roof_row(aoi_id, geom, rsi_value, rsi_conf, clipped_sqft, with_rsi=True):
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
        "roof_spotlight_index": (
            {"value": rsi_value, "confidence": rsi_conf, "modelVersion": "B.0"} if with_rsi else None
        ),
        "survey_date": "2026-03-28",
        "mesh_date": "",
        "belongs_to_parcel": True,
    }


def _bl_row(aoi_id, geom, rsi_value=None, rsi_conf=None):
    return {
        AOI_ID_COLUMN_NAME: aoi_id,
        "feature_id": SHARED_BL_FID,
        "class_id": BUILDING_LIFECYCLE_ID,
        "description": "Building Lifecycle",
        "confidence": 0.9,
        "fidelity": 0.9,
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
        "roof_spotlight_index": (
            {"value": rsi_value, "confidence": rsi_conf, "modelVersion": "B.0"} if rsi_value is not None else None
        ),
        "survey_date": "2026-03-28",
        "mesh_date": "",
        "belongs_to_parcel": True,
    }


def _building_row(aoi_id, feature_id, geom, parent_id=None):
    return {
        AOI_ID_COLUMN_NAME: aoi_id,
        "feature_id": feature_id,
        "class_id": BUILDING_NEW_ID,
        "description": "Building",
        "confidence": 0.95,
        "fidelity": 0.95,
        "parent_id": parent_id,
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


def test_multiparcel_building_headline_rsi_uses_own_parcel_clip():
    """Building(New) export: the headline resolved roof_spotlight_index goes
    through the parent lookup (building → primary child roof row), which must be
    scoped to the building's own parcel — a shared multiparcel building must not
    read the other parcel's roof clip."""
    chunk = _make_chunk(
        [
            _roof_row("parcel_south", _SOUTH_GEOM, 15, 0.57, 691.0),
            _roof_row("parcel_north", _NORTH_GEOM, 96, 0.92, 455.0),
            _building_row("parcel_south", SHARED_BUILDING_FID, _SOUTH_GEOM),
            _building_row("parcel_north", SHARED_BUILDING_FID, _NORTH_GEOM),
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

    assert bldg.loc["parcel_south", "roof_spotlight_index"] == 15
    assert bldg.loc["parcel_south", "roof_spotlight_index_confidence"] == 0.57
    assert bldg.loc["parcel_north", "roof_spotlight_index"] == 96
    assert bldg.loc["parcel_north", "roof_spotlight_index_confidence"] == 0.92


def test_multiparcel_bl_rsi_and_linkage_use_own_parcel_clip():
    """Building Lifecycle export: the Roof →(IoU)→ BN →(parent_id)→ BL chain and
    the resolved RSI must stay within each BL row's own parcel for a shared
    multiparcel roof/building/BL."""
    chunk = _make_chunk(
        [
            _roof_row("parcel_south", _SOUTH_GEOM, 15, 0.57, 691.0),
            _roof_row("parcel_north", _NORTH_GEOM, 96, 0.92, 455.0),
            _building_row("parcel_south", SHARED_BUILDING_FID, _SOUTH_GEOM, parent_id=SHARED_BL_FID),
            _building_row("parcel_north", SHARED_BUILDING_FID, _NORTH_GEOM, parent_id=SHARED_BL_FID),
            _bl_row("parcel_south", _SOUTH_GEOM),
            _bl_row("parcel_north", _NORTH_GEOM),
        ]
    )
    res = _compute_all_per_class_data(
        chunk,
        country="us",
        aoi_input_columns=[],
        whitelisted_classes=[ROOF_ID, BUILDING_NEW_ID, BUILDING_LIFECYCLE_ID],
        threads=1,
    )
    bl = res[BUILDING_LIFECYCLE_ID]["tabular"].to_pandas().set_index(AOI_ID_COLUMN_NAME)

    # Linkage resolves within each parcel
    assert bl.loc["parcel_south", "primary_child_roof_id"] == SHARED_ROOF_FID
    assert bl.loc["parcel_north", "primary_child_roof_id"] == SHARED_ROOF_FID
    assert bl.loc["parcel_south", "child_roof_count"] == 1
    assert bl.loc["parcel_north", "child_roof_count"] == 1

    # Resolved RSI comes from the BL row's own parcel clip of the shared roof
    assert bl.loc["parcel_south", "roof_spotlight_index"] == 15
    assert bl.loc["parcel_south", "roof_spotlight_index_confidence"] == 0.57
    assert bl.loc["parcel_north", "roof_spotlight_index"] == 96
    assert bl.loc["parcel_north", "roof_spotlight_index_confidence"] == 0.92


def test_multiparcel_roof_bl_fallback_rsi_uses_own_parcel_bl():
    """Roof export, no-own-RSI case: RSI falls back through Roof →(IoU)→ BN
    →(parent_id)→ BL, and must land on the roof row's own parcel's BL clip.
    This is the 'RSI applied to the building lifecycle instead of the roof'
    scenario (e.g. structural damage)."""
    chunk = _make_chunk(
        [
            _roof_row("parcel_south", _SOUTH_GEOM, None, None, 691.0, with_rsi=False),
            _roof_row("parcel_north", _NORTH_GEOM, 96, 0.92, 455.0),
            _building_row("parcel_south", SHARED_BUILDING_FID, _SOUTH_GEOM, parent_id=SHARED_BL_FID),
            _building_row("parcel_north", SHARED_BUILDING_FID, _NORTH_GEOM, parent_id=SHARED_BL_FID),
            _bl_row("parcel_south", _SOUTH_GEOM, rsi_value=44, rsi_conf=0.61),
            _bl_row("parcel_north", _NORTH_GEOM, rsi_value=77, rsi_conf=0.71),
        ]
    )
    res = _compute_all_per_class_data(
        chunk,
        country="us",
        aoi_input_columns=[],
        whitelisted_classes=[ROOF_ID],
        threads=1,
    )
    roof = res[ROOF_ID]["tabular"].to_pandas().set_index(AOI_ID_COLUMN_NAME)

    # South roof has no RSI of its own → resolved from parcel_south's BL clip (44),
    # not parcel_north's BL clip (77). North roof keeps its own RSI.
    assert roof.loc["parcel_south", "roof_spotlight_index"] == 44
    assert roof.loc["parcel_north", "roof_spotlight_index"] == 96


def test_multiparcel_rollup_rsi_min_max_use_own_parcel_clip():
    """parcel_rollup: the RSI min/max/weighted-mean loop reads roof rows through
    the chunk-hoisted parent lookup, which must be scoped per parcel — each
    parcel's rollup row reflects its own clip of a shared multiparcel roof."""
    features = _make_chunk(
        [
            _roof_row("parcel_south", _SOUTH_GEOM, 15, 0.57, 691.0),
            _roof_row("parcel_north", _NORTH_GEOM, 96, 0.92, 455.0),
        ]
    )
    parcels_gdf = gpd.GeoDataFrame(
        {"geometry": [_SOUTH_GEOM, _NORTH_GEOM]},
        index=pd.Index(["parcel_south", "parcel_north"], name=AOI_ID_COLUMN_NAME),
        crs="EPSG:4326",
    )
    classes_df = pd.DataFrame({"id": [ROOF_ID], "description": ["roof"]}).set_index("id")

    rollup = parcel_rollup(
        parcels_gdf,
        features,
        classes_df,
        country="us",
        primary_decision="largest_intersection",
    )

    assert rollup.loc["parcel_south", "roof_spotlight_index_min"] == 15
    assert rollup.loc["parcel_south", "roof_spotlight_index_max"] == 15
    assert rollup.loc["parcel_north", "roof_spotlight_index_min"] == 96
    assert rollup.loc["parcel_north", "roof_spotlight_index_max"] == 96
    # Primary-roof RSI resolves from each parcel's own primary roof row
    assert rollup.loc["parcel_south", "primary_roof_spotlight_index"] == 15
    assert rollup.loc["parcel_north", "primary_roof_spotlight_index"] == 96
