"""The per-class Building file's own 3D columns must belong to each row's building.

A chunk mixes classes, so the Building rows keep the chunk's index labels rather than 0..n-1. The rollup and the
per-class Roof file were never affected.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from nmaipy.constants import AOI_ID_COLUMN_NAME, BUILDING_NEW_ID, VEG_MEDHIGH_ID
from nmaipy.exporter import _compute_all_per_class_data

# feature_id: (aoi_id, has3dAttributes, height in m, pitch in degrees)
BUILDINGS = {
    "b1": ("aoi_a", True, 6.0, 20.0),
    "b2": ("aoi_a", False, None, None),
    "b3": ("aoi_b", True, 12.0, 35.0),
}


def _row(feature_id, aoi_id, class_id, x, **extra):
    return {
        AOI_ID_COLUMN_NAME: aoi_id,
        "feature_id": feature_id,
        "class_id": class_id,
        "description": "Building" if class_id == BUILDING_NEW_ID else "Medium and High Vegetation (>2m)",
        "confidence": 0.9,
        "parent_id": None,
        "is_primary": False,
        "geometry": box(x, 0, x + 0.5, 0.5),
        "area_sqm": 100.0,
        "clipped_area_sqm": 100.0,
        "unclipped_area_sqm": 100.0,
        "attributes": [],
        "survey_date": "2026-03-28",
        "mesh_date": "",
        "belongs_to_parcel": True,
        **extra,
    }


def _chunk(order, indexed):
    rows = []
    for i, fid in enumerate(order):
        aoi_id, has_3d, height, pitch = BUILDINGS[fid]
        rows.append(_row(f"tree-{fid}", aoi_id, VEG_MEDHIGH_ID, i))
        building_3d = {
            "Building 3d attributes.has3dAttributes": has_3d,
            "Building 3d attributes.height": height,
            "Building pitch.value": pitch,
        }
        rows.append(_row(fid, aoi_id, BUILDING_NEW_ID, i, **building_3d))
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return gdf.set_index(AOI_ID_COLUMN_NAME) if indexed else gdf


@pytest.mark.parametrize("order", [["b1", "b2", "b3"], ["b3", "b1", "b2"]], ids=["order_1", "order_2"])
@pytest.mark.parametrize("indexed", [False, True], ids=["aoi_column", "aoi_index"])
def test_building_3d_columns_belong_to_their_building(order, indexed):
    res = _compute_all_per_class_data(
        _chunk(order, indexed), country="au", aoi_input_columns=[], whitelisted_classes=[BUILDING_NEW_ID], threads=1
    )
    bldg = res[BUILDING_NEW_ID]["tabular"].to_pandas().set_index("feature_id")
    for fid, (_, has_3d, height, pitch) in BUILDINGS.items():
        row = bldg.loc[fid]
        assert row["has_3d_attributes"] == ("Y" if has_3d else "N"), f"{fid}: has_3d_attributes"
        assert pd.isna(row["height_m"]) if height is None else row["height_m"] == height, f"{fid}: height_m"
        assert pd.isna(row["pitch_degrees"]) if pitch is None else row["pitch_degrees"] == pitch, f"{fid}: pitch"
