"""Roof objects are reported on the roof the API names as their parent."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from nmaipy.constants import AC_CONDENSER_UNIT_ID as AC_ID
from nmaipy.constants import AOI_ID_COLUMN_NAME, ROOF_ID, ROOF_OBJECT_CLASS_IDS, VENT_ID
from nmaipy.feature_attributes import flatten_roof_attributes
from nmaipy.parcels import feature_attributes

CLASSES = pd.DataFrame({"description": ["Roof", "A/C Condenser Unit", "Vent"]}, index=[ROOF_ID, AC_ID, VENT_ID])


def _row(class_id, geom, area, feature_id=None, parent_id=None, confidence=0.9):
    return {
        "class_id": class_id,
        "feature_id": feature_id or f"f{geom.bounds}",
        "confidence": confidence,
        "fidelity": 0.9,
        "parent_id": parent_id,
        "geometry": geom,
        "attributes": [],
        **{f"{kind}area_{unit}": area for kind in ("", "clipped_", "unclipped_") for unit in ("sqm", "sqft")},
    }


ROWS = [
    _row(ROOF_ID, box(0, 0, 10, 10), 100, feature_id="main"),  # primary roof
    _row(ROOF_ID, box(20, 0, 25, 5), 25, feature_id="shed"),
    _row(AC_ID, box(4, 4, 5, 5), 7, parent_id="main", confidence=0.6),
    _row(AC_ID, box(6, 6, 7, 7), 8, parent_id="main", confidence=0.8),
    _row(AC_ID, box(21, 1, 22, 2), 9, parent_id="shed"),
    _row(AC_ID, box(40, 40, 41, 41), 1),  # no roof parent
]
FEATURES = gpd.GeoDataFrame(ROWS, index=pd.Index(["a"] * len(ROWS), name=AOI_ID_COLUMN_NAME), crs="EPSG:4326")
ROOF_OBJECTS = {c: d for c, d in CLASSES.description.items() if c in ROOF_OBJECT_CLASS_IDS}


def test_rollup_reports_objects_on_the_primary_roof():
    result = feature_attributes(
        FEATURES, CLASSES, country="us", parcel_geom=None, primary_decision="largest_intersection"
    )
    assert result["a/c_condenser_unit_count"] == 4
    assert result["primary_roof_a/c_condenser_unit_present"] == "Y"
    assert result["primary_roof_a/c_condenser_unit_count"] == 2
    assert result["primary_roof_a/c_condenser_unit_area_sqft"] == 15
    assert result["primary_roof_a/c_condenser_unit_confidence"] == 0.8
    assert result["primary_roof_vent_present"] == "N"
    assert result["primary_roof_vent_count"] == 0
    assert result["primary_roof_vent_area_sqft"] == 0
    assert result["primary_roof_vent_confidence"] is None


def test_each_roof_reports_its_own_objects():
    children = FEATURES[FEATURES.class_id != ROOF_ID]
    shed = flatten_roof_attributes([ROWS[1]], "us", child_features=children, roof_object_classes=ROOF_OBJECTS)
    assert shed["a/c_condenser_unit_count"] == 1
    assert shed["a/c_condenser_unit_area_sqft"] == 9
    assert "a/c_condenser_unit_count" not in flatten_roof_attributes([ROWS[1]], "us", child_features=children)
