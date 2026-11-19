"""Roof objects are counted on roofs, and on the primary roof, as well as across the parcel."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from nmaipy.constants import AOI_ID_COLUMN_NAME, ROOF_ID, ROOF_OBJECT_CLASS_IDS
from nmaipy.parcels import class_column_names, feature_attributes

AC_ID = "ed1ea29d-cd2c-55a9-8737-13018a9bb741"
CLASSES = pd.DataFrame({"description": ["Roof", "A/C Condenser Unit"]}, index=[ROOF_ID, AC_ID])


def _row(class_id, geom, area, feature_id=None, parent_id=None):
    return {
        "class_id": class_id,
        "feature_id": feature_id or f"f{geom.bounds}",
        "confidence": 0.9,
        "fidelity": 0.9,
        "parent_id": parent_id,
        "geometry": geom,
        "attributes": [],
        **{f"{kind}area_{unit}": area for kind in ("", "clipped_", "unclipped_") for unit in ("sqm", "sqft")},
    }


def _rollup(rows):
    features = gpd.GeoDataFrame(rows, index=pd.Index(["a"] * len(rows), name=AOI_ID_COLUMN_NAME), crs="EPSG:4326")
    return feature_attributes(
        features, CLASSES, country="us", parcel_geom=None, primary_decision="largest_intersection"
    )


def test_units_are_counted_on_their_parent_roof():
    result = _rollup(
        [
            _row(ROOF_ID, box(0, 0, 10, 10), 100, feature_id="main"),  # primary roof
            _row(ROOF_ID, box(20, 0, 25, 5), 25, feature_id="shed"),
            _row(AC_ID, box(4, 4, 5, 5), 1, parent_id="main"),
            _row(AC_ID, box(21, 1, 22, 2), 1, parent_id="shed"),
            _row(AC_ID, box(40, 40, 41, 41), 1, parent_id="not-a-roof-on-this-parcel"),
            _row(AC_ID, box(50, 50, 51, 51), 1),
        ]
    )
    assert result["a/c_condenser_unit_count"] == 4
    assert result["a/c_condenser_unit_on_roof_count"] == 2
    assert result["primary_roof_a/c_condenser_unit_count"] == 1


def test_no_roof_means_nothing_is_on_a_roof():
    result = _rollup([_row(AC_ID, box(4, 4, 5, 5), 1, parent_id="main")])
    assert result["a/c_condenser_unit_count"] == 1
    assert result["a/c_condenser_unit_on_roof_count"] == 0
    assert result["primary_roof_a/c_condenser_unit_count"] == 0


def test_columns_are_nulled_with_the_rest_of_the_class():
    assert AC_ID in ROOF_OBJECT_CLASS_IDS
    assert {"a/c_condenser_unit_on_roof_count", "primary_roof_a/c_condenser_unit_count"} <= class_column_names(
        CLASSES, "us"
    )[AC_ID]
