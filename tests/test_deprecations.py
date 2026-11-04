"""Names deprecated in 5.4.0 keep working, with a warning, until 6.0."""

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS, ROOF_ID, VEG_WOODY_COMPOSITE_ID
from nmaipy.geometry_utils import combine_features_from_grid


def test_connected_class_ids_still_imports_with_a_warning():
    with pytest.warns(DeprecationWarning, match="CONNECTED_CLASS_IDS is deprecated"):
        from nmaipy.constants import CONNECTED_CLASS_IDS
    assert VEG_WOODY_COMPOSITE_ID in CONNECTED_CLASS_IDS
    assert ROOF_ID not in CONNECTED_CLASS_IDS


def test_unknown_constant_still_raises():
    with pytest.raises(ImportError):
        from nmaipy.constants import NOT_A_CONSTANT  # noqa: F401


def test_connected_class_ids_argument_is_accepted_with_a_warning_and_ignored():
    features = gpd.GeoDataFrame(
        {
            "feature_id": ["roof", "roof"],
            "geometry": [Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])] * 2,
            AOI_ID_COLUMN_NAME: ["aoi1", "aoi1"],
            "area_sqm": [100.0, 100.0],
            "clipped_area_sqm": [60.0, 40.0],
            "unclipped_area_sqm": [100.0, 100.0],
        },
        crs=API_CRS,
    )
    with pytest.warns(DeprecationWarning, match="connected_class_ids is deprecated"):
        merged = combine_features_from_grid(features, connected_class_ids=[ROOF_ID])
    assert merged["area_sqm"].tolist() == [100.0], "The argument should not change the merge"
