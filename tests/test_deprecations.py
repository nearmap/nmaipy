"""Deprecated names keep working, with a warning, until 6.0."""

import warnings

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from nmaipy.api_common import GriddedApiClient
from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    AREA_CRS,
    MAX_AOI_AREA_SQM_BEFORE_GRIDDING,
    ROOF_ID,
    VEG_WOODY_COMPOSITE_ID,
)
from nmaipy.coverage_utils import get_survey_resource_id_from_survey_id_query
from nmaipy.feature_api import FeatureApi
from nmaipy.geometry_utils import combine_features_from_grid
from nmaipy.parcels import build_parent_lookup_by_aoi


def test_connected_class_ids_still_imports_with_a_warning():
    with pytest.warns(DeprecationWarning, match="CONNECTED_CLASS_IDS is deprecated"):
        from nmaipy.constants import CONNECTED_CLASS_IDS  # noqa: PLC0415 -- must run inside pytest.warns
    assert VEG_WOODY_COMPOSITE_ID in CONNECTED_CLASS_IDS
    assert ROOF_ID not in CONNECTED_CLASS_IDS


def test_unknown_constant_still_raises():
    with pytest.raises(ImportError):
        from nmaipy.constants import NOT_A_CONSTANT  # noqa: F401, PLC0415 -- the import is the statement under test


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


def test_build_parent_lookup_by_aoi_still_holds_every_row_with_a_warning():
    frame = pd.DataFrame(
        {AOI_ID_COLUMN_NAME: ["aoi1", "aoi1", "aoi2", "aoi2"], "feature_id": ["roof", "tree", "pool", None]}
    )
    with pytest.warns(DeprecationWarning, match="build_parent_lookup_by_aoi is internal, deprecated"):
        lookup = build_parent_lookup_by_aoi(frame)
    rows = {aoi: set(by_fid) for aoi, by_fid in lookup.items()}
    assert rows == {
        "aoi1": {"roof", "tree"},
        "aoi2": {"pool"},
    }, "The deprecated name should keep every row with a feature_id"


def test_feature_api_builds_without_warnings_and_keeps_its_deprecated_gridding_surfaces():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        api = FeatureApi(api_key="test-key", grid_size=0.01)
    with pytest.warns(DeprecationWarning, match="should_grid_aoi is deprecated"):
        assert api.should_grid_aoi(_us_square(2000.0), "us"), "A 2 km square is above the gridding threshold"
    with pytest.warns(DeprecationWarning, match="should_grid_aoi is deprecated"):
        assert not api.should_grid_aoi(_us_square(10.0), "us"), "A 10 m square is below the gridding threshold"
    with pytest.warns(DeprecationWarning, match="grid_cell_size is deprecated"):
        assert api.grid_cell_size is None
    with pytest.warns(DeprecationWarning, match="max_aoi_area_sqm is deprecated"):
        assert api.max_aoi_area_sqm == MAX_AOI_AREA_SQM_BEFORE_GRIDDING


def test_gridded_api_client_deprecated_arguments_attributes_and_helpers_still_work():
    with pytest.warns(DeprecationWarning, match="will be removed in nmaipy 6.0") as record:
        client = GriddedApiClient(api_key="test-key", grid_cell_size=0.004, max_aoi_area_sqm=10.0)
    messages = [str(w.message) for w in record if issubclass(w.category, DeprecationWarning)]
    assert sum("grid_cell_size is deprecated" in m for m in messages) == 1, messages
    assert sum("max_aoi_area_sqm is deprecated" in m for m in messages) == 1, messages
    with pytest.warns(DeprecationWarning, match="grid_cell_size is deprecated"):
        assert client.grid_cell_size == 0.004
    with pytest.warns(DeprecationWarning, match="max_aoi_area_sqm is deprecated"):
        assert client.max_aoi_area_sqm == 10.0

    square = _us_square(10.0)
    with pytest.warns(DeprecationWarning, match="should_grid_aoi is deprecated"):
        assert client.should_grid_aoi(square, "us"), "100 sqm is above a 10 sqm threshold"
    with pytest.warns(DeprecationWarning, match="max_aoi_area_sqm is deprecated"):
        client.max_aoi_area_sqm = 1000.0
    with pytest.warns(DeprecationWarning, match="should_grid_aoi is deprecated"):
        assert not client.should_grid_aoi(square, "us"), "Setting max_aoi_area_sqm should change the threshold"

    with pytest.warns(DeprecationWarning, match="get_survey_resource_id_from_survey_id_query is deprecated"):
        assert get_survey_resource_id_from_survey_id_query([{"id": "r1"}]) == "r1"


def _us_square(side_m: float) -> BaseGeometry:
    centre = gpd.GeoSeries([Point(-74.29, 40.63)], crs=API_CRS).to_crs(AREA_CRS["us"]).iloc[0]
    return gpd.GeoSeries([centre.buffer(side_m / 2, cap_style="square")], crs=AREA_CRS["us"]).to_crs(API_CRS).iloc[0]
