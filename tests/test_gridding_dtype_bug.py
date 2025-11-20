"""
Test for gridding aoi_id dtype mismatch bug.

When an AOI is gridded and some grid cells fail, the merge operation between
errors_df and df_gridded can fail with a dtype mismatch error if the aoi_id
column types don't match.
"""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon
from shapely.wkt import loads

from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS
from nmaipy.feature_api import FeatureApi


class TestGriddingDtypeBug:
    def test_gridded_aoi_with_integer_id_dtype_consistency(self, cache_directory):
        """
        Test that when an AOI with integer ID is gridded and errors occur,
        the merge operation succeeds without dtype mismatch.

        This reproduces the bug where:
        - Original AOI has integer aoi_id (e.g., 662)
        - Gridding creates temp integer IDs for grid cells
        - Some grid cells fail, creating errors
        - Merge fails with: "You are trying to merge on object and int64 columns for key 'aoi_id'"
        """
        # WKT from user's failing case - a large AOI in Australia that triggers gridding
        wkt = "POLYGON ((153.0582493406651281 -27.54883129891485893, 153.06032991029584878 -27.54997966913306939, 153.06028451026904236 -27.55023015913047857, 153.06025480024922558 -27.55041191912893339, 153.06020789022386452 -27.55065232912610895, 153.05934462971495691 -27.55540872907689831, 153.05857893926673796 -27.55960230903303554, 153.05849275924052222 -27.55988278902660582, 153.05710885941621768 -27.55967242888583968, 153.05568951959475044 -27.55946969874158015, 153.05464346972635781 -27.55932027863527622, 153.05363310985345038 -27.55917597853260048, 153.05263512997896669 -27.55903343843118236, 153.05217402003722782 -27.55896544838430984, 153.05069402022098757 -27.55877341823407534, 153.04724331065105503 -27.5583120478837067, 153.04806241116173737 -27.55357682792861596, 153.04715473127524206 -27.5534523878364439, 153.0447990215697871 -27.55312943759722089, 153.04161512196790795 -27.55269292727393804, 153.04151369198029897 -27.55268116726365335, 153.04178510211770003 -27.55136344728051156, 153.04181784216120832 -27.55099148728086433, 153.04200111214950653 -27.55092784729877309, 153.04291912258150887 -27.54672943735783264, 153.04317221254996184 -27.54676354738352373, 153.05007987169543071 -27.54763968808445185, 153.0582493406651281 -27.54883129891485893))"
        geom = loads(wkt)

        # Integer aoi_id like in the failing case
        aoi_id = 662

        # Create GeoDataFrame with integer index
        aoi_gdf = gpd.GeoDataFrame(
            [{"geometry": geom, AOI_ID_COLUMN_NAME: aoi_id}],
            crs=API_CRS
        )
        aoi_gdf = aoi_gdf.set_index(AOI_ID_COLUMN_NAME)

        # Use conditions that trigger gridding and might cause errors
        feature_api = FeatureApi(
            cache_dir=cache_directory,
            only3d=True,
            aoi_grid_min_pct=75  # Allow some grid cells to fail
        )

        try:
            # This should not raise a dtype mismatch error
            features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(
                gdf=aoi_gdf,
                region="au",
                packs=["building"],
                since_bulk="2025-06-01",
                until_bulk="2025-09-01",
                max_allowed_error_pct=100
            )

            # If we got here without an exception, the bug is fixed!
            # Check that data structures are consistent
            if len(errors_df) > 0:
                assert AOI_ID_COLUMN_NAME in errors_df.columns
                # aoi_id should be the original integer, not grid cell IDs
                assert (errors_df[AOI_ID_COLUMN_NAME] == aoi_id).all()

            # features_gdf and metadata_df should also have consistent aoi_id
            if len(features_gdf) > 0:
                assert (features_gdf.index == aoi_id).all()
            if len(metadata_df) > 0:
                assert (metadata_df.index == aoi_id).all()

        except ValueError as e:
            if "merge on object and int64" in str(e):
                pytest.fail(f"Dtype mismatch bug still present: {e}")
            else:
                # Some other ValueError - let it propagate
                raise

    def test_gridded_errors_have_correct_aoi_id_dtype(self, cache_directory):
        """
        Test that ensures error DataFrame from gridded operations has correct aoi_id dtype
        matching the original AOI's dtype.
        """
        # Test with string aoi_id
        wkt = "POLYGON ((153.0582493406651281 -27.54883129891485893, 153.06032991029584878 -27.54997966913306939, 153.06028451026904236 -27.55023015913047857, 153.06025480024922558 -27.55041191912893339, 153.06020789022386452 -27.55065232912610895, 153.05934462971495691 -27.55540872907689831, 153.05857893926673796 -27.55960230903303554, 153.05849275924052222 -27.55988278902660582, 153.05710885941621768 -27.55967242888583968, 153.05568951959475044 -27.55946969874158015, 153.05464346972635781 -27.55932027863527622, 153.05363310985345038 -27.55917597853260048, 153.05263512997896669 -27.55903343843118236, 153.05217402003722782 -27.55896544838430984, 153.05069402022098757 -27.55877341823407534, 153.04724331065105503 -27.5583120478837067, 153.04806241116173737 -27.55357682792861596, 153.04715473127524206 -27.5534523878364439, 153.0447990215697871 -27.55312943759722089, 153.04161512196790795 -27.55269292727393804, 153.04151369198029897 -27.55268116726365335, 153.04178510211770003 -27.55136344728051156, 153.04181784216120832 -27.55099148728086433, 153.04200111214950653 -27.55092784729877309, 153.04291912258150887 -27.54672943735783264, 153.04317221254996184 -27.54676354738352373, 153.05007987169543071 -27.54763968808445185, 153.0582493406651281 -27.54883129891485893))"
        geom = loads(wkt)
        aoi_id = "string_id_662"

        aoi_gdf = gpd.GeoDataFrame(
            [{"geometry": geom, AOI_ID_COLUMN_NAME: aoi_id}],
            crs=API_CRS
        )
        aoi_gdf = aoi_gdf.set_index(AOI_ID_COLUMN_NAME)

        feature_api = FeatureApi(
            cache_dir=cache_directory,
            only3d=True,
            aoi_grid_min_pct=75
        )

        try:
            features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(
                gdf=aoi_gdf,
                region="au",
                packs=["building"],
                since_bulk="2025-06-01",
                until_bulk="2025-09-01",
                max_allowed_error_pct=100
            )

            # Check dtype consistency
            if len(errors_df) > 0:
                assert errors_df[AOI_ID_COLUMN_NAME].dtype == aoi_gdf.index.dtype
                assert (errors_df[AOI_ID_COLUMN_NAME] == aoi_id).all()

        except ValueError as e:
            if "merge on object and int64" in str(e):
                pytest.fail(f"Dtype mismatch bug still present: {e}")
            else:
                raise
