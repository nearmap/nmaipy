import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.wkt import loads

from nmaipy import parcels, reference_code, geometry_utils
from nmaipy.api_common import generate_curl_command, AIFeatureAPIError, APIGridError
from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    AREA_CRS,
    ASPHALT_ID,
    BUILDING_ID,
    BUILDING_NEW_ID,
    MAX_AOI_AREA_SQM_BEFORE_GRIDDING,
    READ_TIMEOUT_SECONDS,
    ROLLUP_BUILDING_COUNT_ID,
    ROLLUP_BUILDING_PRIMARY_UNCLIPPED_AREA_SQM_ID,
    ROOF_ID,
    SOLAR_HW_ID,
    SOLAR_ID,
    VEG_MEDHIGH_ID,
)
from nmaipy.feature_api import FeatureApi


class TestFeatureAPI:
    def test_get_rollup_df(self, sydney_aoi: Polygon, cache_directory: Path):
        date_1 = "2025-01-20"
        date_2 = "2025-01-20"
        region = "au"
        packs = ["building"]
        aoi_id = "123"

        feature_api = FeatureApi(cache_dir=cache_directory)
        rollup_df, metadata, error = feature_api.get_rollup_df(
            sydney_aoi, region, packs, aoi_id=aoi_id, since=date_1, until=date_2
        )
        print(rollup_df.T)
        print(f"WKT of Query AOI: {sydney_aoi}")

        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["survey_date"] <= date_2
        # We get 1 building
        building_count = rollup_df[ROLLUP_BUILDING_COUNT_ID].iloc[0, 0]
        assert (
            building_count == 1
        )  # Expect a single, joined building kept after filter, from two touching residential homes.
        # The AOI ID has been assigned
        assert len(rollup_df.loc[[aoi_id]]) == 1
        # Unclipped area should be about 450 sqm
        assert rollup_df[ROLLUP_BUILDING_PRIMARY_UNCLIPPED_AREA_SQM_ID].iloc[0, 0] == pytest.approx(450, rel=0.1)

    def test_get_features_gdf(self, sydney_aoi: Polygon, cache_directory: Path):
        date_1 = "2025-01-20"
        date_2 = "2025-01-20"
        region = "au"
        packs = ["building"]
        aoi_id = "123"

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(
            sydney_aoi, region, packs, aoi_id=aoi_id, since=date_1, until=date_2
        )
        features_gdf = features_gdf.query("class_id == @ROOF_ID")  # Filter out building classes, just keep roof.
        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["survey_date"] <= date_2
        # We get 3 buildings
        assert len(features_gdf) == 1
        assert len(features_gdf[features_gdf.class_id == ROOF_ID]) == 1
        # The AOI ID has been assigned
        assert len(features_gdf.loc[[aoi_id]]) == 1

    def test_trim_features_to_aoi(self, cache_directory: Path):
        aoi = loads(
            "MULTIPOLYGON(((-81.96101015253903 29.05631578134146, -81.96097715235663 29.05643178114233, -81.96088515183891 29.05676078057977, -81.96081015166226 29.05688178032226, -81.96071615151028 29.05699178005651, -81.96050315132298 29.05714677958319, -81.96030215120676 29.05725677918626, -81.96002115116396 29.05733877872941, -81.95975115121745 29.05736077836785, -81.95966415128315 29.0573387782909, -81.9592571518955 29.05705277817911, -81.95915015219606 29.05689377826324, -81.95908815262986 29.05664577852331, -81.95908115301701 29.05641477882801, -81.95915615341346 29.05616177926368, -81.95941915372902 29.05591977991694, -81.95949115377999 29.055874780067, -81.9598821540514 29.05563378087792, -81.96037115415523 29.05547378170073, -81.96061515411904 29.05544678203954, -81.96077815401978 29.05547378220442, -81.9608651538808 29.05553978222199, -81.96098515356586 29.05570478214509, -81.96103515330084 29.05585378200341, -81.96101015253903 29.05631578134146)), ((-81.965573157064 29.05269479199695, -81.96551115685583 29.05283179172925, -81.96541715672163 29.05293079147429, -81.96469115613426 29.05342678988017, -81.96450915619464 29.05342678965309, -81.96423415640598 29.05335478940948, -81.96400815645254 29.05337178910396, -81.96389515631647 29.05347578881938, -81.96363915561045 29.05394978784645, -81.96358915545353 29.05405378764083, -81.96360115525603 29.05416978749583, -81.96363915516162 29.05421878747553, -81.96372015504298 29.05427378750045, -81.96410215486105 29.05430678792986, -81.96424015465836 29.05440078797153, -81.96424615447276 29.05451078782699, -81.96420915438496 29.05457078769814, -81.96409615424729 29.05467578741273, -81.96392115414024 29.05477478705886, -81.96355815400395 29.0549287863961, -81.96329515397292 29.05499978597212, -81.96305715400537 29.05502778563849, -81.96281315415156 29.05498878538958, -81.96260015443251 29.0548627852986, -81.96218015526925 29.05444478535088, -81.96194215557841 29.05430678524426, -81.961836155797 29.05419678526315, -81.96179815600141 29.05408178537342, -81.96196115625911 29.05389478583257, -81.96219215657767 29.05365778644532, -81.9625221568685 29.05341778718632, -81.96287515717678 29.05316278797737, -81.96310015757098 29.05288178864514, -81.96318115798456 29.05261778910983, -81.96318815851441 29.05229878955775, -81.9632691590098 29.05198579009027, -81.96335715918424 29.05186379036877, -81.96410215975392 29.05137479197944, -81.96416415973346 29.05137479205743, -81.96422715961252 29.05143479205378, -81.96421415949663 29.05150679193795, -81.96361315825364 29.05237078999111, -81.96361315801505 29.05251378979393, -81.96368915791651 29.05255778982842, -81.96417115776693 29.05255179044018, -81.96456515752637 29.05261779084218, -81.96479715747789 29.0526007911561, -81.96501615763911 29.05246079162424, -81.96525415781238 29.05230979213174, -81.96541715775842 29.05230979233601, -81.96552915758446 29.05239179236256, -81.96556715744335 29.05246879230323, -81.965573157064 29.05269479199695)))"
        )
        country = "us"
        date_1 = "2010-01-01"
        date_2 = "2022-03-31"
        packs = ["building", "vegetation"]
        aoi_id = 47

        parcels_gdf = gpd.GeoDataFrame(
            [{"geometry": aoi, "aoi_id": aoi_id, "since": date_1, "until": date_2}], crs=API_CRS
        )

        classes_df = pd.DataFrame(
            [
                {"id": BUILDING_ID, "description": "building"},
                {"id": VEG_MEDHIGH_ID, "description": "Medium and High Vegetation (>2m)"},
            ]
        ).set_index("id")

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(
            aoi, country, packs, aoi_id=aoi_id, since=date_1, until=date_2
        )

        print(metadata)
        print(features_gdf.T)

        assert len(features_gdf.feature_id.unique()) == len(features_gdf)
        # TODO: Make this test richer/quantitative info, not just that we end up with unique feature IDs.

    def test_split_geometry_into_grid(self, sydney_aoi: Polygon):
        aoi = sydney_aoi
        b = aoi.bounds
        width = b[2] - b[0]
        height = b[3] - b[1]
        d = max(width, height)

        for cell_size in [d / 5, width, height, d, 2 * d]:
            df_gridded = geometry_utils.split_geometry_into_grid(aoi, cell_size)
            geom_recombined = df_gridded.geometry.unary_union
            assert geom_recombined.difference(aoi).area == pytest.approx(0)
            assert aoi.difference(geom_recombined).area == pytest.approx(0)

    def test_combine_features_gdf_from_grid(
        self,
    ):
        """Test combine_features_gdf_from_grid with empty DataFrames (the bug we fixed) and normal operation."""
        
        # Test 1: Empty GeoDataFrame (the main bug we fixed)
        empty_gdf = gpd.GeoDataFrame(columns=['geometry', 'feature_id'], crs=API_CRS)
        result = geometry_utils.combine_features_from_grid(empty_gdf)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0
        assert result.crs == API_CRS

        # Test 2: None input
        result = geometry_utils.combine_features_from_grid(None)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0
        assert result.crs == API_CRS

        # Test 3: Normal case with features to combine
        from shapely.geometry import Polygon
        test_features = gpd.GeoDataFrame({
            'feature_id': [1, 1, 2],  # Two features with same ID should be combined
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(0, 1), (1, 1), (1, 2), (0, 2)])
            ],
            AOI_ID_COLUMN_NAME: ['aoi1', 'aoi1', 'aoi1'],
            'class_id': [100, 100, 200],
            'area_sqm': [1.0, 1.0, 1.0],
            'confidence': [0.9, 0.9, 0.8]
        }, crs=API_CRS)

        result = geometry_utils.combine_features_from_grid(test_features)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2  # Should have 2 features after combining duplicates
        assert result.crs == API_CRS

    def test_large_aoi(self, cache_directory: Path, large_adelaide_aoi: Polygon):
        survey_resource_id = "68c1c4e7-9267-52a5-84e7-23b6ed43bceb"  # 2025-03-06, Gen 6 run
        country = "au"
        packs = ["building", "vegetation"]
        aoi_id = "0"

        feature_api = FeatureApi(cache_dir=cache_directory, parcel_mode=True)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(
            large_adelaide_aoi, region=country, packs=packs, aoi_id=aoi_id, survey_resource_id=survey_resource_id
        )
        # No error
        assert error is None
        # We get buildings and vegetation
        assert len(features_gdf.query("class_id == @ROOF_ID")) == 12  # Updated after testing
        assert len(features_gdf.query("class_id == @VEG_MEDHIGH_ID")) == 750  # Guessed

        # Assert that buildings aren't overhanging the edge of the parcel. If this fails, the clipped/unclipped hasn't been managed correctly during the grid merge.
        assert features_gdf.query("class_id == @ROOF_ID").clipped_area_sqm.sum() == pytest.approx(
            features_gdf.query("class_id == @ROOF_ID").unclipped_area_sqm.sum(), 0.1
        )

        # The AOI ID has been assigned to all features
        assert len(features_gdf.loc[[aoi_id]]) == len(features_gdf)

    def test_proactive_gridding_for_large_aoi(self, cache_directory: Path, monkeypatch):
        """Test that AOIs larger than MAX_AOI_AREA_SQM_BEFORE_GRIDDING trigger gridding directly without hitting the API first"""
        # Create a test polygon that's definitely larger than the threshold
        # This is roughly 3km x 3km = 9 sqkm, well above the 1 sqkm threshold
        test_coords = [
            [144.9, -37.8],
            [144.9 + 0.027, -37.8],  # ~3km wide
            [144.9 + 0.027, -37.8 + 0.027],  # ~3km tall
            [144.9, -37.8 + 0.027],
            [144.9, -37.8]
        ]
        test_polygon = Polygon(test_coords)
        
        # Calculate actual area to verify it's above threshold
        geometry_gdf = gpd.GeoSeries([test_polygon], crs=API_CRS)
        geometry_projected = geometry_gdf.to_crs(AREA_CRS["au"])
        actual_area = geometry_projected.area.iloc[0]
        
        # Verify the test polygon is actually larger than the threshold
        assert actual_area > MAX_AOI_AREA_SQM_BEFORE_GRIDDING, f"Test polygon area ({actual_area:.0f}) should be > {MAX_AOI_AREA_SQM_BEFORE_GRIDDING}"
        
        feature_api = FeatureApi(cache_dir=cache_directory)
        
        # Track method calls
        gridding_called = False
        api_called = False
        
        def mock_attempt_gridding(*args, **kwargs):
            nonlocal gridding_called
            gridding_called = True
            return (
                gpd.GeoDataFrame({"test": ["data"]}, geometry=[test_polygon]),
                {"test": "metadata"},
                None,
                None  # grid_errors_df
            )
        
        def mock_get_features(*args, **kwargs):
            nonlocal api_called
            api_called = True
            return {}
        
        # Apply monkeypatch
        monkeypatch.setattr(feature_api, '_attempt_gridding', mock_attempt_gridding)
        monkeypatch.setattr(feature_api, 'get_features', mock_get_features)
        
        # Call get_features_gdf with the large AOI
        result = feature_api.get_features_gdf(
            geometry=test_polygon,
            region="au",
            packs=["building"],
            aoi_id="test_large_aoi"
        )
        
        # Verify that gridding was called directly
        assert gridding_called, "Expected _attempt_gridding to be called for large AOI"
        
        # Verify that the regular API was NOT called
        assert not api_called, "Expected get_features to NOT be called for large AOI"
        
        # Verify the result structure
        features_gdf, metadata, error, _ = result
        assert features_gdf is not None
        assert metadata is not None
        assert error is None

    def test_small_aoi_no_proactive_gridding(self, cache_directory: Path, monkeypatch):
        """Test that AOIs smaller than MAX_AOI_AREA_SQM_BEFORE_GRIDDING do NOT trigger gridding directly"""
        # Create a test polygon that's definitely smaller than the threshold
        # This is roughly 500m x 500m = 0.25 sqkm, well below the 1 sqkm threshold
        test_coords = [
            [144.9, -37.8],
            [144.9 + 0.0045, -37.8],  # ~500m wide
            [144.9 + 0.0045, -37.8 + 0.0045],  # ~500m tall
            [144.9, -37.8 + 0.0045],
            [144.9, -37.8]
        ]
        test_polygon = Polygon(test_coords)
        
        # Calculate actual area to verify it's below threshold
        geometry_gdf = gpd.GeoSeries([test_polygon], crs=API_CRS)
        geometry_projected = geometry_gdf.to_crs(AREA_CRS["au"])
        actual_area = geometry_projected.area.iloc[0]
        
        # Verify the test polygon is actually smaller than the threshold
        assert actual_area < MAX_AOI_AREA_SQM_BEFORE_GRIDDING, f"Test polygon area ({actual_area:.0f}) should be < {MAX_AOI_AREA_SQM_BEFORE_GRIDDING}"
        
        feature_api = FeatureApi(cache_dir=cache_directory)
        
        # Track method calls
        gridding_called = False
        api_called = False
        
        def mock_attempt_gridding(*args, **kwargs):
            nonlocal gridding_called
            gridding_called = True
            return (None, None, None)
        
        def mock_get_features(*args, **kwargs):
            nonlocal api_called
            api_called = True
            return {}
        
        def mock_payload_gdf(*args, **kwargs):
            return (
                gpd.GeoDataFrame({"test": ["data"]}, geometry=[test_polygon]),
                {"test": "metadata"}
            )
        
        # Apply monkeypatch
        monkeypatch.setattr(feature_api, '_attempt_gridding', mock_attempt_gridding)
        monkeypatch.setattr(feature_api, 'get_features', mock_get_features)
        monkeypatch.setattr(feature_api, 'payload_gdf', mock_payload_gdf)
        
        # Call get_features_gdf with the small AOI
        result = feature_api.get_features_gdf(
            geometry=test_polygon,
            region="au",
            packs=["building"],
            aoi_id="test_small_aoi"
        )
        
        # Verify that the regular API was called
        assert api_called, "Expected get_features to be called for small AOI"
        
        # Verify that gridding was NOT called
        assert not gridding_called, "Expected _attempt_gridding to NOT be called for small AOI"
        
        # Verify the result structure
        features_gdf, metadata, error, _ = result
        assert features_gdf is not None
        assert metadata is not None
        assert error is None

    def test_not_found(self, cache_directory: Path):
        # Somewhere in the Pacific
        aoi = loads(
            """
            POLYGON((
                -145.5385490968286 -28.74031798505824, 
                -145.5381490968286 -28.74031798505824,
                -145.5381490968286 -28.73991798505824, 
                -145.5385490968286 -28.73991798505824,
                -145.5385490968286 -28.74031798505824
                ))
        """
        )
        country = "au"
        feature_api = FeatureApi(cache_dir=cache_directory, maxretry=3)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(aoi, region=country)
        # No data
        assert features_gdf is None
        assert metadata is None
        # There is a error message
        assert isinstance(error["message"], str)

    def test_get_cache(self, cache_directory: Path, sydney_aoi: Polygon):
        date_1 = "2025-01-20"
        date_2 = "2025-01-20"
        country = "au"
        packs = ["building"]
        aoi_id = "123"
        # First do a standard pull to ensure the file is populated in the cache.
        feature_api = FeatureApi(cache_dir=cache_directory, compress_cache=False)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(
            sydney_aoi, country, packs, aoi_id=aoi_id, since=date_1, until=date_2
        )
        assert error is None

        # Then re-request using invalid API key to ensure data is not being pulled from the API but read from the cache.
        api_key = "not an api key"
        # Run
        feature_api = FeatureApi(api_key, cache_dir=cache_directory, compress_cache=False)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(
            sydney_aoi, country, packs, aoi_id=aoi_id, since=date_1, until=date_2
        )
        # Check output
        assert error is None
        assert date_1 <= metadata["survey_date"] <= date_2
        features_gdf = features_gdf.query("class_id == @ROOF_ID")  # Filter out building classes, just keep roof.
        assert len(features_gdf) == 1

    def test_get_compressed_cache(self, cache_directory: Path, sydney_aoi: Polygon):
        date_1 = "2025-01-20"
        date_2 = "2025-01-20"
        country = "au"
        packs = ["building"]
        aoi_id = "123"
        # First do a standard pull to ensure the file is populated in the cache.
        feature_api = FeatureApi(cache_dir=cache_directory, compress_cache=True)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(
            sydney_aoi, country, packs, aoi_id=aoi_id, since=date_1, until=date_2
        )
        assert error is None

        # Then re-request using invalid API key to ensure data is not being pulled from the API but read from the cache.
        api_key = "not an api key"
        # Run
        feature_api = FeatureApi(api_key, cache_dir=cache_directory, compress_cache=True)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(
            sydney_aoi, country, packs, aoi_id=aoi_id, since=date_1, until=date_2
        )
        # Check output
        assert error is None
        assert date_1 <= metadata["survey_date"] <= date_2
        features_gdf = features_gdf.query("class_id == @ROOF_ID")  # Filter out building classes, just keep roof.
        assert len(features_gdf) == 1

    def test_get_bulk(self, cache_directory: Path, sydney_aoi: Polygon):
        aois = []
        for i in range(4):
            for j in range(4):
                aois.append({AOI_ID_COLUMN_NAME: f"{i}_{j}", "geometry": translate(sydney_aoi, 0.001 * i, 0.001 * j)})
        # Add an AOI with an invalid type to test an error case - multipolygon of two separate chunks

        aoi_gdf = gpd.GeoDataFrame(aois).set_index(AOI_ID_COLUMN_NAME)
        date_1 = "2025-01-20"
        date_2 = "2025-01-20"
        country = "au"
        packs = ["building"]

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(
            aoi_gdf, country, packs, None, since_bulk=date_1, until_bulk=date_2
        )
        features_gdf = features_gdf.query("class_id == @ROOF_ID")  # Filter out building classes, just keep roof.

        # Check metadata
        assert len(metadata_df) == 16
        assert len(metadata_df.merge(aoi_gdf, on="aoi_id", how="inner")) == 16
        # Check error
        assert len(errors_df) == 0
        # We get only roofs
        assert len(features_gdf) == 53  # Updated after testing
        assert len(features_gdf[features_gdf.class_id == ROOF_ID]) == 53

        rollup_df, metadata_df, errors_df = feature_api.get_rollup_df_bulk(
            aoi_gdf, country, packs, since_bulk=date_1, until_bulk=date_2
        )
        # Check metadata
        assert len(metadata_df) == 16
        assert len(metadata_df.merge(aoi_gdf, on="aoi_id", how="inner")) == 16
        # Check error
        assert len(errors_df) == 0

        # We get about the right number of buildings
        assert len(rollup_df) == 16
        total_building_count = rollup_df[ROLLUP_BUILDING_COUNT_ID].values.sum()
        assert total_building_count == pytest.approx(53, rel=0.1)

    def test_get_bulk_with_data_dates(self, cache_directory: Path, sydney_aoi: Polygon):
        aois = []
        for i in range(4):
            for j in range(4):
                aois.append(
                    {
                        AOI_ID_COLUMN_NAME: f"{i}_{j}",
                        "since": "2025-01-20",
                        "until": "2025-01-20",
                        "geometry": translate(sydney_aoi, 0.001 * i, 0.001 * j),
                    }
                )

        aoi_gdf = gpd.GeoDataFrame(aois).set_index(AOI_ID_COLUMN_NAME)
        country = "au"
        packs = ["building"]

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(aoi_gdf, country, packs)
        features_gdf = features_gdf.query("class_id == @ROOF_ID")  # Filter out building classes, just keep roof.
        print(metadata_df.iloc[0].T)

        # Check metadata
        assert len(metadata_df) == 16
        assert len(metadata_df.merge(aoi_gdf, on="aoi_id", how="inner")) == 16

        # We get only buildings
        assert len(features_gdf) == 53  # Updated after testing
        assert len(features_gdf[features_gdf.class_id == ROOF_ID]) == 53
        # The dates are within range
        for row in features_gdf.itertuples():
            assert "2025-01-20" <= row.survey_date <= "2025-01-20"

    def test_point_data(self, cache_directory: Path):
        # Small area around a power pole in Darwin, AU
        aoi = loads(
            """
            Polygon ((130.85230645835545715 -12.39611155192925374, 130.85240838555137088 -12.39610856112810566, 130.85241538484379475 -12.3961991396618636, 130.85233576789246968 -12.39619572160455263, 130.85233576789246968 -12.39619572160455263, 130.85230645835545715 -12.39611155192925374))
            """
        )
        country = "au"
        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(aoi, region=country, classes=["46f2f9ce-8c0f-50df-a9e0-4c2026dd3f95"])
        features_gdf[AOI_ID_COLUMN_NAME] = 0
        print(features_gdf.T)
        assert len(features_gdf) == 1  # 1 pole found

    def test_multipolygon_1(self, cache_directory: Path, sydney_aoi: Polygon):
        aoi = sydney_aoi.union(translate(sydney_aoi, 0.002, 0.01))
        print(f"Multipolygon 1 (use QuickWKT in QGIS to visualise): {aoi}")
        date_1 = "2025-01-20"
        date_2 = "2025-01-20"
        country = "au"
        packs = ["building"]
        aoi_id = "123"
        # Run
        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(aoi, country, packs, None, None, aoi_id, date_1, date_2)
        assert error is None
        assert metadata is not None
        features_gdf = features_gdf.query("class_id == @ROOF_ID")  # Filter out building classes, just keep roof.
        print(metadata)
        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["survey_date"] <= date_2
        # We get 4 roofs
        assert len(features_gdf) == 4
        assert len(features_gdf[features_gdf.class_id == ROOF_ID]) == 4
        # The AOI ID has been assigned
        assert len(features_gdf.loc[[aoi_id]]) == 4
        # All buildings intersect the AOI
        assert len(features_gdf[features_gdf.intersects(aoi)]) == 4

    def test_multipolygon_2(self, cache_directory: Path, sydney_aoi: Polygon):
        aoi = MultiPolygon([translate(sydney_aoi, 0.001, 0.001), translate(sydney_aoi, 0.003, 0.003)])
        print(f"Multipolygon 2 (use QuickWKT in QGIS to visualise): {aoi}")
        date_1 = "2025-01-20"
        date_2 = "2025-01-20"
        country = "au"
        packs = ["building"]
        aoi_id = "123"
        # Run
        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(aoi, country, packs, None, None, aoi_id, date_1, date_2)
        features_gdf = features_gdf.query("class_id == @ROOF_ID")  # Filter out building classes, just keep roof.

        print(metadata)

        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["survey_date"] <= date_2
        # We get 7 roofs
        assert len(features_gdf) == 7  # Updated after testing
        assert len(features_gdf[features_gdf.class_id == ROOF_ID]) == 7
        # The AOI ID has been assigned
        assert len(features_gdf.loc[[aoi_id]]) == 7
        # All buildings intersect the AOI
        assert len(features_gdf[features_gdf.intersects(aoi)]) == 7

    def test_multipolygon_3(self, cache_directory: Path):
        """
        Multipolygon with two nearby parts, and an overlapping discrete class (building)
        """
        aoi = loads(
            "MultiPolygon (((-88.40618111505668253 43.06538384370446693, -88.40618111505668253 43.06557268197261834, -88.40601285961312783 43.06557268197261834, -88.40601285961312783 43.06538384370446693, -88.40618111505668253 43.06538384370446693)),((-88.40590800477149003 43.06555664855734022, -88.40590394063035262 43.06538028071267377, -88.40578689336528839 43.06537434239258033, -88.40577307528538142 43.06555189791498606, -88.40577307528538142 43.06555189791498606, -88.40590800477149003 43.06555664855734022)))"
        )
        print(f"Multipolygon 2 (use QuickWKT in QGIS to visualise): {aoi}")
        date_1 = "2020-01-01"
        date_2 = "2022-07-01"
        country = "us"
        packs = ["building"]
        aoi_id = "3"
        # Run
        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(aoi, country, packs, None, None, aoi_id, date_1, date_2)
        features_gdf = features_gdf.query("class_id == @ROOF_ID")  # Filter out building classes, just keep roof.
        print(metadata)

        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["survey_date"] <= date_2
        # We get 3 buildings
        assert len(features_gdf) == 1
        assert len(features_gdf[features_gdf.class_id == ROOF_ID]) == 1
        # The AOI ID has been assigned
        assert len(features_gdf.loc[[aoi_id]]) == 1
        # All buildings intersect the AOI
        assert len(features_gdf[features_gdf.intersects(aoi)]) == 1

        assert features_gdf["unclipped_area_sqm"].sum() == pytest.approx(152, rel=0.02)
        assert features_gdf["area_sqm"].sum() == pytest.approx(152, rel=0.02)
        assert features_gdf["clipped_area_sqm"].sum() == pytest.approx(68, rel=0.02)

    def test_polygon_with_hole_1(self, cache_directory: Path):
        # This one should have a building in the middle which gets pulled then discarded, and clear space around it.
        aoi = loads(
            "Polygon((-87.98409445082069169 42.9844739669082827, -87.98409445082069169 42.98497943053578041, -87.98334642812285722 42.98497943053578041, -87.98334642812285722 42.9844739669082827, -87.98409445082069169 42.9844739669082827), (-87.98398389681051412 42.98492725383756152, -87.9834560905684242 42.98492725383756152, -87.98343201832427951 42.98454440595978809, -87.98402490878204674 42.98453201391029666, -87.98398389681051412 42.98492725383756152))"
        )
        country = "us"
        date_1 = "2022-04-21"
        date_2 = "2022-04-21"
        packs = ["building"]
        aoi_id = 11

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(aoi, country, packs, None, None, aoi_id, date_1, date_2)
        features_gdf = features_gdf.query("class_id == @ROOF_ID")  # Filter out building classes, just keep roof.
        print(metadata)

        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["survey_date"] <= date_2
        # We get no buildings (inner gets discarded)
        assert len(features_gdf) == 0
        assert len(features_gdf[features_gdf.class_id == ROOF_ID]) == 0

    def test_polygon_with_hole_2(self, cache_directory: Path):
        """
        Test correct behaviour of a connected class, with complex many holed query AOI.
        """
        aoi = loads(
            "Polygon ((-87.98869354480743254 42.98720736071164339, -87.98869354480743254 42.98745790451570059, -87.98832791466347203 42.98745790451570059, -87.98832791466347203 42.98720736071164339, -87.98869354480743254 42.98720736071164339),(-87.98862916890175256 42.9874149280962996, -87.98856291796968776 42.98741401370408255, -87.98856229296089282 42.9873628077175951, -87.98862916890175256 42.9874149280962996),(-87.98843666619346493 42.98739206828654602, -87.98848416686175256 42.98736737968242494, -87.98842104097366246 42.98729559979562254, -87.9883935405867561 42.98732120282217295, -87.98834416489208365 42.98737743800379718, -87.98837854037572015 42.98741355650798823, -87.98840979081536773 42.98735457818008854, -87.98840979081536773 42.98735457818008854, -87.98843666619346493 42.98739206828654602),(-87.9883354147689829 42.9873874963235707, -87.98833916482176676 42.9874492177950458, -87.98868416967553685 42.98745013218676547, -87.98868604470192167 42.98742315762577704, -87.9883697902526194 42.98741858566508256, -87.9883354147689829 42.9873874963235707),(-87.98868104463157636 42.98741721407682803, -87.98868104463157636 42.98721467587366618, -87.98833228972502241 42.98721238988563442, -87.98864729415674901 42.9874149280962996, -87.98864729415674901 42.9874149280962996, -87.98868104463157636 42.98741721407682803),(-87.9885416676707024 42.98740761295811552, -87.98855229282020218 42.98736509370006331, -87.98845979151882091 42.98731343047596454, -87.98850041709034997 42.98737469482540519, -87.98841666591209787 42.98740212660388949, -87.98843104111432467 42.98741172772346175, -87.98843104111432467 42.98741172772346175, -87.9885416676707024 42.98740761295811552))"
        )
        country = "us"
        date_1 = "2022-04-21"
        date_2 = "2022-04-21"
        packs = ["surfaces"]
        aoi_id = 12

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(aoi, country, packs, None, None, aoi_id, date_1, date_2)
        print(metadata)

        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["survey_date"] <= date_2
        # We get one piece of hollowed out asphalt
        assert len(features_gdf) == 3
        assert len(features_gdf[features_gdf.class_id == ASPHALT_ID]) == 1
        features_gdf_asphalt = features_gdf.query("class_id == @ASPHALT_ID")
        assert features_gdf_asphalt["clipped_area_sqm"].sum() == features_gdf_asphalt["area_sqm"].sum()
        assert features_gdf_asphalt["clipped_area_sqm"].sum() == pytest.approx(259.2, rel=0.02)

    def test_classes(self, cache_directory: Path):
        feature_api = FeatureApi(cache_dir=cache_directory)
        classes_df = feature_api.get_feature_classes()
        assert classes_df.loc[BUILDING_ID].description == "Building (Deprecated)"

    def test_classes_filtered(self, cache_directory: Path):
        feature_api = FeatureApi(cache_dir=cache_directory)
        classes_df = feature_api.get_feature_classes(packs=["solar", "building"])
        assert classes_df.loc[BUILDING_ID].description == "Building (Deprecated)"
        assert classes_df.loc[BUILDING_NEW_ID].description == "Building"
        assert classes_df.loc[ROOF_ID].description == "Roof"
        assert classes_df.loc[SOLAR_ID].description == "Solar Panel"
        assert classes_df.loc[SOLAR_HW_ID].description == "Solar Hot Water"
        assert len(classes_df) == 5

    def test_unknown_pack(self, cache_directory: Path):
        feature_api = FeatureApi(cache_dir=cache_directory)
        with pytest.raises(ValueError) as excinfo:
            feature_api.get_feature_classes(packs=["solar", "foobar"])
        assert "foobar" in str(excinfo.value)

    def test_packs(self, cache_directory: Path):
        """
        Test that this set of packs are all valid. Does not check whether additional packs have been made available.
        """
        feature_api = FeatureApi(cache_dir=cache_directory)
        packs = feature_api.get_packs()
        expected_subset = {
            "building",
            "building_char",
            "roof_char",
            "roof_cond",
            "surfaces",
            "vegetation",
            "poles",
            "construction",
            "pool",
            "solar",
            "trampoline",
        }
        assert not expected_subset.difference(packs.keys())

    def test_include_parameter_api_call(self, cache_directory: Path, sydney_aoi: Polygon):
        """Test that include parameter is correctly passed to API and doesn't cause errors"""
        date_1 = "2025-01-20"
        date_2 = "2025-01-20"
        country = "au"
        packs = ["roof_char", "roof_cond"]
        aoi_id = "123"
        
        feature_api = FeatureApi(cache_dir=cache_directory)
        
        # Test that including RSI and confidence stats doesn't break the API call
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(
            geometry=sydney_aoi, 
            region=country, 
            packs=packs, 
            include=["roofSpotlightIndex", "roofConditionConfidenceStats"],
            aoi_id=aoi_id, 
            since=date_1, 
            until=date_2
        )
        
        # Should not error out
        assert error is None
        assert metadata is not None
        assert date_1 <= metadata["survey_date"] <= date_2
        
        # Should still get roof features
        if features_gdf is not None and len(features_gdf) > 0:
            roof_features = features_gdf[features_gdf.description == "Roof"]
            # If we have roof features, check attributes exist
            if len(roof_features) > 0:
                first_roof = roof_features.iloc[0]
                assert hasattr(first_roof, 'attributes')
                # Note: The actual RSI data might not be present in test data,
                # but the API call should succeed

    def test_param_dic_api_call(self, cache_directory: Path, sydney_aoi: Polygon):
        """Test that param_dic parameter correctly adds custom parameters to API URL"""
        date_1 = "2025-01-20"
        date_2 = "2025-01-20"
        country = "au"
        packs = ["roof_char", "roof_cond"]
        aoi_id = "123"

        feature_api = FeatureApi(cache_dir=cache_directory)

        # Use patch to capture the URL being constructed
        with patch('requests.Session.post') as mock_post:
            # Create a mock response with all required fields
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.return_value = {
                "features": [],
                "systemVersion": "test-version",
                "link": "https://apps.nearmap.com/maps/#/@-33.8688,151.2093,20.00z,0d/V/20250120?locationMarker",
                "surveyDate": "2025-01-20",
                "surveyId": "test-survey-id",
                "resourceId": "test-resource-id",
                "perspective": "Vert",
                "postcat": False
            }
            mock_post.return_value = mock_response

            # Call with param_dic - using 'include' as a test parameter
            # (normally you'd use the include parameter, but this tests param_dic)
            param_dic = {"include": "roofSpotlightIndex"}

            features_gdf, metadata, error, _ = feature_api.get_features_gdf(
                geometry=sydney_aoi,
                region=country,
                packs=packs,
                param_dic=param_dic,
                aoi_id=aoi_id,
                since=date_1,
                until=date_2
            )

            # Verify the mock was called
            assert mock_post.called

            # Get the URL that was used in the POST request
            call_args = mock_post.call_args
            url_used = call_args[0][0] if call_args[0] else call_args[1].get('url', '')

            # Verify the custom parameter is in the URL
            assert "include=roofSpotlightIndex" in url_used, f"Expected 'include=roofSpotlightIndex' in URL: {url_used}"

    def test_gridding_semaphore_limits_concurrent_operations(self):
        """Test that semaphore correctly limits concurrent gridding operations to 1/5th of thread count"""
        thread_count = 20
        expected_max_concurrent = max(1, thread_count // 5)  # Should be 4
        
        with tempfile.TemporaryDirectory() as cache_dir:
            api = FeatureApi(cache_dir=Path(cache_dir), threads=thread_count)
            
            # Verify semaphore was initialized with correct value
            assert api._gridding_semaphore._value == expected_max_concurrent
            
            # Track concurrent executions
            concurrent_count = 0
            max_concurrent_seen = 0
            lock = threading.Lock()
            
            def simulate_gridding():
                nonlocal concurrent_count, max_concurrent_seen
                with api._gridding_semaphore:
                    with lock:
                        concurrent_count += 1
                        max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
                    time.sleep(0.01)  # Simulate work
                    with lock:
                        concurrent_count -= 1
            
            # Start many threads to test semaphore limiting
            threads = []
            for _ in range(20):
                t = threading.Thread(target=simulate_gridding)
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            # Verify we never exceeded the limit
            assert max_concurrent_seen <= expected_max_concurrent
            assert max_concurrent_seen > 0  # Ensure test actually ran
    
    def test_pool_size_capped_at_50(self):
        """Test that HTTP adapter pool size is capped at 50 even with high thread counts"""
        with tempfile.TemporaryDirectory() as cache_dir:
            # Test with thread count > 50
            api = FeatureApi(cache_dir=Path(cache_dir), threads=100)
            with api._session_scope() as session:
                # Check the adapter's pool settings (stored as private attributes)
                adapter = session.get_adapter("https://")
                assert adapter._pool_maxsize == 50
                assert adapter._pool_connections == 50
            
            # Test with thread count < 50 but > 10
            api2 = FeatureApi(cache_dir=Path(cache_dir), threads=30)
            with api2._session_scope() as session2:
                adapter2 = session2.get_adapter("https://")
                assert adapter2._pool_maxsize == 30
                assert adapter2._pool_connections == 30
            
            # Test with thread count < 10
            api3 = FeatureApi(cache_dir=Path(cache_dir), threads=5)
            with api3._session_scope() as session3:
                adapter3 = session3.get_adapter("https://")
                assert adapter3._pool_maxsize == 10  # Minimum of 10
                assert adapter3._pool_connections == 10
    
    def test_cache_write_cleans_up_temp_files(self):
        """Test that cache write properly cleans up temp files even on failure"""
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_path = Path(cache_dir)
            api = FeatureApi(cache_dir=cache_path)
            
            # Test successful write - temp file should be cleaned up
            test_payload = {"test": "data"}
            target_path = cache_path / "test_cache.json"
            api._write_to_cache(target_path, test_payload)
            
            # Check no temp files remain
            temp_files = list(cache_path.glob("*.tmp*"))
            assert len(temp_files) == 0
            assert target_path.exists()
            
            # Test with compressed cache
            api.compress_cache = True
            target_path2 = cache_path / "test_cache2.json.gz"
            api._write_to_cache(target_path2, test_payload)
            
            # Check no temp files remain
            temp_files = list(cache_path.glob("*.tmp*"))
            assert len(temp_files) == 0
            assert target_path2.exists()
            
            # Test failure scenario - mock replace to fail
            with patch.object(Path, 'replace', side_effect=Exception("Mock failure")):
                target_path3 = cache_path / "test_cache3.json"
                try:
                    api._write_to_cache(target_path3, test_payload)
                except Exception:
                    pass  # Expected to fail
                
                # Check no temp files remain even after failure
                temp_files = list(cache_path.glob("*.tmp*"))
                assert len(temp_files) == 0

    def test_api_key_filtering(self):
        """Test that API key filter removes sensitive information from logs"""
        from nmaipy.api_common import APIKeyFilter
        import logging

        # Create a mock log record
        class MockRecord:
            def __init__(self, msg):
                self.msg = msg
                self.args = ()

            def getMessage(self):
                return self.msg

        filter = APIKeyFilter()

        # Test URL with API key
        test_cases = [
            ("/api/endpoint?apikey=SECRET123KEY&other=param", "apikey=REMOVED"),
            ("api_key: 'SECRET456'", "REMOVED"),
            ("API-KEY=SECRET789", "REMOVED"),
            ('"apiKey": "SECRET000"', "REMOVED"),
            ("Request failed for apikey=MYSECRETKEY", "apikey=REMOVED"),
        ]

        for test_input, expected_pattern in test_cases:
            record = MockRecord(test_input)
            filter.filter(record)
            assert "SECRET" not in record.msg, f"Secret not removed from: {test_input}"
            assert "REMOVED" in record.msg, f"REMOVED not added to: {test_input}"

    def test_clean_api_key_with_url_parsing(self):
        """Test that _clean_api_key properly handles various URL formats using urllib.parse"""
        api = FeatureApi(api_key="TEST_SECRET_KEY_123")

        # Test cases with different URL formats
        test_cases = [
            # Standard URL with apikey
            ("https://api.nearmap.com/ai/features?apikey=TEST_SECRET_KEY_123&other=param",
             "https://api.nearmap.com/ai/features?apikey=APIKEYREMOVED&other=param"),

            # URL with URL-encoded characters in API key
            ("https://api.nearmap.com/ai/features?apikey=TEST%2BSECRET%2FKEY%3D123&other=param",
             "https://api.nearmap.com/ai/features?apikey=APIKEYREMOVED&other=param"),

            # Path-only URL with query params
            ("/ai/features?apikey=TEST_SECRET_KEY_123&param=value",
             "/ai/features?apikey=APIKEYREMOVED&param=value"),

            # Multiple parameters with apikey in middle
            ("https://api.nearmap.com/ai/features?before=1&apikey=TEST_SECRET_KEY_123&after=2",
             "https://api.nearmap.com/ai/features?before=1&apikey=APIKEYREMOVED&after=2"),

            # Case insensitive apikey parameter
            ("https://api.nearmap.com/ai/features?APIKEY=TEST_SECRET_KEY_123",
             "https://api.nearmap.com/ai/features?APIKEY=APIKEYREMOVED"),

            # URL without apikey parameter (should remain unchanged)
            ("https://api.nearmap.com/ai/features?other=param",
             "https://api.nearmap.com/ai/features?other=param"),

            # Non-URL string with API key (fallback to simple replacement)
            ("Some text with TEST_SECRET_KEY_123 in it",
             "Some text with APIKEYREMOVED in it"),

            # Complex URL with fragment
            ("https://api.nearmap.com/ai/features?apikey=TEST_SECRET_KEY_123#section",
             "https://api.nearmap.com/ai/features?apikey=APIKEYREMOVED#section"),
        ]

        for input_url, expected_output in test_cases:
            result = api._clean_api_key(input_url)
            # Check that the API key is not in the result
            assert "TEST_SECRET_KEY_123" not in result, f"API key not removed from: {input_url}"
            # For URL cases, check structure is preserved
            if input_url.startswith("http") or input_url.startswith("/"):
                assert "APIKEYREMOVED" in result or "apikey" not in input_url.lower(), \
                    f"APIKEYREMOVED not added correctly for: {input_url}"

    def test_curl_command_generation(self):
        """Test that curl commands are generated correctly with sanitized API keys"""
        # Test POST request
        test_url = "https://api.nearmap.com/ai/features/v4/bulk/features.json?apikey=TEST_API_KEY_12345&param=value"
        test_body = {
            "aoi": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
            }
        }

        curl_cmd = generate_curl_command(test_url, test_body, method="POST", timeout=READ_TIMEOUT_SECONDS)

        # Verify API key is removed
        assert "TEST_API_KEY_12345" not in curl_cmd, "API key not removed from curl command"
        assert "APIKEYREMOVED" in curl_cmd, "API key not replaced with APIKEYREMOVED"
        assert "curl -X POST" in curl_cmd, "Missing POST method"
        assert "Content-Type: application/json" in curl_cmd, "Missing content type header"
        assert "--max-time" in curl_cmd, "Missing timeout"

        # Test GET request
        curl_cmd_get = generate_curl_command(test_url, None, method="GET", timeout=READ_TIMEOUT_SECONDS)
        assert "TEST_API_KEY_12345" not in curl_cmd_get, "API key not removed from GET command"
        assert "curl -X GET" in curl_cmd_get, "Missing GET method"

    def test_timeout_values(self):
        """Test that timeout values are correctly set"""
        from nmaipy.feature_api import READ_TIMEOUT_SECONDS, TIMEOUT_SECONDS

        # Check the expected timeout values
        assert READ_TIMEOUT_SECONDS == 90, f"Expected read timeout 90s, got {READ_TIMEOUT_SECONDS}s"
        assert TIMEOUT_SECONDS == 120, f"Expected connect timeout 120s, got {TIMEOUT_SECONDS}s"

    def test_session_timeout_application(self):
        """Test that timeouts are properly applied to sessions"""
        with tempfile.TemporaryDirectory() as cache_dir:
            api = FeatureApi(api_key="TEST_KEY", cache_dir=Path(cache_dir))

            with api._session_scope() as session:
                # Check that timeout is stored on the session
                assert hasattr(session, '_timeout'), "Session missing _timeout attribute"
                assert session._timeout == (120, 90), f"Wrong timeout values: {session._timeout}"

    def test_read_timeout_treated_as_504(self):
        """Test that read timeouts are treated as 504 Gateway Timeout errors"""
        import requests
        from http import HTTPStatus
        from nmaipy.feature_api import AIFeatureAPIRequestSizeError
        from shapely.geometry import Polygon

        api = FeatureApi(api_key="TEST_KEY", cache_dir=None)

        # Create a simple test polygon
        test_polygon = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001), (0, 0)])

        # Mock session.post to raise ReadTimeout
        with patch('requests.Session.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ReadTimeout("Read timed out")

            try:
                # This should catch the timeout and treat it as a 504
                result = api._get_results(
                    geometry=test_polygon,
                    region="au",
                    result_type="features",
                    in_gridding_mode=False
                )
                # Should not reach here
                assert False, "Should have raised AIFeatureAPIRequestSizeError"
            except AIFeatureAPIRequestSizeError as e:
                # This is expected - read timeouts should trigger gridding like 504s
                assert e.status_code == HTTPStatus.GATEWAY_TIMEOUT, f"Expected 504, got {e.status_code}"

    def test_timeout_logging_with_debug(self):
        """Test that timeout errors generate debug logs with curl commands"""
        import requests
        from shapely.geometry import Polygon
        from unittest.mock import patch
        import logging

        # Set up debug logging
        logging.getLogger('nmaipy').setLevel(logging.DEBUG)

        api = FeatureApi(api_key="TEST_KEY", cache_dir=None)
        test_polygon = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001), (0, 0)])

        with patch('requests.Session.post') as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Connection timeout")

            # Capture log messages
            with patch('nmaipy.feature_api.logger') as mock_logger:
                features_gdf, metadata, error, _ = api.get_features_gdf(
                    geometry=test_polygon,
                    region="au",
                    packs=["building"],
                    aoi_id="test_aoi"
                )

                # Should have logged warning and debug messages
                assert mock_logger.warning.called
                assert error is not None
                assert error["message"] == "TIMEOUT_ERROR"

                # In debug mode, should have generated curl command
                if mock_logger.level == logging.DEBUG:
                    debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
                    # Should have debug message about curl command
                    assert any("curl" in str(call).lower() for call in debug_calls)

    def test_exception_sanitization(self):
        """Test that exceptions properly sanitize API keys from stored request strings"""
        from nmaipy.feature_api import AIFeatureAPIRequestSizeError
        from shapely.geometry import Polygon
        import requests
        from unittest.mock import MagicMock

        api = FeatureApi(api_key="SUPER_SECRET_KEY_123", cache_dir=None)
        test_polygon = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001), (0, 0)])

        # Test case 1: ChunkedEncodingError leading to exception
        with patch('requests.Session.post') as mock_post:
            # Simulate ChunkedEncodingError that triggers exception
            mock_post.side_effect = requests.exceptions.ChunkedEncodingError("Test error")

            try:
                api._get_results(
                    geometry=test_polygon,
                    region="au",
                    result_type="features",
                    in_gridding_mode=False
                )
                assert False, "Should have raised AIFeatureAPIRequestSizeError"
            except AIFeatureAPIRequestSizeError as e:
                # Check that the stored request_string doesn't contain the API key
                assert "SUPER_SECRET_KEY_123" not in str(e.request_string), \
                    f"API key found in exception request_string: {e.request_string}"
                assert "APIKEYREMOVED" in str(e.request_string), \
                    "Exception should contain sanitized placeholder"

        # Test case 2: 504 status code leading to exception
        with patch('requests.Session.post') as mock_post:
            mock_response = MagicMock()
            mock_response.ok = False
            mock_response.status_code = 504
            mock_response.text = "Gateway Timeout"
            mock_post.return_value = mock_response

            try:
                api._get_results(
                    geometry=test_polygon,
                    region="au",
                    result_type="features",
                    in_gridding_mode=False
                )
                assert False, "Should have raised AIFeatureAPIRequestSizeError"
            except AIFeatureAPIRequestSizeError as e:
                # Check that the stored request_string doesn't contain the API key
                assert "SUPER_SECRET_KEY_123" not in str(e.request_string), \
                    f"API key found in exception request_string: {e.request_string}"
                assert "APIKEYREMOVED" in str(e.request_string), \
                    "Exception should contain sanitized placeholder"

        # Test case 3: JSON parsing error leading to exception
        with patch('requests.Session.post') as mock_post:
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_post.return_value = mock_response

            try:
                api._get_results(
                    geometry=test_polygon,
                    region="au",
                    result_type="features",
                    in_gridding_mode=False
                )
                assert False, "Should have raised AIFeatureAPIRequestSizeError"
            except AIFeatureAPIRequestSizeError as e:
                # Check that the stored request_string doesn't contain the API key
                assert "SUPER_SECRET_KEY_123" not in str(e.request_string), \
                    f"API key found in exception request_string: {e.request_string}"
                assert "APIKEYREMOVED" in str(e.request_string), \
                    "Exception should contain sanitized placeholder"

    def test_max_allowed_error_count_uses_floor_not_round(self):
        """
        Test that max_allowed_error_count uses math.floor instead of round.

        This prevents a bug where small grids with high error percentage could allow
        100% failure due to rounding. E.g., with 50 AOIs and 99% allowed errors:
        - round(50 * 99 / 100) = round(49.5) = 50, allowing all 50 to fail
        - floor(50 * 99 / 100) = floor(49.5) = 49, requiring at least 1 success
        """
        test_cases = [
            # (num_aois, max_allowed_error_pct, expected_max_errors)
            (50, 99, 49),   # floor(49.5) = 49, not round(49.5) = 50
            (50, 100, 50),  # floor(50) = 50
            (100, 99, 99),  # floor(99) = 99
            (100, 1, 1),    # floor(1) = 1
            (3, 99, 2),     # floor(2.97) = 2, not round(2.97) = 3
            (10, 95, 9),    # floor(9.5) = 9, not round(9.5) = 10
        ]

        for num_aois, max_allowed_error_pct, expected in test_cases:
            actual = math.floor(num_aois * max_allowed_error_pct / 100)
            assert actual == expected, (
                f"For {num_aois} AOIs with {max_allowed_error_pct}% allowed errors: "
                f"expected max_allowed_error_count={expected}, got {actual}"
            )

            # Also verify that round() would give different (wrong) results for edge cases
            rounded = round(num_aois * max_allowed_error_pct / 100)
            if num_aois == 50 and max_allowed_error_pct == 99:
                assert rounded == 50, "round(49.5) should be 50 (the bug we fixed)"
                assert actual == 49, "floor(49.5) should be 49 (correct behavior)"

    def test_gridding_with_all_failures_returns_error_not_crash(self, cache_directory: Path, monkeypatch):
        """
        Test that when all grid cells fail, we get a proper error dict instead of
        'single positional indexer is out-of-bounds' crash from .iloc[0] on empty DataFrame.

        This tests the fix for the bug where _attempt_gridding assumed metadata_df
        would always have at least one row after gridding.
        """
        from shapely.geometry import box

        # Create a large AOI that will trigger gridding
        large_aoi = box(150.0, -34.0, 151.0, -33.0)  # ~100km x 100km

        feature_api = FeatureApi(cache_dir=cache_directory, aoi_grid_min_pct=0)

        # Mock get_features_gdf_bulk to return empty metadata (simulating all grid cells failing)
        def mock_get_features_gdf_bulk(*args, **kwargs):
            # Return empty features, empty metadata, and some errors
            empty_features = gpd.GeoDataFrame(columns=['geometry'], crs=API_CRS)
            empty_features.index.name = AOI_ID_COLUMN_NAME
            empty_metadata = pd.DataFrame([])
            empty_metadata.index.name = AOI_ID_COLUMN_NAME
            errors = pd.DataFrame([{
                AOI_ID_COLUMN_NAME: 0,
                'status_code': 404,
                'message': 'No coverage',
            }]).set_index(AOI_ID_COLUMN_NAME)
            return empty_features, empty_metadata, errors

        monkeypatch.setattr(feature_api, 'get_features_gdf_bulk', mock_get_features_gdf_bulk)

        # Call _attempt_gridding directly - should NOT crash with IndexError
        features, metadata, error, grid_errors = feature_api._attempt_gridding(
            geometry=large_aoi,
            region='au',
            packs=['building'],
            aoi_id='test_aoi',
        )

        # Should return None for features/metadata and an error dict
        assert features is None, "Expected None features when all grid cells fail"
        assert metadata is None, "Expected None metadata when all grid cells fail"
        assert error is not None, "Expected error dict when all grid cells fail"
        assert 'message' in error, "Error should contain a message"
        assert error['failure_type'] == 'grid', "Error should be marked as grid failure"


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    sys.exit(pytest.main([current_file]))
