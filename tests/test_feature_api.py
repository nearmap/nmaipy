from pathlib import Path

import geopandas as gpd
import pytest
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.wkt import loads

from nearmap_ai.constants import BUILDING_ID, SOLAR_ID, VEG_MEDHIGH_ID
from nearmap_ai.feature_api import FeatureApi


class TestFeatureAPI:
    def test_get_features_gdf(self, sydney_aoi: Polygon, cache_directory: Path):
        date_1 = "2020-01-01"
        date_2 = "2020-06-01"
        packs = ["building"]
        aoi_id = "123"

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(sydney_aoi, packs, aoi_id, date_1, date_2)
        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["date"] <= date_2
        # We get 3 buildings
        assert len(features_gdf) == 3
        assert len(features_gdf[features_gdf.class_id == BUILDING_ID]) == 3
        # The AOI ID has been assigned
        assert len(features_gdf[features_gdf.aoi_id == aoi_id]) == 3

    def test_large_aoi(self, cache_directory: Path, large_adelaide_aoi: Polygon):
        survey_resource_id = "fe48a583-da45-5cd3-9fee-8321354bdf7a" # 2011-03-03
        packs = ["building", "vegetation"]
        aoi_id = "0"

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(large_adelaide_aoi, packs=packs, aoi_id=aoi_id, survey_resource_id=survey_resource_id)
        # No error
        assert error is None
        # We get 3 buildings
        assert len(features_gdf.query("class_id == @BUILDING_ID")) == 6 # Guessed
        assert len(features_gdf.query("class_id == @VEG_MEDHIGH_ID")) == 213 # Guessed

        # Assert that buildings aren't overhanging the edge of the parcel. If this fails, the clipped/unclipped hasn't been managed correctly during the grid merge.
        assert features_gdf.query("class_id == @BUILDING_ID").clipped_area_sqm.sum() == features_gdf.query("class_id == @BUILDING_ID").unclipped_area_sqm.sum()


        assert len(features_gdf.query("class_id == @VEG_MEDHIGH_ID")) == 213  # Guessed

        # The AOI ID has been assigned to all features
        assert len(features_gdf[features_gdf.aoi_id == aoi_id]) == len(features_gdf)

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
        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(aoi)
        # No data
        assert features_gdf is None
        assert metadata is None
        # There is a error message
        assert isinstance(error["message"], str)

    def test_get_cache(self, cache_directory: Path, sydney_aoi: Polygon):
        date_1 = "2020-06-01"
        date_2 = "2020-12-01"
        packs = ["building"]
        aoi_id = "123"
        # First do a standard pull to ensure the file is populated in the cache.
        feature_api = FeatureApi(cache_dir=cache_directory, compress_cache=False)
        features_gdf, metadata, error = feature_api.get_features_gdf(sydney_aoi, packs, aoi_id, date_1, date_2)
        assert error is None

        # Then re-request using invalid API key to ensure data is not being pulled from the API but read from the cache.
        api_key = "not an api key"
        # Run
        feature_api = FeatureApi(api_key, cache_dir=cache_directory, compress_cache=False)
        features_gdf, metadata, error = feature_api.get_features_gdf(sydney_aoi, packs, aoi_id, date_1, date_2)
        # Check output
        assert error is None
        assert date_1 <= metadata["date"] <= date_2
        assert len(features_gdf) == 3

    def test_get_compressed_cache(self, cache_directory: Path, sydney_aoi: Polygon):
        date_1 = "2020-06-01"
        date_2 = "2020-12-01"
        packs = ["building"]
        aoi_id = "123"
        # First do a standard pull to ensure the file is populated in the cache.
        feature_api = FeatureApi(cache_dir=cache_directory, compress_cache=True)
        features_gdf, metadata, error = feature_api.get_features_gdf(sydney_aoi, packs, aoi_id, date_1, date_2)
        assert error is None

        # Then re-request using invalid API key to ensure data is not being pulled from the API but read from the cache.
        api_key = "not an api key"
        # Run
        feature_api = FeatureApi(api_key, cache_dir=cache_directory, compress_cache=True)
        features_gdf, metadata, error = feature_api.get_features_gdf(sydney_aoi, packs, aoi_id, date_1, date_2)
        # Check output
        assert error is None
        assert date_1 <= metadata["date"] <= date_2
        assert len(features_gdf) == 3

    def test_get_bulk(self, cache_directory: Path, sydney_aoi: Polygon):
        aois = []
        for i in range(4):
            for j in range(4):
                aois.append({"aoi_id": f"{i}_{j}", "geometry": translate(sydney_aoi, 0.001 * i, 0.001 * j)})
        # Add a massive AOI to test an error care
        aois.append({"aoi_id": "error_case", "geometry": sydney_aoi.buffer(1)})

        aoi_gdf = gpd.GeoDataFrame(aois)
        date_1 = "2020-01-01"
        date_2 = "2020-12-01"
        packs = ["building"]

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(aoi_gdf, packs, date_1, date_2)
        # Check metadata
        assert len(metadata_df) == 16
        assert len(metadata_df.merge(aoi_gdf, on="aoi_id", how="inner")) == 16
        # Check error
        assert len(errors_df) == 1
        assert errors_df.iloc[0].aoi_id == "error_case"
        # We get only buildings
        assert len(features_gdf) == 69
        assert len(features_gdf[features_gdf.class_id == BUILDING_ID]) == 69

    def test_get_bulk_with_data_dates(self, cache_directory: Path, sydney_aoi: Polygon):
        aois = []
        for i in range(4):
            for j in range(4):
                aois.append(
                    {
                        "aoi_id": f"{i}_{j}",
                        "since": "2020-01-01",
                        "until": "2020-03-01",
                        "geometry": translate(sydney_aoi, 0.001 * i, 0.001 * j),
                    }
                )

        aoi_gdf = gpd.GeoDataFrame(aois)
        packs = ["building"]

        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(aoi_gdf, packs)
        # Check metadata
        assert len(metadata_df) == 16
        assert len(metadata_df.merge(aoi_gdf, on="aoi_id", how="inner")) == 16

        # We get only buildings
        assert len(features_gdf) == 70
        assert len(features_gdf[features_gdf.class_id == BUILDING_ID]) == 70
        # The dates are within range
        for row in features_gdf.itertuples():
            print(row.survey_date)
            assert "2020-01-01" <= row.survey_date <= "2020-03-01"

    def test_multipolygon(self, cache_directory: Path, sydney_aoi: Polygon):
        aoi = sydney_aoi.union(translate(sydney_aoi, 0.002, 0.01))
        date_1 = "2020-01-01"
        date_2 = "2020-06-01"
        packs = ["building"]
        aoi_id = "123"
        # Run
        feature_api = FeatureApi(cache_dir=cache_directory)
        features_gdf, metadata, error = feature_api.get_features_gdf(aoi, packs, aoi_id, date_1, date_2)
        # No error
        assert error is None
        # Date is in range
        assert date_1 <= metadata["date"] <= date_2
        # We get 3 buildings
        assert len(features_gdf) == 6
        assert len(features_gdf[features_gdf.class_id == BUILDING_ID]) == 6
        # The AOI ID has been assigned
        assert len(features_gdf[features_gdf.aoi_id == aoi_id]) == 6
        # All buildings intersect the AOI
        assert len(features_gdf[features_gdf.intersects(aoi)]) == 6

    def test_classes(self, cache_directory: Path):
        feature_api = FeatureApi(cache_dir=cache_directory)
        classes_df = feature_api.get_feature_classes()
        assert classes_df.loc[BUILDING_ID].description == "Building"

    def test_classes_filtered(self, cache_directory: Path):
        feature_api = FeatureApi(cache_dir=cache_directory)
        classes_df = feature_api.get_feature_classes(packs=["solar", "building"])
        assert classes_df.loc[BUILDING_ID].description == "Building"
        assert classes_df.loc[SOLAR_ID].description == "Solar Panel"
        assert len(classes_df) == 2

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
