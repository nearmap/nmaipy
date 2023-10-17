import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon
import warnings

from nmaipy.feature_api import FeatureApi
from nmaipy.constants import *

sys.path.append(Path(__file__).parent.parent.absolute() / "scripts")
import ai_offline_parcel

TEST_TMP_FOLDER = Path("data/tmp")

class TestAIOfflineParcel:
    @pytest.mark.filterwarnings("ignore:.*initial implementation of Parquet.*")
    def test_process_chunk_rollup_single_multi_polygon_combo(
        self, parcels_3_gdf: gpd.GeoDataFrame, cache_directory: Path, processed_output_directory: Path
    ):
        """
        Comparison of results from the rollup api, or the feature api with local logic, confirming the implementations
        give the same result.
        # TODO: Create a larger, more diverse set of test parcels and test full identical nature of equivalent results.
        Args:
            parcel_gdf_au_tests:
            cache_directory:
            processed_output_directory:

        Returns:

        """
        tag = "tests3"
        tag_rollup_api = "tests3_rollup"
        chunk_id = 0

        output_dir = Path(TEST_TMP_FOLDER) / tag
        output_dir_rollup_api = Path(TEST_TMP_FOLDER) / tag_rollup_api
        packs = ["surface_permeability"]
        country = "au"
        final_path_rollup_api = output_dir_rollup_api / "final"  # Permanent path for later visual inspection
        final_path_rollup_api.mkdir(parents=True, exist_ok=True)

        chunk_path_rollup_api = output_dir_rollup_api / "chunks"
        chunk_path_rollup_api.mkdir(parents=True, exist_ok=True)

        cache_path_rollup_api = output_dir_rollup_api / "cache"
        cache_path_rollup_api.mkdir(parents=True, exist_ok=True)

        feature_api = FeatureApi()
        classes_df = feature_api.get_feature_classes(packs)

        print("Processing Chunk")
        ai_offline_parcel.process_chunk(
            chunk_id=chunk_id,
            parcel_gdf=parcels_3_gdf,
            classes_df=classes_df,
            output_dir=output_dir_rollup_api,
            key_file=None,
            config=None,
            country=country,
            packs=packs,
            include_parcel_geometry=True,
            save_features=False,
            since_bulk="2022-10-29",
            until_bulk="2022-10-29",
            alpha=False,
            beta=False,
            endpoint="rollup",
        )

        data_rollup_api_errors = []
        for cp in chunk_path_rollup_api.glob(f"errors_*.parquet"):
            data_rollup_api_errors.append(pd.read_parquet(cp))
        data_rollup_api_errors = pd.concat(data_rollup_api_errors)

        data_rollup_api = []
        for cp in chunk_path_rollup_api.glob(f"rollup_*.parquet"):
            data_rollup_api.append(pd.read_parquet(cp))
        data_rollup_api = pd.concat(data_rollup_api)
        print("data rollup api")
        print(data_rollup_api.T)
        print(data_rollup_api.loc[:, "link"].values)

    def test_process_chunk_au(
        self, parcel_gdf_au_tests: gpd.GeoDataFrame, cache_directory: Path, processed_output_directory: Path
    ):
        tag = "tests_au"
        chunk_id = 0

        output_dir = Path(TEST_TMP_FOLDER) / tag
        packs = ["building", "vegetation"]
        country = "au"
        final_path = output_dir / "final"  # Permanent path for later visual inspection
        final_path.mkdir(parents=True, exist_ok=True)

        chunk_path = output_dir / "chunks"
        chunk_path.mkdir(parents=True, exist_ok=True)

        cache_path = output_dir / "cache"

        feature_api = FeatureApi()
        classes_df = feature_api.get_feature_classes(packs)

        ai_offline_parcel.process_chunk(
            chunk_id=chunk_id,
            parcel_gdf=parcel_gdf_au_tests,
            classes_df=classes_df,
            output_dir=output_dir,
            key_file=None,
            config=None,
            country=country,
            packs=packs,
            include_parcel_geometry=True,
            save_features=True,
        )

        assert chunk_path.exists()
        assert (chunk_path / f"rollup_{chunk_id}.parquet").exists()
        assert cache_path.exists()

        data = []
        data_features = []
        errors = []

        for cp in chunk_path.glob(f"rollup_*.parquet"):
            data.append(pd.read_parquet(cp))

        outpath = final_path / f"{tag}.csv"
        outpath_features = final_path / f"{tag}_features.gpkg"
        data = pd.concat(data)
        data.to_csv(outpath, index=True)

        outpath_errors = final_path / f"{tag}_errors.csv"
        for cp in chunk_path.glob(f"errors_*.parquet"):
            errors.append(pd.read_parquet(cp))
        errors = pd.concat(errors)
        errors.to_csv(outpath_errors, index=True)

        for cp in [p for p in chunk_path.glob(f"features_*.parquet")]:
            data_features.append(gpd.read_parquet(cp))
        data_features = pd.concat(data_features)
        if len(data_features) > 0:
            data_features.to_file(outpath_features, driver="GPKG")

        assert outpath.exists()
        assert outpath_errors.exists()
        assert outpath_features.exists()

        assert len(data) == len(parcel_gdf_au_tests)  # Assert got a result for every parcel.
        print(data.T)

    def test_process_chunk_rollup_vs_feature_calc(
        self, parcels_2_gdf: gpd.GeoDataFrame, cache_directory: Path, processed_output_directory: Path
    ):
        """
        Comparison of results from the rollup api, or the feature api with local logic, confirming the implementations
        give the same result.
        Args:
            parcel_gdf_au_tests:
            cache_directory:
            processed_output_directory:

        Returns:

        """

        tag = "tests2"
        tag_rollup_api = "tests2_rollup"
        chunk_id = 0

        output_dir = Path(TEST_TMP_FOLDER) / tag
        output_dir_rollup_api = Path(TEST_TMP_FOLDER) / tag_rollup_api
        packs = ["building", "vegetation"]
        country = "us"
        final_path = output_dir / "final"  # Permanent path for later visual inspection
        final_path_rollup_api = output_dir_rollup_api / "final"  # Permanent path for later visual inspection
        final_path.mkdir(parents=True, exist_ok=True)
        final_path_rollup_api.mkdir(parents=True, exist_ok=True)

        chunk_path = output_dir / "chunks"
        chunk_path_rollup_api = output_dir_rollup_api / "chunks"
        chunk_path.mkdir(parents=True, exist_ok=True)
        chunk_path_rollup_api.mkdir(parents=True, exist_ok=True)

        cache_path = output_dir / "cache"
        cache_path_rollup_api = output_dir_rollup_api / "cache"
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_path_rollup_api.mkdir(parents=True, exist_ok=True)

        feature_api = FeatureApi()
        classes_df = feature_api.get_feature_classes(packs)

        for endpoint, outdir in [("feature", output_dir), ("rollup", output_dir_rollup_api)]:
            ai_offline_parcel.process_chunk(
                chunk_id=chunk_id,
                parcel_gdf=parcels_2_gdf,
                classes_df=classes_df,
                output_dir=outdir,
                key_file=None,
                config=None,
                country=country,
                packs=packs,
                include_parcel_geometry=True,
                save_features=False,
                since_bulk="2022-06-29",
                until_bulk="2022-06-29",
                alpha=False,
                beta=False,
                endpoint=endpoint,
            )

        data_feature_api = []
        for cp in chunk_path.glob(f"rollup_*.parquet"):
            data_feature_api.append(pd.read_parquet(cp))
        data_feature_api = pd.concat(data_feature_api)

        data_rollup_api = []
        for cp in chunk_path_rollup_api.glob(f"rollup_*.parquet"):
            data_rollup_api.append(pd.read_parquet(cp))
        data_rollup_api = pd.concat(data_rollup_api)

        # print("data feature api")
        # print(data_feature_api.T)
        print("data rollup api")
        print(data_rollup_api.T)
        print(data_rollup_api.loc[:, "link"].values)


        # Test continuous class - tree canopy
        ## Check that counts differ by at most one - sometimes a tiny touching part of a polygon differs between rollup API and local computation due to rounding.
        pd.testing.assert_series_equal(
            data_feature_api.loc[:, "medium_and_high_vegetation_(>2m)_count"],
            data_rollup_api.filter(like=ROLLUP_TREE_CANOPY_COUNT_ID).iloc[:, 0],
            check_exact=False,
            check_names=False,
            atol=1,
        )

        ## Check small error tolerance (max 1 square foot), only where there was no in/out discrepancy on counts
        idx_equal_counts = (
            data_feature_api.loc[:, "medium_and_high_vegetation_(>2m)_count"]
            - data_rollup_api.filter(like=ROLLUP_TREE_CANOPY_COUNT_ID).iloc[:, 0]
        ) == 0

        pd.testing.assert_series_equal(
            data_feature_api.loc[idx_equal_counts, "medium_and_high_vegetation_(>2m)_total_clipped_area_sqft"],
            data_rollup_api.filter(like=ROLLUP_TREE_CANOPY_AREA_CLIPPED_SQFT_ID).loc[idx_equal_counts].iloc[:, 0],
            check_exact=False,
            check_names=False,
            atol=1,
        )
        pd.testing.assert_series_equal(
            data_feature_api.loc[idx_equal_counts, "medium_and_high_vegetation_(>2m)_total_unclipped_area_sqft"],
            data_rollup_api.filter(like=ROLLUP_TREE_CANOPY_AREA_UNCLIPPED_SQFT_ID).loc[idx_equal_counts].iloc[:, 0],
            check_exact=False,
            check_names=False,
            atol=1,
        )

        # Test discrete class - building
        ## Check that counts differ by at most one - sometimes a tiny touching part of a polygon differs between rollup API and local computation due to rounding.
        pd.testing.assert_series_equal(
            data_feature_api.loc[:, "building_count"],
            data_rollup_api.filter(like=ROLLUP_BUILDING_COUNT_ID).iloc[:, 0],
            check_names=False,
            atol=1,
        )

        ## Check small error tolerance (max 1 square foot), only where there was no in/out discrepancy on counts
        idx_equal_counts = (
            data_feature_api.loc[:, "building_count"] - data_rollup_api.filter(like=ROLLUP_BUILDING_COUNT_ID).iloc[:, 0]
        ) == 0

        ## Implicitly test sqm to sqft conversion...
        pd.testing.assert_series_equal(
            data_feature_api.loc[idx_equal_counts, "primary_building_clipped_area_sqft"]
            / SQUARED_METERS_TO_SQUARED_FEET,
            data_rollup_api.filter(like=ROLLUP_BUILDING_PRIMARY_CLIPPED_AREA_SQM_ID).loc[idx_equal_counts]
            .fillna(0)
            .iloc[:, 0],
            check_exact=False,
            check_names=False,
            atol=1,
        )
        pd.testing.assert_series_equal(
            data_feature_api.loc[idx_equal_counts, "primary_building_unclipped_area_sqft"]
            / SQUARED_METERS_TO_SQUARED_FEET,
            data_rollup_api.filter(like=ROLLUP_BUILDING_PRIMARY_UNCLIPPED_AREA_SQM_ID).loc[idx_equal_counts]
            .fillna(0)
            .iloc[:, 0],
            check_exact=False,
            check_names=False,
            atol=1,
        )

        ## Test confidence aggregation is correct to within 1%
        pd.testing.assert_series_equal(
            data_feature_api.loc[idx_equal_counts, "building_confidence"],
            data_rollup_api.filter(like=ROLLUP_BUILDING_PRESENT_CONFIDENCE).loc[idx_equal_counts].iloc[:, 0],
            check_exact=False,
            check_names=False,
            rtol=0.01,
        )

        # TODO: Enable once fidelity score in rollup API.
        # ## Test fidelity score copied correctly to within 1%
        # pd.testing.assert_series_equal(
        #     data_feature_api.loc[idx_equal_counts, "building_primary_fidelity"],
        #     data_rollup_api.loc[idx_equal_counts, ROLLUP_BUILDING_PRIMARY_FIDELITY]
        #     .iloc[:, 0],
        #     check_exact=False,
        #     check_names=False,
        #     rtol=0.01,
        # )

        # Test metadata columns which should be identical
        pd.testing.assert_series_equal(
            data_feature_api.loc[:, "date"],
            data_rollup_api.filter(like=ROLLUP_SURVEY_DATE_ID).iloc[:, 0],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            data_feature_api.loc[:, "system_version"],
            data_rollup_api.filter(like=ROLLUP_SYSTEM_VERSION_ID).iloc[:, 0],
            check_names=False,
        )

        # TODO: Not provided, but should be: ["link", "mesh_date", "fidelity"]
        for ident_col in [
            "aoi_id",
            "query_aoi_lat",
            "query_aoi_lon",
            "geometry",
            # "link",
            # "mesh_date",
        ]:
            pd.testing.assert_series_equal(
                data_feature_api.loc[:, ident_col], data_rollup_api.loc[:, ident_col], check_names=False
            )
