import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from nmaipy.constants import *
from nmaipy.exporter import AOIExporter
from nmaipy.feature_api import FeatureApi


class TestExporter:
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

        output_dir = Path(processed_output_directory) / tag
        output_dir_rollup_api = Path(processed_output_directory) / tag_rollup_api
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
        my_exporter = AOIExporter(
            output_dir=output_dir_rollup_api,
            country=country,
            packs=packs,
            include_parcel_geometry=True,
            save_features=False,
            since="2022-10-29",
            until="2022-10-29",
            alpha=False,
            beta=False,
            endpoint="rollup",
            save_buildings=True,
        )
        my_exporter.process_chunk(
            chunk_id=chunk_id,
            aoi_gdf=parcels_3_gdf,
            classes_df=classes_df,
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

        output_dir = Path(processed_output_directory) / tag
        packs = ["building", "vegetation"]
        country = "au"
        final_path = output_dir / "final"  # Permanent path for later visual inspection
        final_path.mkdir(parents=True, exist_ok=True)

        chunk_path = output_dir / "chunks"
        chunk_path.mkdir(parents=True, exist_ok=True)

        cache_path = output_dir / "cache"
        cache_path.mkdir(parents=True, exist_ok=True)

        feature_api = FeatureApi()
        classes_df = feature_api.get_feature_classes(packs)

        my_exporter = AOIExporter(
            output_dir=output_dir,
            country=country,
            packs=packs,
            include_parcel_geometry=True,
            save_features=True,
            no_cache=True,
            parcel_mode=True,
            system_version_prefix="gen6-",
            processes=8,
            save_buildings=True,
        )
        my_exporter.process_chunk(
            chunk_id=chunk_id,
            aoi_gdf=parcel_gdf_au_tests,
            classes_df=classes_df,
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

    @pytest.mark.skip(reason="Rollup API not yet updated to be compatible with output.")
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

        output_dir = Path(processed_output_directory) / tag
        output_dir_rollup_api = Path(processed_output_directory) / tag_rollup_api
        packs = ["building", "roof_char", "vegetation"]
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
            my_exporter = AOIExporter(
                output_dir=outdir,
                country=country,
                packs=packs,
                include_parcel_geometry=True,
                save_features=False,
                since="2022-06-29",
                until="2022-06-29",
                alpha=False,
                beta=False,
                prerelease=False,
                endpoint=endpoint,
                processes=8,
                threads=20,
            )
            my_exporter.process_chunk(
                chunk_id=chunk_id,
                aoi_gdf=parcels_2_gdf,
                classes_df=classes_df,
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
        # TODO: Enable once we've reconciled formats and filtering rules with rollup API.
        # pd.testing.assert_series_equal(
        #     data_feature_api.loc[:, "roof_count"],
        #     data_rollup_api.filter(like=ROLLUP_BUILDING_COUNT_ID).iloc[:, 0],
        #     check_names=False,
        #     atol=1,
        # )

        ## Check small error tolerance (max 1 square foot), only where there was no in/out discrepancy on counts
        # TODO: Enable once we've reconciled formats and filtering rules with rollup API.
        # idx_equal_counts = (
        #     data_feature_api.loc[:, "roof_count"] - data_rollup_api.filter(like=ROLLUP_BUILDING_COUNT_ID).iloc[:, 0]
        # ) == 0
        # assert idx_equal_counts.sum() == len(idx_equal_counts)

        ## Implicitly test sqm to sqft conversion...
        pd.testing.assert_series_equal(
            data_feature_api.loc[idx_equal_counts, "primary_roof_clipped_area_sqft"] / SQUARED_METERS_TO_SQUARED_FEET,
            data_rollup_api.filter(like=ROLLUP_BUILDING_PRIMARY_CLIPPED_AREA_SQM_ID)
            .loc[idx_equal_counts]
            .fillna(0)
            .iloc[:, 0],
            check_exact=False,
            check_names=False,
            atol=1,
        )
        pd.testing.assert_series_equal(
            data_feature_api.loc[idx_equal_counts, "primary_roof_unclipped_area_sqft"] / SQUARED_METERS_TO_SQUARED_FEET,
            data_rollup_api.filter(like=ROLLUP_BUILDING_PRIMARY_UNCLIPPED_AREA_SQM_ID)
            .loc[idx_equal_counts]
            .fillna(0)
            .iloc[:, 0],
            check_exact=False,
            check_names=False,
            atol=1,
        )

        # TODO: Enable once fidelity score in rollup API.
        # ## Test confidence aggregation is correct to within 1%
        # pd.testing.assert_series_equal(
        #     data_feature_api.loc[idx_equal_counts, "roof_confidence"],
        #     data_rollup_api.filter(like="roof presence confidence").loc[idx_equal_counts].iloc[:, 0],
        #     check_exact=False,
        #     check_names=False,
        #     rtol=0.01,
        # )

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
            pass  # TODO: Enable once we've reconciled formats and filtering rules with rollup API.
            # pd.testing.assert_series_equal(
            #     data_feature_api.loc[:, ident_col], data_rollup_api.loc[:, ident_col], check_names=False
            # )

    def test_full_export_with_incremental_features(
        self, parcel_gdf_au_tests: gpd.GeoDataFrame, cache_directory: Path, processed_output_directory: Path, tmp_path: Path
    ):
        """
        Test the full export workflow with save_features=True to validate the new incremental
        parquet writing works correctly in the actual run() method.
        """
        # Create a temporary AOI file
        aoi_file = tmp_path / "test_aoi.geojson"
        parcel_gdf_au_tests.to_file(aoi_file, driver="GeoJSON")
        
        output_dir = processed_output_directory / "test_full_incremental"
        
        # Run the full export with features enabled
        exporter = AOIExporter(
            aoi_file=str(aoi_file),
            output_dir=str(output_dir),
            country="au",
            packs=["building"],
            save_features=True,
            chunk_size=2,  # Small chunks to ensure multiple feature files
            no_cache=True,
            system_version_prefix="gen6-",
            processes=1,  # Single process for testing
        )
        
        exporter.run()
        
        # Verify outputs exist
        final_path = output_dir / "final"
        chunk_path = output_dir / "chunks"
        
        expected_rollup_file = final_path / "test_aoi.csv"
        expected_features_file = final_path / "test_aoi_features.parquet"
        
        assert expected_rollup_file.exists(), "Rollup CSV file was not created"
        assert expected_features_file.exists(), "Features parquet file was not created"
        
        # Verify chunk files were created
        feature_chunk_files = list(chunk_path.glob("features_test_aoi_*.parquet"))
        assert len(feature_chunk_files) >= 1, f"Expected at least one feature chunk, got {len(feature_chunk_files)}"
        
        # Load and validate the consolidated features file
        consolidated_features = gpd.read_parquet(expected_features_file)
        
        # Load individual chunks for comparison
        chunk_data = []
        for chunk_file in feature_chunk_files:
            chunk_gdf = gpd.read_parquet(chunk_file)
            if len(chunk_gdf) > 0:
                chunk_data.append(chunk_gdf)
        
        if chunk_data:
            manual_concat = pd.concat(chunk_data, ignore_index=True)
            
            # Verify same number of features
            assert len(consolidated_features) == len(manual_concat), \
                f"Feature count mismatch: consolidated={len(consolidated_features)}, manual_concat={len(manual_concat)}"
            
            # Verify CRS preservation
            if hasattr(manual_concat, 'crs') and manual_concat.crs:
                assert consolidated_features.crs == manual_concat.crs, "CRS not preserved"
            
            # Verify essential columns exist
            assert 'geometry' in consolidated_features.columns, "Missing geometry column"
            # Features data uses 'index' column (from the original parcel index) instead of 'aoi_id'
            assert 'index' in consolidated_features.columns, "Missing index column"


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    sys.exit(pytest.main([current_file]))
