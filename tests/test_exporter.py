import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    BUILDING_NEW_ID,
    FEATURE_CLASS_DESCRIPTIONS,
    PER_CLASS_FILE_CLASS_IDS,
    ROLLUP_BUILDING_PRIMARY_CLIPPED_AREA_SQM_ID,
    ROLLUP_BUILDING_PRIMARY_UNCLIPPED_AREA_SQM_ID,
    ROLLUP_SURVEY_DATE_ID,
    ROLLUP_SYSTEM_VERSION_ID,
    ROLLUP_TREE_CANOPY_AREA_CLIPPED_SQFT_ID,
    ROLLUP_TREE_CANOPY_AREA_UNCLIPPED_SQFT_ID,
    ROLLUP_TREE_CANOPY_COUNT_ID,
    ROOF_ID,
    SQUARED_METERS_TO_SQUARED_FEET,
)
from nmaipy.exporter import (
    AOIExporter,
    _dataframe_to_records_with_index,
    _description_to_cname,
    _per_class_chunk_regexes,
    _read_parquet_chunks_parallel,
    _unify_and_concat_tables,
)
from nmaipy.feature_api import FeatureApi


class TestExporter:
    @pytest.mark.filterwarnings("ignore:.*initial implementation of Parquet.*")
    def test_process_chunk_rollup_single_multi_polygon_combo(
        self,
        parcels_3_gdf: gpd.GeoDataFrame,
        cache_directory: Path,
        processed_output_directory: Path,
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
        final_path_rollup_api = (
            output_dir_rollup_api / "final"
        )  # Permanent path for later visual inspection
        final_path_rollup_api.mkdir(parents=True, exist_ok=True)

        chunk_path_rollup_api = output_dir_rollup_api / "chunks"
        chunk_path_rollup_api.mkdir(parents=True, exist_ok=True)

        cache_path_rollup_api = output_dir_rollup_api / "cache"
        cache_path_rollup_api.mkdir(parents=True, exist_ok=True)

        feature_api = FeatureApi()
        classes_df = feature_api.get_feature_classes(packs)

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
        for cp in chunk_path_rollup_api.glob(f"feature_api_errors_*.parquet"):
            data_rollup_api_errors.append(pd.read_parquet(cp))
        if len(data_rollup_api_errors) > 0:
            data_rollup_api_errors = pd.concat(data_rollup_api_errors)
        else:
            data_rollup_api_errors = pd.DataFrame()

        data_rollup_api = []
        for cp in chunk_path_rollup_api.glob(f"rollup_*.parquet"):
            data_rollup_api.append(pd.read_parquet(cp))
        data_rollup_api = pd.concat(data_rollup_api)
    @pytest.mark.live_api
    @pytest.mark.skipif(not os.environ.get("API_KEY"), reason="API_KEY not set")
    def test_process_chunk_au(
        self,
        parcel_gdf_au_tests: gpd.GeoDataFrame,
        cache_directory: Path,
        processed_output_directory: Path,
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

        outpath_errors = final_path / f"{tag}_feature_api_errors.csv"
        for cp in chunk_path.glob(f"feature_api_errors_*.parquet"):
            errors.append(pd.read_parquet(cp))
        if len(errors) > 0:
            errors = pd.concat(errors)
        else:
            errors = pd.DataFrame()
        errors.to_csv(outpath_errors, index=True)

        for cp in [p for p in chunk_path.glob(f"features_*.parquet")]:
            data_features.append(gpd.read_parquet(cp))
        data_features = pd.concat(data_features)
        if len(data_features) > 0:
            data_features.to_file(outpath_features, driver="GPKG")

        assert outpath.exists()
        assert outpath_errors.exists()
        assert outpath_features.exists()

        assert len(data) == len(
            parcel_gdf_au_tests
        )  # Assert got a result for every parcel.

    @pytest.mark.skip(reason="Rollup API not yet updated to be compatible with output.")
    def test_process_chunk_rollup_vs_feature_calc(
        self,
        parcels_2_gdf: gpd.GeoDataFrame,
        cache_directory: Path,
        processed_output_directory: Path,
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
        final_path_rollup_api = (
            output_dir_rollup_api / "final"
        )  # Permanent path for later visual inspection
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

        for endpoint, outdir in [
            ("feature", output_dir),
            ("rollup", output_dir_rollup_api),
        ]:
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
            data_feature_api.loc[
                idx_equal_counts,
                "medium_and_high_vegetation_(>2m)_total_clipped_area_sqft",
            ],
            data_rollup_api.filter(like=ROLLUP_TREE_CANOPY_AREA_CLIPPED_SQFT_ID)
            .loc[idx_equal_counts]
            .iloc[:, 0],
            check_exact=False,
            check_names=False,
            atol=1,
        )
        pd.testing.assert_series_equal(
            data_feature_api.loc[
                idx_equal_counts,
                "medium_and_high_vegetation_(>2m)_total_unclipped_area_sqft",
            ],
            data_rollup_api.filter(like=ROLLUP_TREE_CANOPY_AREA_UNCLIPPED_SQFT_ID)
            .loc[idx_equal_counts]
            .iloc[:, 0],
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
            data_feature_api.loc[idx_equal_counts, "primary_roof_clipped_area_sqft"]
            / SQUARED_METERS_TO_SQUARED_FEET,
            data_rollup_api.filter(like=ROLLUP_BUILDING_PRIMARY_CLIPPED_AREA_SQM_ID)
            .loc[idx_equal_counts]
            .fillna(0)
            .iloc[:, 0],
            check_exact=False,
            check_names=False,
            atol=1,
        )
        pd.testing.assert_series_equal(
            data_feature_api.loc[idx_equal_counts, "primary_roof_unclipped_area_sqft"]
            / SQUARED_METERS_TO_SQUARED_FEET,
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
            data_feature_api.loc[:, "survey_date"],
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

    @pytest.mark.live_api
    @pytest.mark.skipif(not os.environ.get("API_KEY"), reason="API_KEY not set")
    def test_full_export_with_incremental_features(
        self,
        parcel_gdf_au_tests: gpd.GeoDataFrame,
        cache_directory: Path,
        processed_output_directory: Path,
        tmp_path: Path,
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

        expected_rollup_file = final_path / "rollup.csv"
        expected_features_file = final_path / "features.parquet"

        assert (
            expected_rollup_file.exists()
        ), f"Rollup CSV file was not created at {expected_rollup_file}. Found files: {list(final_path.glob('*'))}"
        assert expected_features_file.exists(), "Features parquet file was not created"

        # Verify chunk files were created
        feature_chunk_files = list(chunk_path.glob("features_*.parquet"))
        assert (
            len(feature_chunk_files) >= 1
        ), f"Expected at least one feature chunk, got {len(feature_chunk_files)}"

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
            assert len(consolidated_features) == len(
                manual_concat
            ), f"Feature count mismatch: consolidated={len(consolidated_features)}, manual_concat={len(manual_concat)}"

            # Verify CRS preservation
            if hasattr(manual_concat, "crs") and manual_concat.crs:
                assert (
                    consolidated_features.crs == manual_concat.crs
                ), "CRS not preserved"

            # Verify essential columns exist
            assert (
                "geometry" in consolidated_features.columns
            ), "Missing geometry column"
            # Features data uses 'index' column (from the original parcel index) instead of 'aoi_id'
            assert "index" in consolidated_features.columns, "Missing index column"

    def test_aoi_exporter_has_run_method(self):
        """Test that AOIExporter has run() method, not export()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = AOIExporter(
                aoi_file="data/examples/sydney_parcels.geojson",
                output_dir=tmpdir,
                country="au",
                packs=["building"],
            )

            # Check run() method exists
            assert hasattr(exporter, "run"), "AOIExporter should have run() method"
            assert callable(exporter.run), "run() should be callable"

            # Check export() method does NOT exist
            assert not hasattr(
                exporter, "export"
            ), "AOIExporter should NOT have export() method"

    def test_aoi_exporter_initialization(self):
        """Test AOIExporter can be initialized with minimal parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Minimal initialization
            exporter = AOIExporter(
                aoi_file="data/examples/sydney_parcels.geojson",
                output_dir=tmpdir,
                country="au",
                packs=["building"],
            )

            assert exporter.aoi_file == "data/examples/sydney_parcels.geojson"
            assert str(exporter.output_dir) == tmpdir  # output_dir may be Path object
            assert exporter.country == "au"
            assert exporter.packs == ["building"]
            assert exporter.processes > 0
            assert exporter.chunk_size > 0

    def test_aoi_exporter_with_invalid_country(self):
        """Test AOIExporter validates country parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = AOIExporter(
                aoi_file="data/examples/sydney_parcels.geojson",
                output_dir=tmpdir,
                country="invalid",  # Invalid country
                packs=["building"],
            )

            # The validation happens during run(), not initialization
            # So we need to mock the API call to test validation
            with patch.object(exporter, "process_chunk") as mock_process:
                with pytest.raises(Exception) as exc_info:
                    exporter.run()

                # Check that an appropriate error was raised
                # The exact error depends on implementation
                assert exc_info.value is not None

    @pytest.mark.parametrize("chunk_size", [1, 10, 100])
    def test_aoi_exporter_chunk_sizes(self, chunk_size):
        """Test that different chunk sizes work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = AOIExporter(
                aoi_file="data/examples/sydney_parcels.geojson",
                output_dir=tmpdir,
                country="au",
                packs=["building"],
                chunk_size=chunk_size,
                processes=1,  # Single process for predictable testing
            )

            assert exporter.chunk_size == chunk_size

    def test_aoi_exporter_parallel_processing(self):
        """Test that parallel processing parameters are respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with different process counts
            for processes in [1, 2, 4]:
                exporter = AOIExporter(
                    aoi_file="data/examples/sydney_parcels.geojson",
                    output_dir=tmpdir,
                    country="au",
                    packs=["building"],
                    processes=processes,
                )

                assert exporter.processes == processes

    def test_aoi_exporter_output_formats(self):
        """Test different output format options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CSV format (default)
            exporter_csv = AOIExporter(
                aoi_file="data/examples/sydney_parcels.geojson",
                output_dir=tmpdir,
                country="au",
                packs=["building"],
                tabular_file_format="csv",
            )
            assert exporter_csv.tabular_file_format == "csv"

            # Test Parquet format
            exporter_parquet = AOIExporter(
                aoi_file="data/examples/sydney_parcels.geojson",
                output_dir=tmpdir,
                country="au",
                packs=["building"],
                tabular_file_format="parquet",
            )
            assert exporter_parquet.tabular_file_format == "parquet"

    def test_aoi_exporter_save_features_flag(self):
        """Test save_features parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Without features
            exporter_no_features = AOIExporter(
                aoi_file="data/examples/sydney_parcels.geojson",
                output_dir=tmpdir,
                country="au",
                packs=["building"],
                save_features=False,
            )
            assert exporter_no_features.save_features == False

            # With features
            exporter_with_features = AOIExporter(
                aoi_file="data/examples/sydney_parcels.geojson",
                output_dir=tmpdir,
                country="au",
                packs=["building"],
                save_features=True,
            )
            assert exporter_with_features.save_features == True

    def test_stream_and_convert_features_schema_mismatch(self):
        """Test that _stream_and_convert_features handles schema mismatches correctly.

        This test simulates the scenario where:
        - Chunk 1 has features with proper schema (system_version: string, etc.)
        - Chunk 2 has no features, resulting in null-type columns

        The fix should handle this by creating properly-typed null arrays.
        """
        from shapely.geometry import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            chunk_dir = tmpdir / "chunks"
            chunk_dir.mkdir()

            # Create chunk 1 with real data
            chunk1_data = gpd.GeoDataFrame(
                {
                    "system_version": ["gen6-"],
                    "link": ["http://example.com"],
                    "survey_date": ["2024-11-06"],
                    "survey_id": ["survey1"],
                    "survey_resource_id": ["resource1"],
                    "perspective": ["vertical"],
                    "postcat": [True],
                    "feature_id": ["feat1"],
                    "class_id": ["class1"],
                    "internal_class_id": [1],
                    "description": ["Test feature"],
                    "geometry": [Point(0, 0)],
                },
                crs="EPSG:4326",
            )
            chunk1_path = chunk_dir / "features_test_1.parquet"
            chunk1_data.to_parquet(chunk1_path)

            # Create chunk 2 with null-type columns (simulating empty features)
            # This mimics what happens when all addresses in a chunk have no features
            chunk2_data = gpd.GeoDataFrame(
                {
                    "system_version": [None],
                    "link": [None],
                    "survey_date": [None],
                    "survey_id": [None],
                    "survey_resource_id": [None],
                    "perspective": [None],
                    "postcat": [None],
                    "feature_id": [None],
                    "class_id": [None],
                    "internal_class_id": [None],
                    "description": [None],
                    "geometry": [Point(1, 1)],
                },
                crs="EPSG:4326",
            )
            chunk2_path = chunk_dir / "features_test_2.parquet"
            chunk2_data.to_parquet(chunk2_path)

            # Now test the streaming function
            exporter = AOIExporter(
                output_dir=tmpdir,
                country="au",
                packs=["building"],
                save_features=True,
            )

            output_path = tmpdir / "merged_features.parquet"
            feature_paths = [chunk1_path, chunk2_path]

            # This should not raise an error
            result = exporter._stream_and_convert_features(feature_paths, output_path)

            # Verify the output file exists and can be read
            assert output_path.exists()
            merged_data = gpd.read_parquet(output_path)
            assert len(merged_data) == 2
            # Verify first row has data, second row has nulls
            assert merged_data.iloc[0]["system_version"] == "gen6-"
            assert pd.isna(merged_data.iloc[1]["system_version"])

    def test_stream_and_convert_features_null_columns_no_warning(self, caplog):
        """Test that null-type columns in later chunks are promoted silently without warnings."""
        from shapely.geometry import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            chunk_dir = tmpdir / "chunks"
            chunk_dir.mkdir()

            # Chunk 1: has data in all columns
            chunk1_data = gpd.GeoDataFrame(
                {
                    "system_version": ["gen6-"],
                    "lob_bv": ["homeowner"],
                    "geometry": [Point(0, 0)],
                },
                crs="EPSG:4326",
            )
            chunk1_data.to_parquet(chunk_dir / "features_1.parquet")

            # Chunk 2: lob_bv is all null (simulates empty passthrough column)
            chunk2_data = gpd.GeoDataFrame(
                {
                    "system_version": ["gen6-"],
                    "lob_bv": [None],
                    "geometry": [Point(1, 1)],
                },
                crs="EPSG:4326",
            )
            chunk2_data.to_parquet(chunk_dir / "features_2.parquet")

            exporter = AOIExporter(
                output_dir=tmpdir,
                country="au",
                packs=["building"],
                save_features=True,
            )
            output_path = tmpdir / "merged.parquet"
            feature_paths = sorted(chunk_dir.glob("*.parquet"))

            with caplog.at_level(logging.WARNING, logger="nmaipy"):
                exporter._stream_and_convert_features(feature_paths, output_path)

            # Should NOT produce per-chunk warning messages about schema casting
            warning_msgs = [
                r.message for r in caplog.records if r.levelno >= logging.WARNING
            ]
            assert not any(
                "Schema casting failed" in m for m in warning_msgs
            ), f"Should not produce noisy schema casting warnings, got: {warning_msgs}"
            assert not any(
                "Incompatible columns" in m for m in warning_msgs
            ), f"Null-type promotions should not produce incompatible column warnings"

            # Verify data is correct
            merged = gpd.read_parquet(output_path)
            assert len(merged) == 2
            assert merged.iloc[0]["lob_bv"] == "homeowner"
            assert pd.isna(merged.iloc[1]["lob_bv"])

    def test_stream_and_convert_features_reference_schema_null_promotion(self):
        """Test that null-type columns in the first chunk are promoted to string in reference schema."""
        from shapely.geometry import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            chunk_dir = tmpdir / "chunks"
            chunk_dir.mkdir()

            # Chunk 1: lob_bv is all null (first chunk establishes reference schema)
            chunk1_data = gpd.GeoDataFrame(
                {
                    "system_version": ["gen6-"],
                    "lob_bv": [None],
                    "geometry": [Point(0, 0)],
                },
                crs="EPSG:4326",
            )
            chunk1_data.to_parquet(chunk_dir / "features_1.parquet")

            # Chunk 2: lob_bv has data
            chunk2_data = gpd.GeoDataFrame(
                {
                    "system_version": ["gen6-"],
                    "lob_bv": ["homeowner"],
                    "geometry": [Point(1, 1)],
                },
                crs="EPSG:4326",
            )
            chunk2_data.to_parquet(chunk_dir / "features_2.parquet")

            exporter = AOIExporter(
                output_dir=tmpdir,
                country="au",
                packs=["building"],
                save_features=True,
            )
            output_path = tmpdir / "merged.parquet"
            feature_paths = sorted(chunk_dir.glob("*.parquet"))

            exporter._stream_and_convert_features(feature_paths, output_path)

            # Verify data is correct even when first chunk had null column
            merged = gpd.read_parquet(output_path)
            assert len(merged) == 2
            assert pd.isna(merged.iloc[0]["lob_bv"])
            assert merged.iloc[1]["lob_bv"] == "homeowner"

            # Verify the parquet schema has large_string type (not null) for lob_bv.
            # large_string uses 64-bit offsets to avoid overflow on large exports.
            schema = pq.read_schema(output_path)
            lob_field = schema.field("lob_bv")
            assert (
                lob_field.type == pa.large_string()
            ), f"Expected large_string type for promoted null column, got {lob_field.type}"


class TestReadParquetChunksParallel:
    """Tests for _read_parquet_chunks_parallel."""

    def test_reads_valid_parquet_files(self, tmp_path):
        paths = []
        for i in range(3):
            df = pd.DataFrame({"col": [i, i + 10]})
            path = tmp_path / f"chunk_{i}.parquet"
            df.to_parquet(path)
            paths.append(str(path))

        results = _read_parquet_chunks_parallel(paths, max_workers=2)
        assert len(results) == 3
        total_rows = sum(len(df) for df in results)
        assert total_rows == 6

    def test_reads_geoparquet_files(self, tmp_path):
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame({"val": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        path = tmp_path / "geo_chunk.parquet"
        gdf.to_parquet(path)

        results = _read_parquet_chunks_parallel([str(path)], max_workers=1)
        assert len(results) == 1
        assert isinstance(results[0], gpd.GeoDataFrame)

    def test_excludes_empty_dataframes(self, tmp_path):
        empty_path = tmp_path / "empty.parquet"
        pd.DataFrame({"col": pd.Series([], dtype="int64")}).to_parquet(empty_path)

        nonempty_path = tmp_path / "nonempty.parquet"
        pd.DataFrame({"col": [1]}).to_parquet(nonempty_path)

        results = _read_parquet_chunks_parallel(
            [str(empty_path), str(nonempty_path)], max_workers=2
        )
        assert len(results) == 1

    def test_empty_paths_returns_empty_list(self):
        results = _read_parquet_chunks_parallel([])
        assert results == []

    def test_strict_true_raises_on_failure(self, tmp_path):
        bad_path = tmp_path / "bad.parquet"
        bad_path.write_text("not a parquet file")

        with pytest.raises(RuntimeError, match="Failed to read"):
            _read_parquet_chunks_parallel([str(bad_path)], max_workers=1, strict=True)

    def test_strict_false_tolerates_failure(self, tmp_path):
        bad_path = tmp_path / "bad.parquet"
        bad_path.write_text("not a parquet file")

        good_path = tmp_path / "good.parquet"
        pd.DataFrame({"col": [1]}).to_parquet(good_path)

        logger = logging.getLogger("test_parallel_read")
        results = _read_parquet_chunks_parallel(
            [str(bad_path), str(good_path)],
            max_workers=2,
            logger=logger,
            strict=False,
        )
        assert len(results) == 1

    def test_strict_true_is_default(self, tmp_path):
        bad_path = tmp_path / "bad.parquet"
        bad_path.write_text("not a parquet file")

        with pytest.raises(RuntimeError):
            _read_parquet_chunks_parallel([str(bad_path)], max_workers=1)

    def test_summary_distinguishes_failures_from_empties(self, tmp_path, caplog):
        bad_path = tmp_path / "bad.parquet"
        bad_path.write_text("not a parquet file")

        empty_path = tmp_path / "empty.parquet"
        pd.DataFrame({"col": pd.Series([], dtype="int64")}).to_parquet(empty_path)

        good_path = tmp_path / "good.parquet"
        pd.DataFrame({"col": [1]}).to_parquet(good_path)

        # Use a test-specific logger (nmaipy logger has propagate=False,
        # preventing caplog from capturing its records)
        logger = logging.getLogger("test_parallel_read_summary")
        with caplog.at_level(logging.INFO, logger="test_parallel_read_summary"):
            results = _read_parquet_chunks_parallel(
                [str(bad_path), str(empty_path), str(good_path)],
                max_workers=2,
                logger=logger,
                strict=False,
            )

        assert len(results) == 1
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        summary = [m for m in info_messages if "1/3" in m]
        assert len(summary) == 1, f"Expected one summary message, got: {info_messages}"
        assert "1 failed" in summary[0]
        assert "1 empty" in summary[0]


class TestPerClassFileWhitelist:
    """Tests for PER_CLASS_FILE_CLASS_IDS whitelist filtering on real cached data."""

    def test_whitelist_filters_real_feature_classes(self, features_2_all_packs_df):
        """Using real NJ data with all Feature API packs + Roof Age API, verify
        that all 6 whitelisted classes are present and the whitelist correctly
        partitions classes into export vs skip sets.

        Generated by test_gen_data_2_all_packs in test_parcels.py (100 NJ parcels,
        all packs, plus roof age for first 10 parcels).
        """
        from nmaipy.constants import (
            BUILDING_LIFECYCLE_ID,
            POOL_ID,
            ROOF_INSTANCE_CLASS_ID,
            SOLAR_ID,
        )

        unique_classes = set(features_2_all_packs_df["class_id"].unique())
        whitelisted = unique_classes & PER_CLASS_FILE_CLASS_IDS
        skipped = unique_classes - PER_CLASS_FILE_CLASS_IDS

        # All 6 whitelisted classes should be present in the all-packs data
        for class_id, name in [
            (BUILDING_NEW_ID, "building (new semantic)"),
            (ROOF_ID, "roof"),
            (POOL_ID, "pool"),
            (SOLAR_ID, "solar"),
            (BUILDING_LIFECYCLE_ID, "building lifecycle"),
            (ROOF_INSTANCE_CLASS_ID, "roof instance"),
        ]:
            assert class_id in unique_classes, (
                f"{name} ({class_id}) should be present in all-packs NJ data"
            )

        assert whitelisted == PER_CLASS_FILE_CLASS_IDS, (
            f"All 6 whitelisted classes should be present. "
            f"Missing: {PER_CLASS_FILE_CLASS_IDS - whitelisted}"
        )

        # Non-whitelisted classes should be numerous (vegetation, surfaces, etc.)
        assert len(skipped) >= 20, (
            f"Expected 20+ non-whitelisted classes, got {len(skipped)}"
        )

        # Every class in data must be in exactly one partition
        assert whitelisted | skipped == unique_classes

        # Verify every row is accounted for — no data lost by the partition
        total_rows = len(features_2_all_packs_df)
        whitelisted_rows = features_2_all_packs_df[
            features_2_all_packs_df["class_id"].isin(PER_CLASS_FILE_CLASS_IDS)
        ]
        skipped_rows = features_2_all_packs_df[
            ~features_2_all_packs_df["class_id"].isin(PER_CLASS_FILE_CLASS_IDS)
        ]
        assert len(whitelisted_rows) + len(skipped_rows) == total_rows, (
            "Whitelist partition must account for all rows"
        )
        assert len(whitelisted_rows) > 0, "Should have some whitelisted feature rows"
        assert len(skipped_rows) > 0, "Should have some skipped feature rows"


class TestDataframeToRecordsWithIndex:
    """Tests for _dataframe_to_records_with_index helper."""

    def test_preserves_aoi_id_from_index(self):
        """When the DataFrame is indexed by AOI_ID_COLUMN_NAME, the helper re-injects it."""
        df = pd.DataFrame(
            {"value": [10, 20, 30]},
            index=pd.Index(["aoi_a", "aoi_b", "aoi_c"], name=AOI_ID_COLUMN_NAME),
        )
        records = _dataframe_to_records_with_index(df)
        assert len(records) == 3
        assert all(AOI_ID_COLUMN_NAME in r for r in records)
        assert records[0][AOI_ID_COLUMN_NAME] == "aoi_a"
        assert records[1][AOI_ID_COLUMN_NAME] == "aoi_b"
        assert records[2][AOI_ID_COLUMN_NAME] == "aoi_c"

    def test_non_aoi_index_not_injected(self):
        """When the index is not AOI_ID_COLUMN_NAME, it should not be injected."""
        df = pd.DataFrame(
            {"value": [10, 20]},
            index=pd.Index([0, 1], name="other_index"),
        )
        records = _dataframe_to_records_with_index(df)
        assert len(records) == 2
        assert AOI_ID_COLUMN_NAME not in records[0]

    def test_aoi_id_already_in_columns(self):
        """When AOI_ID is both the index name and in records, it gets the index value."""
        df = pd.DataFrame(
            {AOI_ID_COLUMN_NAME: ["wrong_a", "wrong_b"], "value": [1, 2]},
        )
        df.index = pd.Index(["correct_a", "correct_b"], name=AOI_ID_COLUMN_NAME)
        records = _dataframe_to_records_with_index(df)
        # The index value should overwrite since it's set after to_dict
        assert records[0][AOI_ID_COLUMN_NAME] == "correct_a"

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty list."""
        df = pd.DataFrame(
            {"value": pd.Series([], dtype="int64")},
        )
        df.index.name = AOI_ID_COLUMN_NAME
        records = _dataframe_to_records_with_index(df)
        assert records == []


class TestStreamAndConvertFeaturesSchemaUnion:
    """Tests for the schema union pre-scan in _stream_and_convert_features."""

    def test_union_schema_includes_columns_from_all_chunks(self):
        """Columns present in later chunks but not in chunk 0 should be preserved."""
        from shapely.geometry import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            chunk_dir = tmpdir / "chunks"
            chunk_dir.mkdir()

            # Chunk 1: has col_a but not col_b
            chunk1 = gpd.GeoDataFrame(
                {"col_a": ["x"], "class_id": ["c1"], "geometry": [Point(0, 0)]},
                crs="EPSG:4326",
            )
            chunk1.to_parquet(chunk_dir / "features_1.parquet")

            # Chunk 2: has col_b but not col_a
            chunk2 = gpd.GeoDataFrame(
                {"col_b": [42], "class_id": ["c1"], "geometry": [Point(1, 1)]},
                crs="EPSG:4326",
            )
            chunk2.to_parquet(chunk_dir / "features_2.parquet")

            exporter = AOIExporter(
                output_dir=tmpdir, country="au", packs=["building"], save_features=True
            )
            output_path = tmpdir / "merged.parquet"
            feature_paths = sorted(chunk_dir.glob("*.parquet"))

            exporter._stream_and_convert_features(feature_paths, output_path)

            merged = gpd.read_parquet(output_path)
            assert "col_a" in merged.columns, "col_a from chunk 1 should be in union"
            assert "col_b" in merged.columns, "col_b from chunk 2 should be in union"
            assert len(merged) == 2

    def test_corrupt_chunk_skipped_during_schema_scan(self):
        """A corrupt chunk file should be skipped, not abort the entire export."""
        from shapely.geometry import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            chunk_dir = tmpdir / "chunks"
            chunk_dir.mkdir()

            # Valid chunk
            chunk1 = gpd.GeoDataFrame(
                {"value": ["a"], "class_id": ["c1"], "geometry": [Point(0, 0)]},
                crs="EPSG:4326",
            )
            chunk1.to_parquet(chunk_dir / "features_1.parquet")

            # Corrupt chunk
            corrupt_path = chunk_dir / "features_2.parquet"
            corrupt_path.write_text("not a parquet file")

            exporter = AOIExporter(
                output_dir=tmpdir, country="au", packs=["building"], save_features=True
            )
            output_path = tmpdir / "merged.parquet"
            feature_paths = sorted(chunk_dir.glob("*.parquet"))

            # Should succeed despite corrupt chunk
            exporter._stream_and_convert_features(feature_paths, output_path)

            merged = gpd.read_parquet(output_path)
            assert len(merged) == 1
            assert merged.iloc[0]["value"] == "a"

    def test_large_type_promotion(self):
        """string and binary columns are promoted to large_string/large_binary to prevent offset overflow."""
        from shapely.geometry import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            chunk_dir = tmpdir / "chunks"
            chunk_dir.mkdir()

            # Create a chunk with pa.string() and pa.binary() columns
            table = pa.table(
                {
                    "text_col": pa.array(["hello"], type=pa.string()),
                    "bin_col": pa.array([b"\x00\x01"], type=pa.binary()),
                    "int_col": pa.array([1], type=pa.int64()),
                    "class_id": pa.array(["c1"], type=pa.string()),
                    "geometry": pa.array([Point(0, 0).wkb], type=pa.binary()),
                },
            )
            # Write with geopandas so we get valid geoparquet metadata
            gdf = gpd.GeoDataFrame(
                {"text_col": ["hello"], "bin_col": [b"\x00\x01"], "int_col": [1], "class_id": ["c1"]},
                geometry=[Point(0, 0)],
                crs="EPSG:4326",
            )
            gdf.to_parquet(chunk_dir / "features_1.parquet")

            exporter = AOIExporter(
                output_dir=tmpdir, country="au", packs=["building"], save_features=True
            )
            output_path = tmpdir / "merged.parquet"
            feature_paths = sorted(chunk_dir.glob("*.parquet"))

            exporter._stream_and_convert_features(feature_paths, output_path)

            schema = pq.read_schema(output_path)
            text_type = schema.field("text_col").type
            assert text_type == pa.large_string(), f"Expected large_string, got {text_type}"

    def test_all_corrupt_chunks_returns_none(self):
        """If all chunks are corrupt, _stream_and_convert_features returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            chunk_dir = tmpdir / "chunks"
            chunk_dir.mkdir()

            corrupt1 = chunk_dir / "features_1.parquet"
            corrupt1.write_text("bad1")
            corrupt2 = chunk_dir / "features_2.parquet"
            corrupt2.write_text("bad2")

            exporter = AOIExporter(
                output_dir=tmpdir, country="au", packs=["building"], save_features=True
            )
            output_path = tmpdir / "merged.parquet"
            feature_paths = sorted(chunk_dir.glob("*.parquet"))

            result = exporter._stream_and_convert_features(feature_paths, output_path)
            assert result is None


class TestUnifyAndConcatTables:
    def test_int64_vs_float64_promotion(self):
        """int64 and float64 columns are unified to float64 — reproduces the
        primary_child_roof_roof_spotlight_index crash where chunks with all-present
        integer values are int64 but chunks with nulls are inferred as float64."""
        t1 = pa.table({"spotlight_index": pa.array([80], type=pa.int64()), "name": ["a"]})
        t2 = pa.table({"spotlight_index": pa.array([75, None], type=pa.float64()), "name": ["b", "c"]})
        result = _unify_and_concat_tables([t1, t2])
        assert result.schema.field("spotlight_index").type == pa.float64()
        assert result.column("spotlight_index").to_pylist() == [80.0, 75.0, None]
        assert len(result) == 3

    def test_null_column_promoted(self):
        """A null-typed column in one table is promoted to the real type from another."""
        t1 = pa.table({"val": pa.array([None], type=pa.null()), "id": [1]})
        t2 = pa.table({"val": pa.array([3.14], type=pa.float64()), "id": [2]})
        result = _unify_and_concat_tables([t1, t2])
        assert result.schema.field("val").type == pa.float64()
        assert result.column("val").to_pylist() == [None, 3.14]

    def test_single_table_passthrough(self):
        """Single-table input is returned with type promotions applied."""
        t = pa.table({"text": pa.array(["hello"], type=pa.string())})
        result = _unify_and_concat_tables([t])
        assert result.schema.field("text").type == pa.large_string()
        assert len(result) == 1

    def test_missing_columns_padded_with_nulls(self):
        """Tables with different column sets get null-padded."""
        t1 = pa.table({"a": [1], "b": [2]})
        t2 = pa.table({"a": [3], "c": [4]})
        result = _unify_and_concat_tables([t1, t2])
        assert set(result.column_names) == {"a", "b", "c"}
        assert result.column("b").to_pylist() == [2, None]
        assert result.column("c").to_pylist() == [None, 4]

    def test_int32_vs_int64_widening(self):
        """Integer width mismatches are widened to the larger type."""
        t1 = pa.table({"count": pa.array([1], type=pa.int32())})
        t2 = pa.table({"count": pa.array([2], type=pa.int64())})
        result = _unify_and_concat_tables([t1, t2])
        assert result.schema.field("count").type == pa.int64()
        assert result.column("count").to_pylist() == [1, 2]

    def test_string_vs_large_string(self):
        """string and large_string are unified to large_string."""
        t1 = pa.table({"name": pa.array(["a"], type=pa.string())})
        t2 = pa.table({"name": pa.array(["b"], type=pa.large_string())})
        result = _unify_and_concat_tables([t1, t2])
        assert result.schema.field("name").type == pa.large_string()
        assert result.column("name").to_pylist() == ["a", "b"]

    def test_timestamp_unit_mismatch(self):
        """Timestamp columns with different units are unified."""
        t1 = pa.table({"ts": pa.array([1000000], type=pa.timestamp("us"))})
        t2 = pa.table({"ts": pa.array([1000000000], type=pa.timestamp("ns"))})
        result = _unify_and_concat_tables([t1, t2])
        assert pa.types.is_timestamp(result.schema.field("ts").type)
        assert len(result) == 2


class TestPerClassChunkGlobFiltering:
    """Tests that per-class chunk file matching prevents glob collisions.

    Derives all filenames from PER_CLASS_FILE_CLASS_IDS and FEATURE_CLASS_DESCRIPTIONS
    so the test breaks if the naming convention changes.
    """

    @pytest.fixture()
    def per_class_cnames(self):
        """Build {class_id: cname} for all whitelisted per-class file classes."""
        return {
            cid: _description_to_cname(FEATURE_CLASS_DESCRIPTIONS[cid])
            for cid in PER_CLASS_FILE_CLASS_IDS
        }

    def test_each_class_matches_own_chunks(self, per_class_cnames):
        """Every class's regex matches its own tabular and geo chunk filenames."""
        for cid, cname in per_class_cnames.items():
            tabular_re, geo_re = _per_class_chunk_regexes(cname)
            tabular_file = f"{cname}_0.parquet"
            geo_file = f"{cname}_features_0.parquet"
            assert tabular_re.match(tabular_file), (
                f"{cname} tabular regex should match {tabular_file}"
            )
            assert geo_re.match(geo_file), (
                f"{cname} geo regex should match {geo_file}"
            )

    def test_no_class_matches_another_class_chunks(self, per_class_cnames):
        """No class's regex matches chunk files belonging to a different class."""
        cnames = list(per_class_cnames.values())
        for cname in cnames:
            tabular_re, geo_re = _per_class_chunk_regexes(cname)
            for other_cname in cnames:
                if other_cname == cname:
                    continue
                other_tabular = f"{other_cname}_0.parquet"
                other_geo = f"{other_cname}_features_0.parquet"
                assert not tabular_re.match(other_tabular), (
                    f"{cname!r} tabular regex must not match {other_tabular!r}"
                )
                assert not geo_re.match(other_geo), (
                    f"{cname!r} geo regex must not match {other_geo!r}"
                )

    def test_tabular_regex_excludes_geo_files(self, per_class_cnames):
        """Tabular regex must not match geo chunk files for the same class."""
        for cid, cname in per_class_cnames.items():
            tabular_re, _ = _per_class_chunk_regexes(cname)
            geo_file = f"{cname}_features_0.parquet"
            assert not tabular_re.match(geo_file), (
                f"{cname!r} tabular regex must not match geo file {geo_file!r}"
            )


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    sys.exit(pytest.main([current_file]))
