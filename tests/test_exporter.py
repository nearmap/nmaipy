import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from nmaipy.constants import *
from nmaipy.exporter import AOIExporter
from nmaipy.feature_api import FeatureApi
from unittest.mock import patch, MagicMock
import tempfile


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
        print("data rollup api")
        print(data_rollup_api.T)
        print(data_rollup_api.loc[:, "link"].values)

    @pytest.mark.live_api
    @pytest.mark.skipif(not os.environ.get('API_KEY'), reason="API_KEY not set")
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

    @pytest.mark.live_api
    @pytest.mark.skipif(not os.environ.get('API_KEY'), reason="API_KEY not set")
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
        
        # Note: Output filename changed from test_aoi.csv to test_aoi_aoi_rollup.csv
        expected_rollup_file = final_path / "test_aoi_aoi_rollup.csv"
        expected_features_file = final_path / "test_aoi_features.parquet"

        assert expected_rollup_file.exists(), f"Rollup CSV file was not created at {expected_rollup_file}. Found files: {list(final_path.glob('*'))}"
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
    
    def test_aoi_exporter_has_run_method(self):
        """Test that AOIExporter has run() method, not export()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=tmpdir,
                country='au',
                packs=['building'],
            )
            
            # Check run() method exists
            assert hasattr(exporter, 'run'), "AOIExporter should have run() method"
            assert callable(exporter.run), "run() should be callable"
            
            # Check export() method does NOT exist
            assert not hasattr(exporter, 'export'), "AOIExporter should NOT have export() method"
    
    def test_aoi_exporter_initialization(self):
        """Test AOIExporter can be initialized with minimal parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Minimal initialization
            exporter = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=tmpdir,
                country='au',
                packs=['building'],
            )

            assert exporter.aoi_file == 'data/examples/sydney_parcels.geojson'
            assert str(exporter.output_dir) == tmpdir  # output_dir may be Path object
            assert exporter.country == 'au'
            assert exporter.packs == ['building']
            assert exporter.processes > 0
            assert exporter.chunk_size > 0
    
    def test_aoi_exporter_with_invalid_country(self):
        """Test AOIExporter validates country parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=tmpdir,
                country='invalid',  # Invalid country
                packs=['building'],
            )
            
            # The validation happens during run(), not initialization
            # So we need to mock the API call to test validation
            with patch.object(exporter, 'process_chunk') as mock_process:
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
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=tmpdir,
                country='au',
                packs=['building'],
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
                    aoi_file='data/examples/sydney_parcels.geojson',
                    output_dir=tmpdir,
                    country='au',
                    packs=['building'],
                    processes=processes,
                )
                
                assert exporter.processes == processes
    
    def test_aoi_exporter_output_formats(self):
        """Test different output format options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CSV format (default)
            exporter_csv = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=tmpdir,
                country='au',
                packs=['building'],
                rollup_format='csv',
            )
            assert exporter_csv.rollup_format == 'csv'
            
            # Test Parquet format
            exporter_parquet = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=tmpdir,
                country='au',
                packs=['building'],
                rollup_format='parquet',
            )
            assert exporter_parquet.rollup_format == 'parquet'
    
    def test_aoi_exporter_save_features_flag(self):
        """Test save_features parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Without features
            exporter_no_features = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=tmpdir,
                country='au',
                packs=['building'],
                save_features=False,
            )
            assert exporter_no_features.save_features == False
            
            # With features
            exporter_with_features = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=tmpdir,
                country='au',
                packs=['building'],
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
        import pyarrow as pa
        import pyarrow.parquet as pq
        from shapely.geometry import Point

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            chunk_dir = tmpdir / "chunks"
            chunk_dir.mkdir()

            # Create chunk 1 with real data
            chunk1_data = gpd.GeoDataFrame({
                'system_version': ['gen6-'],
                'link': ['http://example.com'],
                'date': ['2024-11-06'],
                'survey_id': ['survey1'],
                'survey_resource_id': ['resource1'],
                'perspective': ['vertical'],
                'postcat': [True],
                'feature_id': ['feat1'],
                'class_id': ['class1'],
                'internal_class_id': [1],
                'description': ['Test feature'],
                'geometry': [Point(0, 0)]
            }, crs='EPSG:4326')
            chunk1_path = chunk_dir / "features_test_1.parquet"
            chunk1_data.to_parquet(chunk1_path)

            # Create chunk 2 with null-type columns (simulating empty features)
            # This mimics what happens when all addresses in a chunk have no features
            chunk2_data = gpd.GeoDataFrame({
                'system_version': [None],
                'link': [None],
                'date': [None],
                'survey_id': [None],
                'survey_resource_id': [None],
                'perspective': [None],
                'postcat': [None],
                'feature_id': [None],
                'class_id': [None],
                'internal_class_id': [None],
                'description': [None],
                'geometry': [Point(1, 1)]
            }, crs='EPSG:4326')
            chunk2_path = chunk_dir / "features_test_2.parquet"
            chunk2_data.to_parquet(chunk2_path)

            # Now test the streaming function
            exporter = AOIExporter(
                output_dir=tmpdir,
                country='au',
                packs=['building'],
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
            assert merged_data.iloc[0]['system_version'] == 'gen6-'
            assert pd.isna(merged_data.iloc[1]['system_version'])


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    sys.exit(pytest.main([current_file]))
