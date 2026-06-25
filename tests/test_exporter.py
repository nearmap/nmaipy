import json
import logging
import os
import sys
import tempfile
import threading
import time
import warnings
import weakref
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import pytest
from shapely.geometry import Polygon

import nmaipy.exporter as exporter_mod
from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    BUILDING_NEW_ID,
    FEATURE_CLASS_DESCRIPTIONS,
    FEATURE_PREFETCH_FLOOR,
    PER_CLASS_FILE_CLASS_IDS,
    ROOF_ID,
)
from nmaipy.exporter import (
    AOIExporter,
    _add_missing_columns,
    _dataframe_to_records_with_index,
    _description_to_cname,
    _parquet_to_csv_streaming,
    _per_class_chunk_regexes,
    _read_parquet_chunks_parallel,
    _resolve_merge_prefetch_workers,
    _resolve_prefetch_workers,
    _staged_file_needs_upload,
    _stream_merge_chunks_to_parquet,
    _unify_and_concat_tables,
    _write_errors_parquet,
)
from nmaipy.feature_api import FeatureApi


class TestResolvePrefetchWorkers:
    """``_resolve_prefetch_workers`` derives the feature-streaming prefetch count from --processes.

    round(1.5 x processes), floored at FEATURE_PREFETCH_FLOOR, so the read-ahead buffer tracks
    --processes (the RAM dial) and never drops below a one-ahead buffer. Pure function, no live API.
    """

    @pytest.mark.parametrize(
        "processes,expected",
        [
            (1, 2),  # round(1.5) -> 2, equals the floor — minimum valid input
            (2, 3),  # round(3.0) -> 3
            (4, 6),  # nmaipy CLI default 4 -> 6
            (8, 12),
            (12, 18),  # current operational default on a 23-CPU box -> 18
            (16, 24),  # AI Exporter's 16 processes -> 24
        ],
    )
    def test_derivation(self, processes, expected):
        assert _resolve_prefetch_workers(processes) == expected

    def test_floor_holds_at_minimum_input(self):
        # processes=1 is the smallest valid input (BaseExporter rejects < 1).
        # The floor guarantees ThreadPoolExecutor(max_workers=...) is >= 1 even there.
        assert _resolve_prefetch_workers(1) >= FEATURE_PREFETCH_FLOOR

    def test_scales_with_processes(self):
        # Dropping --processes shrinks the buffer; raising it reads further ahead.
        assert _resolve_prefetch_workers(4) < _resolve_prefetch_workers(8) < _resolve_prefetch_workers(16)


class TestResolveMergePrefetchWorkers:
    """``_resolve_merge_prefetch_workers`` sizes the per-class merge read-ahead.

    Auto (merge_read_workers<=0) = max(features-stream prefetch, scan_workers), since per-class
    chunks are small enough to read at full scan concurrency; a positive value is an operator
    override. Pure function, no live API."""

    def test_explicit_override_is_honored(self):
        # A positive override wins regardless of processes/scan_workers.
        assert _resolve_merge_prefetch_workers(3, processes=16, scan_workers=24) == 3
        assert _resolve_merge_prefetch_workers(64, processes=4, scan_workers=8) == 64

    def test_auto_uses_scan_workers_when_higher(self):
        # Default processes=4 -> prefetch 6; S3 scan_workers=24 dominates, so the
        # merge reads at 24-way (the regression this restores vs the old 24-way read-all).
        assert _resolve_merge_prefetch_workers(0, processes=4, scan_workers=24) == 24
        # Local: scan_workers=8 dominates the prefetch of 6.
        assert _resolve_merge_prefetch_workers(0, processes=4, scan_workers=8) == 8

    def test_auto_uses_processes_prefetch_when_higher(self):
        # Many processes -> processes-derived prefetch can exceed scan_workers.
        assert _resolve_merge_prefetch_workers(0, processes=32, scan_workers=24) == _resolve_prefetch_workers(32)

    def test_zero_and_negative_are_treated_as_auto(self):
        assert _resolve_merge_prefetch_workers(0, processes=4, scan_workers=8) == 8
        assert _resolve_merge_prefetch_workers(-1, processes=4, scan_workers=8) == 8


class TestStagedFileNeedsUpload:
    """``_staged_file_needs_upload`` skips staged files already in S3 at the same size.

    Guards the per-class staging sweep against re-uploading features.parquet (streamed +
    uploaded earlier, shares the staging dir) while still uploading new/changed files.
    Pure function, no live API.
    """

    def test_absent_remote_uploads(self):
        # No remote object yet -> must upload.
        assert _staged_file_needs_upload(1000, None) is True

    def test_same_size_skips(self):
        # Identical size already in S3 -> skip (the features.parquet case).
        assert _staged_file_needs_upload(636658449520, 636658449520) is False

    def test_different_size_uploads(self):
        # Size differs -> re-upload (changed / incomplete remote object).
        assert _staged_file_needs_upload(1000, 999) is True
        assert _staged_file_needs_upload(1000, 0) is True


class TestAddMissingColumns:
    """``_add_missing_columns`` backfills absent columns without the per-column fragmentation warning.

    Regression guard for the no-coverage chunk path: a chunk that returns no survey resources has an
    empty ``metadata_df``, so all metadata columns are absent from the (very wide) ``final_df`` and must
    be added. Doing that one column at a time floods the logs with pandas PerformanceWarnings.
    """

    META_COLUMNS = [
        "system_version",
        "link",
        "survey_date",
        "survey_id",
        "survey_resource_id",
        "perspective",
        "postcat",
        "mesh_date",
    ]

    @staticmethod
    def _fragmented_frame(n_cols: int = 300, n_rows: int = 5) -> pd.DataFrame:
        """Build a deliberately block-fragmented frame, mirroring the wide per-chunk rollup."""
        df = pd.DataFrame({AOI_ID_COLUMN_NAME: range(n_rows)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the fragmenting inserts warn too; not what we're testing
            for i in range(n_cols):
                df[f"col_{i}"] = i
        return df

    def test_adds_all_missing_columns_as_null(self):
        df = self._fragmented_frame()
        result = _add_missing_columns(df, self.META_COLUMNS)
        for col in self.META_COLUMNS:
            assert col in result.columns, f"{col} should have been added"
            assert result[col].isna().all(), f"{col} should be all-null"
        assert len(result) == len(df)

    def test_no_fragmentation_warning_on_wide_frame(self):
        # Sanity: the naive per-column approach DOES warn on a wide frame, so the regression
        # assertion below is meaningful (it fails if anyone reverts to that loop).
        naive = self._fragmented_frame()
        with pytest.warns(pd.errors.PerformanceWarning):
            for col in self.META_COLUMNS:
                naive[col] = None

        # The helper must add the same columns to an equally-wide frame without warning.
        df = self._fragmented_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            result = _add_missing_columns(df, self.META_COLUMNS)
        assert all(col in result.columns for col in self.META_COLUMNS)

    def test_returns_input_unchanged_when_nothing_missing(self):
        df = pd.DataFrame({AOI_ID_COLUMN_NAME: [1], "system_version": ["gen6-"]})
        result = _add_missing_columns(df, ["system_version"])
        assert result is df  # no-op, no needless copy

    def test_non_unique_index_does_not_explode(self):
        # final_df is indexed by aoi_id; a left-merge against metadata carrying duplicate
        # aoi_ids makes that index non-unique. Backfill must not cartesian-explode the rows,
        # and must preserve the original index (duplicate labels and name included).
        df = pd.DataFrame(
            {"area": [1, 2, 3, 4]},
            index=pd.Index(["a", "a", "b", "c"], name=AOI_ID_COLUMN_NAME),
        )
        result = _add_missing_columns(df, self.META_COLUMNS)
        assert len(result) == 4
        assert list(result.index) == ["a", "a", "b", "c"]
        assert result.index.name == AOI_ID_COLUMN_NAME
        assert result["area"].tolist() == [1, 2, 3, 4]
        for col in self.META_COLUMNS:
            assert result[col].isna().all()

    def test_added_columns_are_object_none_not_nan(self):
        # Object None (not float NaN) keeps a no-coverage chunk schema-compatible with chunks
        # that carry string metadata, so cross-chunk parquet unification doesn't clash.
        result = _add_missing_columns(pd.DataFrame({"area": [1, 2]}), ["survey_id"])
        assert result["survey_id"].dtype == object
        assert result["survey_id"].iloc[0] is None

    def test_empty_frame_gets_columns_with_zero_rows(self):
        # A fully-failed chunk (all Feature API requests errored) can reach the backfill with
        # zero rows; the metadata columns must still be added so the chunk's schema is consistent.
        df = pd.DataFrame({"area": pd.Series([], dtype="float64")})
        result = _add_missing_columns(df, ["survey_id", "link"])
        assert len(result) == 0
        assert "survey_id" in result.columns
        assert "link" in result.columns

    def test_partial_backfill_preserves_present_values(self):
        df = self._fragmented_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # setup-only insert on the fragmented frame
            df["system_version"] = "gen6-"
        result = _add_missing_columns(df, self.META_COLUMNS)
        assert (result["system_version"] == "gen6-").all()
        for col in (c for c in self.META_COLUMNS if c != "system_version"):
            assert result[col].isna().all()


class TestExporter:
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
            assert len(consolidated_features) == len(
                manual_concat
            ), f"Feature count mismatch: consolidated={len(consolidated_features)}, manual_concat={len(manual_concat)}"

            # Verify CRS preservation
            if hasattr(manual_concat, "crs") and manual_concat.crs:
                assert consolidated_features.crs == manual_concat.crs, "CRS not preserved"

            # Verify essential columns exist
            assert "geometry" in consolidated_features.columns, "Missing geometry column"
            assert "aoi_id" in consolidated_features.columns, "Missing aoi_id column"

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
            assert not hasattr(exporter, "export"), "AOIExporter should NOT have export() method"

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
            warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
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

        results = _read_parquet_chunks_parallel([str(empty_path), str(nonempty_path)], max_workers=2)
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


class TestWriteErrorsParquet:
    """Tests for _write_errors_parquet.

    Regression for a crash seen in production: gridding attaches shapely Polygons to the
    errors dataframe, and the early-return "all AOIs errored" path in process_chunk wrote
    that dataframe via plain pandas to_parquet, which fails with ArrowInvalid on shapely types.
    """

    def test_plain_errors_roundtrip(self, tmp_path):
        errors_df = pd.DataFrame(
            {
                AOI_ID_COLUMN_NAME: ["a", "b"],
                "status_code": [404, 500],
                "message": ["not found", "internal"],
            }
        ).set_index(AOI_ID_COLUMN_NAME)
        outfile = tmp_path / "feature_api_errors.parquet"

        _write_errors_parquet(errors_df, str(outfile))

        assert outfile.exists()
        read_back = pd.read_parquet(outfile)
        assert len(read_back) == 2
        assert set(read_back["status_code"]) == {404, 500}

    def test_errors_with_geometry_roundtrip(self, tmp_path):
        from shapely.geometry import Polygon

        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        errors_df = pd.DataFrame(
            {
                AOI_ID_COLUMN_NAME: ["a", "b"],
                "status_code": [500, 504],
                "message": ["grid cell failed", "grid cell timeout"],
                "geometry": [poly1, poly2],
            }
        ).set_index(AOI_ID_COLUMN_NAME)
        outfile = tmp_path / "feature_api_errors.parquet"

        _write_errors_parquet(errors_df, str(outfile))

        assert outfile.exists()
        read_back = gpd.read_parquet(outfile)
        assert isinstance(read_back, gpd.GeoDataFrame)
        assert len(read_back) == 2
        assert read_back.crs == API_CRS
        assert read_back.geometry.iloc[0].equals(poly1)
        assert read_back.geometry.iloc[1].equals(poly2)

    def test_empty_errors_with_geometry_column(self, tmp_path):
        errors_df = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: pd.Series([], dtype="object"),
                "status_code": pd.Series([], dtype="int64"),
                "message": pd.Series([], dtype="object"),
            },
            geometry=gpd.GeoSeries([], crs=API_CRS),
        ).set_index(AOI_ID_COLUMN_NAME)
        outfile = tmp_path / "feature_api_errors.parquet"

        _write_errors_parquet(errors_df, str(outfile))

        assert outfile.exists()


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
            assert class_id in unique_classes, f"{name} ({class_id}) should be present in all-packs NJ data"

        assert whitelisted == PER_CLASS_FILE_CLASS_IDS, (
            f"All 6 whitelisted classes should be present. " f"Missing: {PER_CLASS_FILE_CLASS_IDS - whitelisted}"
        )

        # Non-whitelisted classes should be numerous (vegetation, surfaces, etc.)
        assert len(skipped) >= 20, f"Expected 20+ non-whitelisted classes, got {len(skipped)}"

        # Every class in data must be in exactly one partition
        assert whitelisted | skipped == unique_classes

        # Verify every row is accounted for — no data lost by the partition
        total_rows = len(features_2_all_packs_df)
        whitelisted_rows = features_2_all_packs_df[features_2_all_packs_df["class_id"].isin(PER_CLASS_FILE_CLASS_IDS)]
        skipped_rows = features_2_all_packs_df[~features_2_all_packs_df["class_id"].isin(PER_CLASS_FILE_CLASS_IDS)]
        assert len(whitelisted_rows) + len(skipped_rows) == total_rows, "Whitelist partition must account for all rows"
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

            exporter = AOIExporter(output_dir=tmpdir, country="au", packs=["building"], save_features=True)
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

            exporter = AOIExporter(output_dir=tmpdir, country="au", packs=["building"], save_features=True)
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
                {
                    "text_col": ["hello"],
                    "bin_col": [b"\x00\x01"],
                    "int_col": [1],
                    "class_id": ["c1"],
                },
                geometry=[Point(0, 0)],
                crs="EPSG:4326",
            )
            gdf.to_parquet(chunk_dir / "features_1.parquet")

            exporter = AOIExporter(output_dir=tmpdir, country="au", packs=["building"], save_features=True)
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

            exporter = AOIExporter(output_dir=tmpdir, country="au", packs=["building"], save_features=True)
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
        t2 = pa.table(
            {
                "spotlight_index": pa.array([75, None], type=pa.float64()),
                "name": ["b", "c"],
            }
        )
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
        return {cid: _description_to_cname(FEATURE_CLASS_DESCRIPTIONS[cid]) for cid in PER_CLASS_FILE_CLASS_IDS}

    def test_each_class_matches_own_chunks(self, per_class_cnames):
        """Every class's regex matches its own tabular and geo chunk filenames."""
        for cid, cname in per_class_cnames.items():
            tabular_re, geo_re = _per_class_chunk_regexes(cname)
            tabular_file = f"{cname}_0.parquet"
            geo_file = f"{cname}_features_0.parquet"
            assert tabular_re.match(tabular_file), f"{cname} tabular regex should match {tabular_file}"
            assert geo_re.match(geo_file), f"{cname} geo regex should match {geo_file}"

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
                assert not tabular_re.match(other_tabular), f"{cname!r} tabular regex must not match {other_tabular!r}"
                assert not geo_re.match(other_geo), f"{cname!r} geo regex must not match {other_geo!r}"

    def test_tabular_regex_excludes_geo_files(self, per_class_cnames):
        """Tabular regex must not match geo chunk files for the same class."""
        for cid, cname in per_class_cnames.items():
            tabular_re, _ = _per_class_chunk_regexes(cname)
            geo_file = f"{cname}_features_0.parquet"
            assert not tabular_re.match(geo_file), f"{cname!r} tabular regex must not match geo file {geo_file!r}"


def _geo_metadata():
    """Build the same GeoParquet 1.0.0 metadata dict the exporter writes."""
    return {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": [],
                "crs": pyproj.CRS(API_CRS).to_json_dict(),
                "edges": "planar",
                "orientation": "counterclockwise",
            }
        },
    }


def _write_chunks(tmp_path, tables, prefix="roof"):
    """Write Arrow tables to per-class chunk parquet files; return their paths."""
    paths = []
    for i, table in enumerate(tables):
        p = str(tmp_path / f"{prefix}_{i}.parquet")
        pq.write_table(table, p)
        paths.append(p)
    return paths


class TestStreamMergeChunksToParquet:
    """The streaming per-class merge must reproduce the read-all-then-concat
    schema/row semantics of _unify_and_concat_tables() while bounding peak
    memory to a small read-ahead window rather than the whole class."""

    def _mismatched_chunks(self):
        """Three chunks with deliberately mismatched schemas:
        int64 vs float64, a missing column, and string vs large_string."""
        c0 = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "val": pa.array([1.0, 2.0], type=pa.float64()),
                "name": pa.array(["a", "b"], type=pa.string()),
            }
        )
        c1 = pa.table(
            {
                "id": pa.array([3], type=pa.int64()),
                "val": pa.array([3], type=pa.int64()),  # int64 vs float64
                "name": pa.array(["c"], type=pa.large_string()),  # large vs string
            }
        )
        c2 = pa.table(
            {
                "id": pa.array([4], type=pa.int64()),
                "name": pa.array(["d"], type=pa.string()),  # missing "val"
            }
        )
        return [c0, c1, c2]

    def test_schema_matches_oracle_and_row_count(self, tmp_path):
        chunks = self._mismatched_chunks()
        paths = _write_chunks(tmp_path, chunks)
        out = str(tmp_path / "merged.parquet")

        rows = _stream_merge_chunks_to_parquet(paths, out, scan_workers=2, prefetch_workers=1)

        # Oracle: the read-all-then-concat path this replaces, fed the same files
        # in the same order so column order is comparable.
        oracle = _unify_and_concat_tables([pq.read_table(p) for p in paths])
        result = pq.read_table(out)

        expected_rows = sum(len(c) for c in chunks)
        assert rows == expected_rows
        assert result.num_rows == expected_rows
        # Schema (names + types) must match the oracle; metadata differs trivially.
        assert result.schema.equals(oracle.schema, check_metadata=False)
        # Promotions preserved: int64+float64 -> float64, string -> large_string.
        assert result.schema.field("val").type == pa.float64()
        assert result.schema.field("name").type == pa.large_string()

    def test_progress_desc_renders_without_breaking_output(self, tmp_path):
        """With progress_desc set, a tqdm bar is drawn but output is unchanged."""
        chunks = self._mismatched_chunks()
        paths = _write_chunks(tmp_path, chunks)
        out = str(tmp_path / "merged.parquet")
        rows = _stream_merge_chunks_to_parquet(
            paths, out, scan_workers=2, prefetch_workers=2, progress_desc="Building geo"
        )
        assert rows == sum(len(c) for c in chunks)
        assert pq.read_table(out).num_rows == rows

    def test_missing_column_null_filled(self, tmp_path):
        chunks = self._mismatched_chunks()
        paths = _write_chunks(tmp_path, chunks)
        out = str(tmp_path / "merged.parquet")
        _stream_merge_chunks_to_parquet(paths, out, scan_workers=2, prefetch_workers=1)

        result = pq.read_table(out).sort_by("id")
        # id==4 came from the chunk lacking "val"; it must be null, not dropped.
        by_id = dict(zip(result.column("id").to_pylist(), result.column("val").to_pylist()))
        assert by_id[4] is None
        assert by_id[1] == 1.0

    def test_geo_metadata_present_and_wellformed(self, tmp_path):
        geom = pa.array([b"\x00wkb-a", b"\x00wkb-b"], type=pa.binary())
        c0 = pa.table({"id": pa.array([1, 2], type=pa.int64()), "geometry": geom})
        c1 = pa.table({"id": pa.array([3], type=pa.int64()), "geometry": pa.array([b"\x00wkb-c"], type=pa.binary())})
        paths = _write_chunks(tmp_path, [c0, c1], prefix="roof_features")
        out = str(tmp_path / "roof_features.parquet")

        rows = _stream_merge_chunks_to_parquet(
            paths, out, scan_workers=2, prefetch_workers=1, geo_metadata=_geo_metadata()
        )
        assert rows == 3

        meta = pq.read_schema(out).metadata
        assert meta is not None and b"geo" in meta
        geo = json.loads(meta[b"geo"])
        assert geo["version"] == "1.0.0"
        assert geo["primary_column"] == "geometry"
        assert geo["columns"]["geometry"]["encoding"] == "WKB"
        assert geo["columns"]["geometry"]["crs"]  # CRS projjson present

    def test_empty_chunks_produce_schema_only_output(self, tmp_path):
        """Chunks with zero rows still yield a valid output file with the schema."""
        empty = pa.table({"id": pa.array([], type=pa.int64()), "name": pa.array([], type=pa.string())})
        paths = _write_chunks(tmp_path, [empty, empty])
        out = str(tmp_path / "merged.parquet")
        rows = _stream_merge_chunks_to_parquet(paths, out, scan_workers=2, prefetch_workers=1)
        assert rows == 0
        result = pq.read_table(out)
        assert result.num_rows == 0
        assert set(result.column_names) == {"id", "name"}

    def test_empty_input_list_returns_zero(self, tmp_path):
        out = str(tmp_path / "merged.parquet")
        assert _stream_merge_chunks_to_parquet([], out, scan_workers=2, prefetch_workers=1) == 0
        assert not os.path.exists(out)

    def test_all_unreadable_chunks_returns_zero(self, tmp_path, caplog):
        """If no chunk schema can be read, the merge logs and returns 0 rather
        than crashing — matching the resilience of the previous path."""
        bad = str(tmp_path / "roof_0.parquet")
        Path(bad).write_text("not a parquet file")
        out = str(tmp_path / "merged.parquet")
        with caplog.at_level(logging.ERROR):
            rows = _stream_merge_chunks_to_parquet([bad], out, scan_workers=2, prefetch_workers=1)
        assert rows == 0

    def test_resident_tables_bounded_independent_of_chunk_count(self, tmp_path, monkeypatch):
        """Peak resident chunk tables stays a small constant regardless of how
        many chunks are merged — proving memory is decoupled from total class
        size. Counts live pyarrow Tables via weakref finalizers (deterministic
        under CPython refcounting), so it needs no RSS sampling."""
        n_chunks = 24
        prefetch_workers = 1
        chunks = [
            pa.table(
                {
                    "id": pa.array(list(range(i * 100, i * 100 + 100)), type=pa.int64()),
                    "name": pa.array([f"r{i}-{j}" for j in range(100)], type=pa.string()),
                }
            )
            for i in range(n_chunks)
        ]
        paths = _write_chunks(tmp_path, chunks)
        out = str(tmp_path / "merged.parquet")

        original_read_table = pq.read_table
        state = {"live": 0, "peak": 0}

        def tracking_read_table(path, *args, **kwargs):
            table = original_read_table(path, *args, **kwargs)
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])

            def _on_collect():
                state["live"] -= 1

            weakref.finalize(table, _on_collect)
            return table

        monkeypatch.setattr("nmaipy.exporter.pq.read_table", tracking_read_table)
        rows = _stream_merge_chunks_to_parquet(paths, out, scan_workers=4, prefetch_workers=prefetch_workers)

        assert rows == n_chunks * 100
        # Resident tables never accumulate: a small constant, not O(n_chunks).
        assert state["peak"] <= prefetch_workers + 3, state["peak"]
        assert state["peak"] < n_chunks
        assert pq.read_table(out).num_rows == n_chunks * 100

    def test_resident_tables_bounded_at_production_prefetch(self, tmp_path, monkeypatch):
        """At a production-like prefetch_workers (8) with a slow consume step,
        resident tables stay capped at ~prefetch_workers, not the chunk count.
        The pw=1 test above can't catch a 'submit all reads up front' OOM mutant
        (which peaks at 1 there); this regime does (it would peak at n_chunks)."""
        n_chunks = 24
        prefetch_workers = 8
        chunks = [pa.table({"id": pa.array([i], type=pa.int64())}) for i in range(n_chunks)]
        paths = _write_chunks(tmp_path, chunks)
        out = str(tmp_path / "merged.parquet")

        original_read_table = pq.read_table
        state = {"live": 0, "peak": 0}

        def tracking_read_table(path, *args, **kwargs):
            table = original_read_table(path, *args, **kwargs)
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
            weakref.finalize(table, lambda: state.__setitem__("live", state["live"] - 1))
            return table

        original_reconcile = exporter_mod._reconcile_table_schema

        def slow_reconcile(table, target):
            time.sleep(0.02)  # let the prefetch window fill while the writer is busy
            return original_reconcile(table, target)

        monkeypatch.setattr("nmaipy.exporter.pq.read_table", tracking_read_table)
        monkeypatch.setattr("nmaipy.exporter._reconcile_table_schema", slow_reconcile)
        rows = _stream_merge_chunks_to_parquet(paths, out, scan_workers=8, prefetch_workers=prefetch_workers)

        assert rows == n_chunks
        assert 2 <= state["peak"] <= prefetch_workers + 1, state["peak"]
        assert state["peak"] < n_chunks

    def test_null_typed_column_promotion(self, tmp_path):
        """A column that is null-typed in one chunk and float64 in another unifies
        to float64; a column null across ALL chunks promotes to large_string. The
        streaming path reaches these via a footer-only Pass 1 that the in-memory
        oracle test never exercises."""
        c0 = pa.table(
            {
                "id": pa.array([1], type=pa.int64()),
                "conf": pa.array([None], type=pa.null()),  # null-typed here
                "note": pa.array([None], type=pa.null()),  # null in every chunk
            }
        )
        c1 = pa.table(
            {
                "id": pa.array([2], type=pa.int64()),
                "conf": pa.array([0.5], type=pa.float64()),  # real type here
                "note": pa.array([None], type=pa.null()),
            }
        )
        paths = _write_chunks(tmp_path, [c0, c1])
        out = str(tmp_path / "merged.parquet")
        _stream_merge_chunks_to_parquet(paths, out, scan_workers=2, prefetch_workers=1)

        result = pq.read_table(out)
        assert result.schema.field("conf").type == pa.float64()
        assert result.schema.field("note").type == pa.large_string()
        assert sorted(v for v in result.column("conf").to_pylist() if v is not None) == [0.5]

    def test_writer_failure_leaves_no_output_file(self, tmp_path, monkeypatch):
        """A write failure mid-stream must not leave a readable, truncated file at
        output_path (the regression: the writer's close() in finally finalises a
        valid footer over the rows written so far). The error must propagate, no
        .tmp must leak, and the prefetch pool must be shut down."""
        chunks = [pa.table({"id": pa.array([i, i + 100], type=pa.int64())}) for i in range(6)]
        paths = _write_chunks(tmp_path, chunks)
        out = str(tmp_path / "merged.parquet")

        original_writer_cls = pq.ParquetWriter

        class FailingWriter:
            def __init__(self, *args, **kwargs):
                self._w = original_writer_cls(*args, **kwargs)
                self._calls = 0

            def write_table(self, table, *args, **kwargs):
                self._calls += 1
                if self._calls == 3:
                    raise OSError("simulated disk full")
                return self._w.write_table(table, *args, **kwargs)

            def close(self):
                return self._w.close()

        threads_before = threading.active_count()
        monkeypatch.setattr("nmaipy.exporter.pq.ParquetWriter", FailingWriter)
        with pytest.raises(OSError):
            _stream_merge_chunks_to_parquet(paths, out, scan_workers=2, prefetch_workers=2)

        assert not os.path.exists(out), "truncated parquet must not be left at output_path"
        assert not os.path.exists(out + ".tmp"), "temp file must be cleaned on failure"
        assert threading.active_count() <= threads_before, "prefetch pool leaked threads"

    def test_all_data_reads_fail_writes_no_file(self, tmp_path, monkeypatch):
        """Footers read but every data page is unreadable: write nothing (old
        parity) rather than a spurious counted 0-row file."""
        chunks = [pa.table({"id": pa.array([i], type=pa.int64())}) for i in range(3)]
        paths = _write_chunks(tmp_path, chunks)
        out = str(tmp_path / "merged.parquet")

        def failing_read_table(path, *args, **kwargs):
            raise OSError("corrupt data page")

        monkeypatch.setattr("nmaipy.exporter.pq.read_table", failing_read_table)
        rows = _stream_merge_chunks_to_parquet(paths, out, scan_workers=2, prefetch_workers=1)
        assert rows == 0
        assert not os.path.exists(out)

    def test_one_corrupt_chunk_among_good_is_skipped(self, tmp_path):
        """A single unreadable chunk (bad footer) is skipped; the others merge."""
        good = [pa.table({"id": pa.array([1, 2], type=pa.int64())}), pa.table({"id": pa.array([3], type=pa.int64())})]
        paths = _write_chunks(tmp_path, good)
        bad = str(tmp_path / "roof_2.parquet")
        Path(bad).write_text("not a parquet file")
        out = str(tmp_path / "merged.parquet")

        rows = _stream_merge_chunks_to_parquet(paths + [bad], out, scan_workers=2, prefetch_workers=2)
        assert rows == 3
        assert sorted(pq.read_table(out).column("id").to_pylist()) == [1, 2, 3]

    def test_footer_ok_data_corrupt_chunk_skipped(self, tmp_path, monkeypatch):
        """A chunk whose footer reads but whose data read raises is dropped; the
        good chunks still merge, and a column unique to the dropped chunk is
        present (from its footer) as all-null."""
        good = pa.table({"id": pa.array([1, 2], type=pa.int64())})
        damaged = pa.table({"id": pa.array([3], type=pa.int64()), "extra": pa.array(["x"], type=pa.string())})
        paths = _write_chunks(tmp_path, [good, damaged])
        out = str(tmp_path / "merged.parquet")

        original_read_table = pq.read_table

        def selective_read(path, *args, **kwargs):
            if path == paths[1]:
                raise OSError("corrupt data page")
            return original_read_table(path, *args, **kwargs)

        monkeypatch.setattr("nmaipy.exporter.pq.read_table", selective_read)
        rows = _stream_merge_chunks_to_parquet(paths, out, scan_workers=2, prefetch_workers=1)

        result = pq.read_table(out)
        assert rows == 2
        assert sorted(result.column("id").to_pylist()) == [1, 2]
        # 'extra' came only from the damaged chunk's footer -> column exists, all null.
        assert "extra" in result.column_names
        assert result.column("extra").to_pylist() == [None, None]

    def test_all_unreadable_removes_stale_output(self, tmp_path):
        """An all-footers-unreadable merge must clear a stale prior-run file so
        the caller's existence check can't count it as this run's output."""
        out = str(tmp_path / "merged.parquet")
        pq.write_table(pa.table({"stale": pa.array([1], type=pa.int64())}), out)
        bad = str(tmp_path / "roof_0.parquet")
        Path(bad).write_text("garbage")

        rows = _stream_merge_chunks_to_parquet([bad], out, scan_workers=2, prefetch_workers=1)
        assert rows == 0
        assert not os.path.exists(out)


class TestParquetToCsvStreaming:
    """The tabular->CSV conversion streams batches instead of loading the whole
    file, but must still produce a byte-faithful CSV."""

    def test_roundtrip_matches_pandas(self, tmp_path):
        table = pa.table({"id": [1, 2, 3, 4, 5, 6], "name": ["a", "b", "c", "d", "e", "f"]})
        parquet_path = str(tmp_path / "t.parquet")
        # Multiple row groups to exercise the batched read path.
        pq.write_table(table, parquet_path, row_group_size=2)
        csv_path = str(tmp_path / "t.csv")

        _parquet_to_csv_streaming(parquet_path, csv_path)

        got = pd.read_csv(csv_path)
        expected = table.to_pandas()
        pd.testing.assert_frame_equal(got, expected)

    def test_empty_parquet_produces_header_only_csv(self, tmp_path):
        table = pa.table({"id": pa.array([], type=pa.int64()), "name": pa.array([], type=pa.string())})
        parquet_path = str(tmp_path / "empty.parquet")
        pq.write_table(table, parquet_path)
        csv_path = str(tmp_path / "empty.csv")

        _parquet_to_csv_streaming(parquet_path, csv_path)

        got = pd.read_csv(csv_path)
        assert list(got.columns) == ["id", "name"]
        assert len(got) == 0

    def test_multibatch_is_byte_identical_to_whole_file(self, tmp_path):
        """Over multiple iter_batches() batches (>64Ki rows), the appended CSV
        must be byte-for-byte identical to a single pd.read_parquet().to_csv() —
        catching any per-batch formatting/dtype drift (nulls, floats, strings)."""
        n = 70_000  # forces >1 batch at pyarrow's default 65536 batch size
        table = pa.table(
            {
                "id": pa.array(range(n), type=pa.int64()),
                "val": pa.array([None if i % 7 == 0 else i / 3.0 for i in range(n)], type=pa.float64()),
                "name": pa.array([None if i % 5 == 0 else f"r{i}" for i in range(n)], type=pa.string()),
            }
        )
        parquet_path = str(tmp_path / "wide.parquet")
        pq.write_table(table, parquet_path, row_group_size=10_000)
        # Guard the premise: the file really does yield more than one batch.
        with pq.ParquetFile(parquet_path) as pf:
            assert sum(1 for _ in pf.iter_batches()) > 1

        csv_path = str(tmp_path / "wide.csv")
        _parquet_to_csv_streaming(parquet_path, csv_path)

        expected = pd.read_parquet(parquet_path).to_csv(index=False).encode()
        with open(csv_path, "rb") as f:
            assert f.read() == expected


class TestMergePerClassChunks:
    """Integration test for the rewired caller: real per-class chunk files on
    local disk are merged, the tabular output is converted to CSV, the geo output
    keeps its GeoParquet metadata, and the staging dir is cleaned up."""

    def test_merge_read_workers_plumbed_to_instance(self, tmp_path):
        exporter = AOIExporter(
            aoi_file="tests/data/test_parcels_2.csv",
            output_dir=str(tmp_path),
            country="us",
            packs=["building"],
            merge_read_workers=7,
        )
        assert exporter.merge_read_workers == 7

    def test_merges_real_chunks_to_final(self, tmp_path):
        exporter = AOIExporter(
            aoi_file="tests/data/test_parcels_2.csv",
            output_dir=str(tmp_path),
            country="us",
            packs=["building"],
            processes=2,
        )
        b_cname = _description_to_cname(FEATURE_CLASS_DESCRIPTIONS[BUILDING_NEW_ID])
        r_cname = _description_to_cname(FEATURE_CLASS_DESCRIPTIONS[ROOF_ID])

        # Two tabular building chunks; the second drops a column to force a union
        # null-fill (the real-world cross-chunk schema variation this must handle).
        b0 = pd.DataFrame({"aoi_id": ["a", "b"], "area_sqm": [101.0, 202.0], "height_m": [4.5, 6.0]})
        b1 = pd.DataFrame({"aoi_id": ["c"], "area_sqm": [303.0]})  # no height_m
        b0.to_parquet(os.path.join(exporter.chunk_path, f"{b_cname}_0.parquet"), index=False)
        b1.to_parquet(os.path.join(exporter.chunk_path, f"{b_cname}_1.parquet"), index=False)

        # Two geo roof chunks written as real geoparquet (WKB geometry + CRS).
        def square(x):
            return Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)])

        r0 = gpd.GeoDataFrame({"aoi_id": ["a", "b"]}, geometry=[square(0), square(2)], crs=API_CRS)
        r1 = gpd.GeoDataFrame({"aoi_id": ["c"]}, geometry=[square(4)], crs=API_CRS)
        r0.to_parquet(os.path.join(exporter.chunk_path, f"{r_cname}_features_0.parquet"))
        r1.to_parquet(os.path.join(exporter.chunk_path, f"{r_cname}_features_1.parquet"))

        exporter._merge_per_class_chunks(tabular_file_format="csv")

        # Tabular -> CSV: union of rows and columns, height_m null for chunk 1.
        building_csv = os.path.join(exporter.final_path, f"{b_cname}.csv")
        assert os.path.exists(building_csv)
        bdf = pd.read_csv(building_csv)
        assert len(bdf) == 3
        assert set(bdf.columns) == {"aoi_id", "area_sqm", "height_m"}  # no unexpected columns
        assert bdf.set_index("aoi_id").loc["c", "height_m"] != bdf.set_index("aoi_id").loc["c", "height_m"]  # NaN
        # No stray intermediate parquet left next to the CSV.
        assert not os.path.exists(os.path.join(exporter.final_path, f"{b_cname}.parquet"))

        # Geo -> GeoParquet: opens in geopandas with CRS and the union of rows.
        roof_parquet = os.path.join(exporter.final_path, f"{r_cname}_features.parquet")
        assert os.path.exists(roof_parquet)
        rgdf = gpd.read_parquet(roof_parquet)
        assert len(rgdf) == 3
        assert rgdf.crs is not None and rgdf.crs.to_epsg() == pyproj.CRS(API_CRS).to_epsg()
        assert rgdf.geometry.is_valid.all()

        # Staging dir removed after a clean local merge.
        assert not os.path.exists(os.path.join(exporter.final_path, ".per_class_staging"))


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    sys.exit(pytest.main([current_file]))


# ---------------------------------------------------------------------------
# Fast-path: skip rebuild when README.md exists
# ---------------------------------------------------------------------------


class TestSkipRebuildWhenReadmeExists:
    """README.md is the very last file written by a successful run, so its
    presence implies every other final/ artifact was successfully produced.
    The fast-path in _run_inner returns immediately and skips chunk reading,
    consolidation, and per-class merging."""

    def test_fast_path_returns_immediately_when_readme_exists(self, tmp_path):
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "README.md").write_text("# Nearmap AI Export\n\nany content")

        exporter = AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="au",
            packs=["building"],
        )

        # FeatureApi construction is the first heavy step after the fast-path.
        # If the fast-path triggers, FeatureApi is never instantiated.
        with patch("nmaipy.exporter.FeatureApi") as mock_api:
            exporter.run()
            mock_api.assert_not_called()

    def test_full_pipeline_runs_when_readme_missing(self, tmp_path):
        """Sanity check the inverse: with no README, _run_inner proceeds past
        the fast-path. We force an early controlled failure so the test
        doesn't have to actually run an export."""
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        # No README.md present

        exporter = AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="au",
            packs=["building"],
        )

        # If we get past the fast-path, FeatureApi gets constructed. We patch
        # it to raise; observing the raise confirms we did NOT take the fast
        # path. This is a structural assertion — the actual export work isn't
        # executed.
        sentinel = RuntimeError("got past the fast-path as expected")

        def boom(*args, **kwargs):
            raise sentinel

        with patch("nmaipy.exporter.FeatureApi", side_effect=boom):
            with pytest.raises(RuntimeError, match="got past the fast-path"):
                exporter.run()

    def test_fast_path_triggered_when_config_unchanged(self, tmp_path):
        """README + matching config → fast-path triggers.

        Constructing exporter1 saves export_config.json. Constructing
        exporter2 with identical args reads that config as `_previous_config_params`
        before saving its own; comparison sees no diff, so the fast-path fires.
        """
        # Run 1: constructor saves export_config.json
        AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="au",
            packs=["building"],
        )
        # Mark "completed" by writing the README sentinel
        (tmp_path / "final" / "README.md").write_text("# done")

        # Run 2: same args
        exporter2 = AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="au",
            packs=["building"],
        )

        with patch("nmaipy.exporter.FeatureApi") as mock_api:
            exporter2.run()
            mock_api.assert_not_called()

    def test_fast_path_skipped_when_output_affecting_param_changed(self, tmp_path):
        """README present but country changed → rebuild (don't silently return prior output)."""
        # Run 1: country=au
        AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="au",
            packs=["building"],
        )
        (tmp_path / "final" / "README.md").write_text("# done")

        # Run 2: country=us — output-affecting change, must rebuild
        exporter2 = AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="us",
            packs=["building"],
        )

        with patch("nmaipy.exporter.FeatureApi", side_effect=RuntimeError("got past fast-path")):
            with pytest.raises(RuntimeError, match="got past fast-path"):
                exporter2.run()

    def test_stale_readme_removed_when_config_changed_rebuild_crashes(self, tmp_path):
        """A config-change rebuild must remove the old README sentinel up front.

        __init__ overwrites export_config.json with the current params before
        _run_inner runs, so if the rebuild dies mid-way and the old README were
        left in place, the next invocation would see an empty config diff plus
        the README and incorrectly fast-path back to the previous config's output.
        """
        AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="au",
            packs=["building"],
        )
        (tmp_path / "final" / "README.md").write_text("# done")

        exporter2 = AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="us",
            packs=["building"],
        )

        with patch("nmaipy.exporter.FeatureApi", side_effect=RuntimeError("simulated mid-rebuild crash")):
            with pytest.raises(RuntimeError, match="simulated mid-rebuild crash"):
                exporter2.run()

        assert not (tmp_path / "final" / "README.md").exists(), (
            "Stale README sentinel must be removed when a config-change rebuild starts, "
            "otherwise a crashed rebuild leaves a false 'complete' marker"
        )

    def test_fast_path_skipped_when_packs_changed(self, tmp_path):
        """Packs change → rebuild even though README is present."""
        AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="au",
            packs=["building"],
        )
        (tmp_path / "final" / "README.md").write_text("# done")

        exporter2 = AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="au",
            packs=["building", "vegetation"],  # added vegetation
        )

        with patch("nmaipy.exporter.FeatureApi", side_effect=RuntimeError("got past fast-path")):
            with pytest.raises(RuntimeError, match="got past fast-path"):
                exporter2.run()

    def test_fast_path_triggered_when_only_ignored_param_changed(self, tmp_path):
        """Performance / ergonomics knobs (processes, threads, log_level, etc.)
        do NOT invalidate the fast-path. The previous run's output is still
        valid; the user just changed how the export would have been computed,
        not what would have come out."""
        AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="au",
            packs=["building"],
            processes=4,
        )
        (tmp_path / "final" / "README.md").write_text("# done")

        exporter2 = AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="au",
            packs=["building"],
            processes=8,  # ignored — pure perf knob
        )

        with patch("nmaipy.exporter.FeatureApi") as mock_api:
            exporter2.run()
            mock_api.assert_not_called()

    def test_fast_path_triggered_when_no_previous_config(self, tmp_path):
        """Backward-compat: README present but no export_config.json → fast-path
        still triggers. Pre-existing exports from before this comparison logic
        landed shouldn't lose their fast-path on upgrade.
        """
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "README.md").write_text("# legacy")
        # No export_config.json — emulate a pre-feature export

        exporter = AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",
            output_dir=str(tmp_path),
            country="au",
            packs=["building"],
        )
        # Constructor will write its own export_config.json (overwriting nothing
        # since none was there). _previous_config_params should be None and the
        # diff should be empty → fast-path.

        with patch("nmaipy.exporter.FeatureApi") as mock_api:
            exporter.run()
            mock_api.assert_not_called()
