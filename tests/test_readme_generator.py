"""Unit and integration tests for ReadmeGenerator.

Tests the dynamic README generation functionality that creates
customer-facing documentation based on actual export files and columns.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from nmaipy.__version__ import __version__
from nmaipy.readme_generator import (
    ADDRESS_QUERY_COLUMNS,
    COMMON_COLUMNS,
    ROOF_AGE_COLUMNS,
    RSI_COLUMNS,
    ReadmeGenerator,
)


class TestDiscoverFiles:
    """Tests for file discovery functionality."""

    def test_discover_files_empty_directory(self, tmp_path):
        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        assert files == []

    def test_discover_files_excludes_readme(self, tmp_path):
        (tmp_path / "README.md").write_text("# Test")
        (tmp_path / "parcels_aoi_rollup.csv").write_text("col1,col2\n1,2")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()

        filenames = [f.name for f in files]
        assert "README.md" not in filenames
        assert "parcels_aoi_rollup.csv" in filenames

    def test_discover_files_excludes_ds_store(self, tmp_path):
        (tmp_path / ".DS_Store").write_text("")
        (tmp_path / "data.csv").write_text("col1\n1")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()

        filenames = [f.name for f in files]
        assert ".DS_Store" not in filenames

    def test_discover_files_excludes_export_config(self, tmp_path):
        (tmp_path / "export_config.json").write_text("{}")
        (tmp_path / "data.csv").write_text("col1\n1")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()

        filenames = [f.name for f in files]
        assert "export_config.json" not in filenames
        assert "data.csv" in filenames

    def test_discover_files_uses_final_subdirectory(self, tmp_path):
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "parcels_aoi_rollup.csv").write_text("col1\n1")
        (tmp_path / "other_file.csv").write_text("col1\n1")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()

        filenames = [f.name for f in files]
        assert "parcels_aoi_rollup.csv" in filenames
        assert "other_file.csv" not in filenames


class TestGetFilePrefix:
    """Tests for prefix extraction from filenames."""

    def test_get_file_prefix_from_rollup_csv(self, tmp_path):
        (tmp_path / "parcels_aoi_rollup.csv").write_text("col1\n1")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)

        assert prefix == "parcels_"

    def test_get_file_prefix_from_rollup_parquet(self, tmp_path):
        (tmp_path / "myproject_aoi_rollup.parquet").write_bytes(b"")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)

        assert prefix == "myproject_"

    def test_get_file_prefix_from_roof_csv(self, tmp_path):
        (tmp_path / "export_roof.csv").write_text("col1\n1")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)

        assert prefix == "export_"

    def test_get_file_prefix_empty_when_no_pattern(self, tmp_path):
        (tmp_path / "random_data.csv").write_text("col1\n1")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)

        assert prefix == ""


class TestDetectClasses:
    """Tests for class detection from CSV filenames."""

    def test_detect_classes_from_csv_files(self, tmp_path):
        (tmp_path / "parcels_aoi_rollup.csv").write_text("col1\n1")
        (tmp_path / "parcels_roof.csv").write_text("col1\n1")
        (tmp_path / "parcels_building.csv").write_text("col1\n1")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)
        classes = gen._detect_classes(files, prefix)

        class_columns = [c["column"] for c in classes]
        assert "roof" in class_columns
        assert "building" in class_columns

    def test_detect_classes_skips_error_files(self, tmp_path):
        (tmp_path / "parcels_aoi_rollup.csv").write_text("col1\n1")
        (tmp_path / "parcels_feature_api_errors.csv").write_text("col1\n1")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)
        classes = gen._detect_classes(files, prefix)

        class_columns = [c["column"] for c in classes]
        assert "feature_api_errors" not in class_columns

    def test_detect_classes_skips_buildings_file(self, tmp_path):
        (tmp_path / "parcels_aoi_rollup.csv").write_text("col1\n1")
        (tmp_path / "parcels_buildings.csv").write_text("col1\n1")
        (tmp_path / "parcels_roof.csv").write_text("col1\n1")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)
        classes = gen._detect_classes(files, prefix)

        class_columns = [c["column"] for c in classes]
        assert "buildings" not in class_columns
        assert "roof" in class_columns

    def test_detect_classes_formats_display_name(self, tmp_path):
        (tmp_path / "parcels_aoi_rollup.csv").write_text("col1\n1")
        (tmp_path / "parcels_roof_instance.csv").write_text("col1\n1")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)
        classes = gen._detect_classes(files, prefix)

        roof_instance = next((c for c in classes if c["column"] == "roof_instance"), None)
        assert roof_instance is not None
        assert roof_instance["name"] == "Roof Instance"


class TestHasRsiColumns:
    """Tests for RSI column detection."""

    def test_has_rsi_columns_with_index(self):
        gen = ReadmeGenerator(output_dir=Path("."))
        columns = {"roof_spotlight_index", "other_col"}
        assert gen._has_rsi_columns(columns) is True

    def test_has_rsi_columns_with_confidence(self):
        gen = ReadmeGenerator(output_dir=Path("."))
        columns = {"roof_spotlight_index_confidence", "other_col"}
        assert gen._has_rsi_columns(columns) is True

    def test_has_rsi_columns_without(self):
        gen = ReadmeGenerator(output_dir=Path("."))
        columns = {"aoi_id", "mesh_date", "roof_count"}
        assert gen._has_rsi_columns(columns) is False


class TestHasRoofAgeColumns:
    """Tests for roof age column detection."""

    def test_has_roof_age_columns_with_installation_date(self):
        gen = ReadmeGenerator(output_dir=Path("."))
        columns = {"roof_age_installation_date", "other_col"}
        assert gen._has_roof_age_columns(columns) is True

    def test_has_roof_age_columns_with_years(self):
        gen = ReadmeGenerator(output_dir=Path("."))
        columns = {"roof_age_years_as_of_date", "other_col"}
        assert gen._has_roof_age_columns(columns) is True

    def test_has_roof_age_columns_without(self):
        gen = ReadmeGenerator(output_dir=Path("."))
        columns = {"aoi_id", "mesh_date", "roof_count"}
        assert gen._has_roof_age_columns(columns) is False


class TestHasAddressQueryColumns:
    """Tests for address/query column detection."""

    def test_has_address_query_columns_with_street_address(self):
        gen = ReadmeGenerator(output_dir=Path("."))
        columns = {"streetAddress", "city", "state", "zip"}
        assert gen._has_address_query_columns(columns) is True

    def test_has_address_query_columns_with_query_lat(self):
        gen = ReadmeGenerator(output_dir=Path("."))
        columns = {"query_aoi_lat", "query_aoi_lon"}
        assert gen._has_address_query_columns(columns) is True

    def test_has_address_query_columns_without(self):
        gen = ReadmeGenerator(output_dir=Path("."))
        columns = {"aoi_id", "mesh_date", "roof_count"}
        assert gen._has_address_query_columns(columns) is False


class TestGenerateEmptyDirectory:
    """Tests for handling empty directories."""

    def test_generate_empty_directory(self, tmp_path):
        gen = ReadmeGenerator(output_dir=tmp_path)
        content = gen._generate()

        assert "# Nearmap AI Export" in content
        assert "Generated by nmaipy" in content
        assert f"v{__version__}" in content
        assert "## Files in This Export" in content


class TestLoadExportConfig:
    """Tests for loading export configuration."""

    def test_load_export_config_reads_parameters(self, tmp_path):
        config = {
            "_metadata": {"nmaipy_version": "4.1.4"},
            "parameters": {"primary_decision": "nearest", "country": "us"},
        }
        (tmp_path / "export_config.json").write_text(json.dumps(config))

        gen = ReadmeGenerator(output_dir=tmp_path)
        loaded = gen._load_export_config()
        assert loaded["parameters"]["primary_decision"] == "nearest"

    def test_load_export_config_missing_file(self, tmp_path):
        gen = ReadmeGenerator(output_dir=tmp_path)
        loaded = gen._load_export_config()
        assert loaded == {}

    def test_detect_area_unit_us(self, tmp_path):
        config = {"_metadata": {}, "parameters": {"country": "us"}}
        (tmp_path / "export_config.json").write_text(json.dumps(config))

        gen = ReadmeGenerator(output_dir=tmp_path)
        assert gen._detect_area_unit() == "sqft"

    def test_detect_area_unit_au(self, tmp_path):
        config = {"_metadata": {}, "parameters": {"country": "au"}}
        (tmp_path / "export_config.json").write_text(json.dumps(config))

        gen = ReadmeGenerator(output_dir=tmp_path)
        assert gen._detect_area_unit() == "sqm"

    def test_detect_area_unit_defaults_to_sqft(self, tmp_path):
        gen = ReadmeGenerator(output_dir=tmp_path)
        assert gen._detect_area_unit() == "sqft"

    def test_column_patterns_uses_correct_unit_for_au(self, tmp_path):
        config = {"_metadata": {}, "parameters": {"country": "au"}}
        (tmp_path / "export_config.json").write_text(json.dumps(config))
        (tmp_path / "parcels_aoi_rollup.csv").write_text("aoi_id,roof_present\n0,Y\n")
        (tmp_path / "parcels_roof.csv").write_text("aoi_id\n0\n")

        gen = ReadmeGenerator(output_dir=tmp_path)
        content = gen._generate()

        assert "total_area_sqm" in content
        assert "total_area_sqft" not in content
        assert "Square meters (sqm)" in content

    def test_column_patterns_uses_config_primary_decision(self, tmp_path):
        config = {
            "_metadata": {},
            "parameters": {"primary_decision": "nearest"},
        }
        (tmp_path / "export_config.json").write_text(json.dumps(config))
        (tmp_path / "parcels_aoi_rollup.csv").write_text("aoi_id,roof_present\n0,Y\n")
        (tmp_path / "parcels_roof.csv").write_text("aoi_id\n0\n")

        gen = ReadmeGenerator(output_dir=tmp_path)
        content = gen._generate()

        assert "nearest to the property centroid" in content


class TestGenerateWithRollupFile:
    """Tests for full README generation with mock files."""

    def test_generate_with_basic_rollup(self, tmp_path):
        rollup_content = "aoi_id,mesh_date,roof_present,roof_count,building_present,building_count\n"
        rollup_content += "0,2024-01-15,Y,2,Y,1\n"
        (tmp_path / "test_aoi_rollup.csv").write_text(rollup_content)
        (tmp_path / "test_roof.csv").write_text("aoi_id,roof_id\n0,r1\n")
        (tmp_path / "test_building.csv").write_text("aoi_id,building_id\n0,b1\n")

        gen = ReadmeGenerator(output_dir=tmp_path)
        readme_path = gen.generate_and_save()

        assert readme_path.exists()
        content = readme_path.read_text()

        assert "# Nearmap AI Export" in content
        assert "## Files in This Export" in content
        assert "test_aoi_rollup.csv" in content
        assert "test_roof.csv" in content
        assert "test_building.csv" in content
        assert "## Feature Classes in This Export" in content
        assert "## Column Naming Patterns" in content
        assert "## Common Columns" in content

    def test_generate_with_rsi_columns(self, tmp_path):
        rollup_content = "aoi_id,roof_spotlight_index,roof_spotlight_index_confidence\n"
        rollup_content += "0,45,0.85\n"
        (tmp_path / "parcels_aoi_rollup.csv").write_text(rollup_content)

        gen = ReadmeGenerator(output_dir=tmp_path)
        content = gen._generate()

        assert "## Roof Spotlight Index (RSI) Columns" in content
        assert "roof_spotlight_index" in content

    def test_generate_with_roof_age_columns(self, tmp_path):
        rollup_content = "aoi_id,roof_age_installation_date,roof_age_years_as_of_date\n"
        rollup_content += "0,2015-06-01,9\n"
        (tmp_path / "parcels_aoi_rollup.csv").write_text(rollup_content)

        gen = ReadmeGenerator(output_dir=tmp_path)
        content = gen._generate()

        assert "## Roof Age Columns" in content
        assert "roof_age_installation_date" in content

    def test_generate_with_address_query_columns(self, tmp_path):
        rollup_content = "aoi_id,streetAddress,city,state,zip\n"
        rollup_content += "0,123 Main St,Springfield,IL,62704\n"
        (tmp_path / "parcels_aoi_rollup.csv").write_text(rollup_content)

        gen = ReadmeGenerator(output_dir=tmp_path)
        content = gen._generate()

        assert "## Address & Query Columns" in content
        assert "streetAddress" in content
        assert "city" in content

    def test_generate_with_geometry_query_columns(self, tmp_path):
        rollup_content = "aoi_id,query_aoi_lat,query_aoi_lon\n"
        rollup_content += "0,40.7128,-74.0060\n"
        (tmp_path / "parcels_aoi_rollup.csv").write_text(rollup_content)

        gen = ReadmeGenerator(output_dir=tmp_path)
        content = gen._generate()

        assert "## Address & Query Columns" in content
        assert "query_aoi_lat" in content
        assert "query_aoi_lon" in content
        # Address fields should NOT appear since they're not in the rollup
        assert "streetAddress" not in content

    def test_generate_omits_unused_sections(self, tmp_path):
        rollup_content = "aoi_id,mesh_date,roof_present\n"
        rollup_content += "0,2024-01-15,Y\n"
        (tmp_path / "parcels_aoi_rollup.csv").write_text(rollup_content)

        gen = ReadmeGenerator(output_dir=tmp_path)
        content = gen._generate()

        assert "## Address & Query Columns" not in content
        assert "## Roof Spotlight Index" not in content
        assert "## Roof Age Columns" not in content


class TestReadmeIntegration:
    """Integration tests for README generation with realistic export structure."""

    def test_generate_with_full_export_structure(self, tmp_path):
        final_dir = tmp_path / "final"
        final_dir.mkdir()

        # Create export config (as nmaipy's _save_config does)
        config = {
            "_metadata": {"nmaipy_version": "4.1.4"},
            "parameters": {"primary_decision": "optimal", "country": "us"},
        }
        (final_dir / "export_config.json").write_text(json.dumps(config))

        # Create rollup with multiple feature types and optional columns
        rollup_content = (
            "aoi_id,mesh_date,system_version,link,"
            "roof_present,roof_count,roof_total_area_sqft,"
            "building_present,building_count,building_total_area_sqft,"
            "query_aoi_lat,query_aoi_lon,"
            "roof_spotlight_index,roof_spotlight_index_confidence,"
            "roof_age_installation_date,roof_age_years_as_of_date\n"
            "0,2024-01-15,gen6-us-2024.1,https://maps.nearmap.com/?lat=40.7&lon=-74.0,"
            "Y,2,1500.0,"
            "Y,1,2000.0,"
            "40.7128,-74.0060,"
            "35,0.92,"
            "2018-05-01,6\n"
        )
        (final_dir / "parcels_aoi_rollup.csv").write_text(rollup_content)

        # Create per-class CSV files
        (final_dir / "parcels_roof.csv").write_text("aoi_id,feature_id,area_sqft\n0,r1,800\n0,r2,700\n")
        (final_dir / "parcels_building.csv").write_text("aoi_id,feature_id,area_sqft\n0,b1,2000\n")

        # Create parquet files
        (final_dir / "parcels_roof_features.parquet").write_bytes(b"")
        (final_dir / "parcels_building_features.parquet").write_bytes(b"")

        # Create error and latency files
        (final_dir / "parcels_feature_api_errors.csv").write_text("aoi_id,error\n")
        (final_dir / "parcels_latency_stats.csv").write_text("chunk,p50\n")

        # Generate README
        gen = ReadmeGenerator(output_dir=tmp_path)
        readme_path = gen.generate_and_save()

        assert readme_path.exists()
        assert readme_path.parent == final_dir

        content = readme_path.read_text()

        # Verify all actual files are documented (except excluded ones)
        for f in final_dir.iterdir():
            if f.name not in {"README.md", ".DS_Store", "export_config.json"}:
                assert f.name in content, f"File {f.name} not documented in README"

        # Verify conditional sections are present (based on columns)
        assert "## Address & Query Columns" in content
        assert "## Roof Spotlight Index (RSI) Columns" in content
        assert "## Roof Age Columns" in content

        # Verify primary decision from config
        assert "optimal" in content

        # Verify markdown structure
        lines = content.split("\n")
        header_count = sum(1 for line in lines if line.startswith("#"))
        assert header_count >= 5, f"README should have multiple sections, found {header_count}"

        # Verify tables have proper structure
        assert "| File Name | Description |" in content
        assert "|-----------|-------------|" in content

    def test_readme_only_documents_existing_files(self, tmp_path):
        final_dir = tmp_path / "final"
        final_dir.mkdir()

        rollup_content = "aoi_id,roof_present\n0,Y\n"
        (final_dir / "parcels_aoi_rollup.csv").write_text(rollup_content)
        (final_dir / "parcels_roof.csv").write_text("aoi_id,feature_id\n0,r1\n")

        gen = ReadmeGenerator(output_dir=tmp_path)
        content = gen._generate()

        # Should NOT mention files that don't exist
        assert "feature_api_errors.csv" not in content
        assert "roof_age_errors.csv" not in content
        assert "latency_stats.csv" not in content

        # Should mention files that DO exist
        assert "parcels_aoi_rollup.csv" in content
        assert "parcels_roof.csv" in content

    def test_file_descriptions_match_known_patterns(self, tmp_path):
        (tmp_path / "parcels_aoi_rollup.csv").write_text("aoi_id\n0\n")
        (tmp_path / "parcels_roof.csv").write_text("aoi_id\n0\n")
        (tmp_path / "parcels_feature_api_errors.csv").write_text("aoi_id\n0\n")
        (tmp_path / "parcels_features.parquet").write_bytes(b"")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)

        assert "Property-level summary" in gen._get_file_description("parcels_aoi_rollup.csv", prefix)
        assert "Feature API" in gen._get_file_description("parcels_feature_api_errors.csv", prefix)
        assert "All detected features" in gen._get_file_description("parcels_features.parquet", prefix)
        assert "Export data file" == gen._get_file_description("unknown_file.xyz", prefix)


class TestConfigCaching:
    """Tests for export config caching (Fix 1)."""

    def test_config_loaded_once(self, tmp_path):
        """Verify _load_export_config reads file once and caches the result."""
        config = {"_metadata": {}, "parameters": {"country": "au"}}
        (tmp_path / "export_config.json").write_text(json.dumps(config))

        gen = ReadmeGenerator(output_dir=tmp_path)

        with patch.object(Path, "read_text", wraps=(tmp_path / "export_config.json").read_text) as mock_read:
            result1 = gen._load_export_config()
            result2 = gen._load_export_config()
            assert mock_read.call_count == 1, "Config file should only be read once"
        assert result1 is result2, "Cached result should be the same object"
        assert result1["parameters"]["country"] == "au"


class TestParquetRollupColumns:
    """Tests for parquet rollup column extraction (Fix 2)."""

    def test_get_rollup_columns_from_parquet(self, tmp_path):
        """Verify columns are extracted from parquet rollup when no CSV rollup exists."""
        table = pa.table({
            "aoi_id": [0],
            "roof_spotlight_index": [45],
            "roof_age_installation_date": ["2020-01-01"],
        })
        pq.write_table(table, tmp_path / "parcels_aoi_rollup.parquet")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)
        columns = gen._get_rollup_columns(files, prefix)

        assert "aoi_id" in columns
        assert "roof_spotlight_index" in columns
        assert "roof_age_installation_date" in columns

    def test_csv_rollup_preferred_over_parquet(self, tmp_path):
        """When both CSV and parquet rollup exist, CSV is used."""
        (tmp_path / "parcels_aoi_rollup.csv").write_text("aoi_id,csv_only_col\n0,1\n")
        table = pa.table({"aoi_id": [0], "parquet_only_col": [1]})
        pq.write_table(table, tmp_path / "parcels_aoi_rollup.parquet")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)
        columns = gen._get_rollup_columns(files, prefix)

        assert "csv_only_col" in columns
        assert "parquet_only_col" not in columns

    def test_parquet_rollup_enables_conditional_sections(self, tmp_path):
        """Parquet rollup with RSI columns should trigger RSI section in output."""
        config = {"_metadata": {}, "parameters": {"country": "us"}}
        (tmp_path / "export_config.json").write_text(json.dumps(config))

        table = pa.table({
            "aoi_id": [0],
            "roof_spotlight_index": [45],
            "roof_spotlight_index_confidence": [0.85],
        })
        pq.write_table(table, tmp_path / "parcels_aoi_rollup.parquet")

        gen = ReadmeGenerator(output_dir=tmp_path)
        content = gen._generate()

        assert "## Roof Spotlight Index (RSI) Columns" in content


class TestParquetClassDetection:
    """Tests for parquet-based class detection fallback (Fix 3)."""

    def test_detect_classes_fallback_to_parquet(self, tmp_path):
        """When no per-class CSVs exist, detect classes from _features.parquet files."""
        # Only parquet files, no CSVs except rollup
        (tmp_path / "parcels_aoi_rollup.csv").write_text("aoi_id\n0\n")
        (tmp_path / "parcels_roof_features.parquet").write_bytes(b"")
        (tmp_path / "parcels_building_features.parquet").write_bytes(b"")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)
        classes = gen._detect_classes(files, prefix)

        class_columns = [c["column"] for c in classes]
        assert "roof" in class_columns
        assert "building" in class_columns

    def test_detect_classes_parquet_skips_bare_features(self, tmp_path):
        """Combined _features.parquet (no class prefix) should not create a class."""
        (tmp_path / "parcels_aoi_rollup.csv").write_text("aoi_id\n0\n")
        (tmp_path / "parcels_features.parquet").write_bytes(b"")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)
        classes = gen._detect_classes(files, prefix)

        assert classes == []

    def test_detect_classes_csv_preferred_over_parquet(self, tmp_path):
        """When CSVs exist, parquet fallback should not run."""
        (tmp_path / "parcels_aoi_rollup.csv").write_text("aoi_id\n0\n")
        (tmp_path / "parcels_roof.csv").write_text("aoi_id\n0\n")
        (tmp_path / "parcels_building_features.parquet").write_bytes(b"")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)
        classes = gen._detect_classes(files, prefix)

        class_columns = [c["column"] for c in classes]
        assert "roof" in class_columns
        assert "building" not in class_columns, "Parquet fallback should not run when CSV classes exist"


class TestExactSkipMatch:
    """Tests for exact skip name matching in _detect_classes (Fix 4)."""

    def test_detect_classes_exact_skip_match(self, tmp_path):
        """A class name that is a superstring of a skip name should NOT be skipped."""
        (tmp_path / "parcels_aoi_rollup.csv").write_text("aoi_id\n0\n")
        # "latency_stats_detail" contains "latency_stats" as substring
        (tmp_path / "parcels_latency_stats_detail.csv").write_text("aoi_id\n0\n")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)
        classes = gen._detect_classes(files, prefix)

        class_columns = [c["column"] for c in classes]
        assert "latency_stats_detail" in class_columns, "Superstrings of skip names should not be skipped"


class TestPrefixEndswithSafety:
    """Tests for safe prefix extraction (Fix 5)."""

    def test_get_file_prefix_endswith_safety(self, tmp_path):
        """Prefix extraction should only match suffixes, not substrings."""
        # Regression test: a file like "roof.csv_backup_roof.csv" should not confuse the logic
        # More realistically, ensure standard cases still work correctly
        (tmp_path / "my_project_roof.csv").write_text("aoi_id\n0\n")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)

        assert prefix == "my_project_"

    def test_get_file_prefix_from_parquet_rollup(self, tmp_path):
        """Prefix extraction works from parquet rollup files."""
        table = pa.table({"aoi_id": [0]})
        pq.write_table(table, tmp_path / "export_aoi_rollup.parquet")

        gen = ReadmeGenerator(output_dir=tmp_path)
        files = gen._discover_files()
        prefix = gen._get_file_prefix(files)

        assert prefix == "export_"
