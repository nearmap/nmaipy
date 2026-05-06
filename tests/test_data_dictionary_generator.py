"""Tests for the data dictionary generator and column metadata loader.

Covers the cases enumerated in the INDS-2080 plan (NearmapAIExporter outputs only;
the standalone Roof Age exporter is intentionally out of scope):
- Known-column lookup
- Scope-aware pattern matcher (parcel-scope vs primary-feature scope)
- Generic suffix patterns
- Unknown columns -> sentinel
- Order preservation
- File discovery (skips operational/error files; ignores existing dictionaries)
- Per-class file coverage
- Atomic write
- Source attribution (input data / base ai model / score model / roof age model)
- JSON round-trip (PM editability)
- JSON validation (malformed JSON / missing keys)
- Confidence-null caveat in pattern descriptions
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nmaipy import column_metadata as cm
from nmaipy.column_metadata import ColumnMeta, lookup_column, reload_metadata
from nmaipy.data_dictionary_generator import DataDictionaryGenerator, _DD_COLUMNS


@pytest.fixture(autouse=True)
def _reset_metadata_cache():
    """Ensure the loader cache doesn't leak between tests that may swap configs."""
    reload_metadata()
    yield
    reload_metadata()


def _write_csv(path: Path, columns: list[str]) -> None:
    """Write a tiny CSV file with just the header row (no data needed)."""
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def _write_parquet(path: Path, columns: list[str]) -> None:
    """Write a tiny parquet with the given columns (zero rows)."""
    schema = pa.schema([pa.field(c, pa.string()) for c in columns])
    pq.write_table(pa.Table.from_pylist([], schema=schema), path)


# ---------------------------------------------------------------------------
# 1. Known-column lookup
# ---------------------------------------------------------------------------


class TestKnownColumnLookup:
    """Curated entries in exact_matches return their stored metadata."""

    def test_aoi_id(self):
        meta = lookup_column("aoi_id")
        assert meta.dtype == "string \\| int"
        assert "Property identifier" in meta.description
        assert meta.source == "input data"

    def test_roof_age_kind(self):
        meta = lookup_column("roof_age_kind")
        assert meta.dtype == "string"
        assert meta.allowed_values == "roof | parcel"
        assert meta.source == "roof age model"

    def test_dominant_roof_material_ratio(self):
        meta = lookup_column("dominant_roof_material_ratio")
        assert meta.dtype == "float"
        assert meta.min == "0.0"
        assert meta.max == "1.0"
        # Curated description (not the generic ? from the _ratio pattern).
        assert meta.description != "?"
        assert "dominant material" in meta.description.lower()


# ---------------------------------------------------------------------------
# 2. Scope-aware pattern matcher (the critical PM concern)
# ---------------------------------------------------------------------------


class TestScopeAwarePatterns:
    """staining_area_sqft (parcel) and primary_roof_staining_area_sqft must differ."""

    def test_parcel_scope_vs_primary_roof_scope_descriptions_differ(self):
        parcel = lookup_column("staining_area_sqft", area_unit="sqft")
        primary = lookup_column("primary_roof_staining_area_sqft", area_unit="sqft")
        assert parcel.description != primary.description

    def test_parcel_scope_phrase(self):
        parcel = lookup_column("staining_area_sqft", area_unit="sqft")
        assert "across the entire parcel" in parcel.description
        assert "primary roof" not in parcel.description

    def test_primary_roof_scope_phrase(self):
        primary = lookup_column("primary_roof_staining_area_sqft", area_unit="sqft")
        assert "primary roof" in primary.description

    def test_ratio_scopes_differ(self):
        parcel = lookup_column("staining_ratio", area_unit="sqft")
        primary = lookup_column("primary_roof_staining_ratio", area_unit="sqft")
        assert parcel.description != primary.description
        assert "across the entire parcel" in parcel.description
        assert "primary roof" in primary.description

    def test_confidence_scopes_differ(self):
        parcel = lookup_column("staining_confidence", area_unit="sqft")
        primary = lookup_column("primary_roof_staining_confidence", area_unit="sqft")
        assert parcel.description != primary.description
        assert "across the entire parcel" in parcel.description
        assert "primary roof" in primary.description


# ---------------------------------------------------------------------------
# 3. Generic suffix patterns
# ---------------------------------------------------------------------------


class TestGenericPatterns:
    """Each suffix family produces the expected dtype/source default."""

    @pytest.mark.parametrize("name, dtype, expect_min", [
        ("tile_area_sqft", "float", "0"),
        ("metal_area_sqm", "float", "0"),
    ])
    def test_area_pattern(self, name, dtype, expect_min):
        unit = "sqft" if name.endswith("_sqft") else "sqm"
        meta = lookup_column(name, area_unit=unit)
        assert meta.dtype == dtype
        assert meta.min == expect_min
        assert meta.source == "base ai model"

    def test_count_pattern(self):
        meta = lookup_column("solar_count", area_unit="sqft")
        assert meta.dtype == "int"
        assert meta.min == "0"

    def test_present_pattern(self):
        meta = lookup_column("solar_present", area_unit="sqft")
        assert meta.dtype == "Y/N"
        assert meta.allowed_values == "Y | N"

    def test_confidence_includes_null_caveat(self):
        meta = lookup_column("tile_confidence", area_unit="sqft")
        assert meta.dtype == "float (quantised uint8)"
        assert "Null when the corresponding area is 0" in meta.description


# ---------------------------------------------------------------------------
# 4. Unknown columns
# ---------------------------------------------------------------------------


class TestUnknownColumns:
    def test_completely_unknown_column_returns_sentinel(self):
        meta = lookup_column("totally_invented_column_name_xyz")
        assert meta.is_unknown()
        assert meta.description == "?"
        assert meta.source == "?"
        assert meta.dtype == ""


# ---------------------------------------------------------------------------
# 5. Source attribution
# ---------------------------------------------------------------------------


class TestSourceAttribution:
    def test_aoi_id_is_input_data(self):
        assert lookup_column("aoi_id").source == "input data"

    def test_roof_age_columns_are_roof_age_model(self):
        assert lookup_column("roof_age_installation_date").source == "roof age model"
        assert lookup_column("roof_age_trust_score").source == "roof age model"

    def test_pack_columns_are_base_ai_model(self):
        assert lookup_column("tile_area_sqft", area_unit="sqft").source == "base ai model"
        assert lookup_column("gable_ratio", area_unit="sqft").source == "base ai model"

    def test_rsi_is_score_model(self):
        assert lookup_column("roof_spotlight_index").source == "score model"
        assert lookup_column("roof_spotlight_index_confidence").source == "score model"


# ---------------------------------------------------------------------------
# 6. Order preservation + per-file generation
# ---------------------------------------------------------------------------


class TestColumnOrderPreservation:
    def test_dictionary_rows_match_source_column_order(self, tmp_path):
        # Shuffled order, deliberately not alphabetical.
        cols = ["zip", "aoi_id", "tile_area_sqft", "mesh_date", "roof_age_kind"]
        _write_csv(tmp_path / "rollup.csv", cols)
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()
        dd = pd.read_csv(tmp_path / "rollup_data_dictionary.csv")
        assert list(dd["column_name"]) == cols


# ---------------------------------------------------------------------------
# 7. File discovery
# ---------------------------------------------------------------------------


class TestFileDiscovery:
    def test_skips_operational_and_error_files(self, tmp_path):
        _write_csv(tmp_path / "rollup.csv", ["aoi_id"])
        _write_parquet(tmp_path / "roof_features.parquet", ["aoi_id", "tile_area_sqft"])
        _write_csv(tmp_path / "latency_stats.csv", ["chunk_id", "p50"])
        _write_csv(tmp_path / "feature_api_errors.csv", ["aoi_id", "status_code"])
        _write_csv(tmp_path / "roof_age_errors.csv", ["aoi_id", "status_code"])
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()

        # Only rollup and roof_features should have dictionaries.
        existing = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".csv" and p.name.endswith("_data_dictionary.csv"))
        assert set(existing) == {"roof_features_data_dictionary.csv", "rollup_data_dictionary.csv"}

    def test_does_not_recursively_dictionary_existing_dictionaries(self, tmp_path):
        _write_csv(tmp_path / "rollup.csv", ["aoi_id"])
        # Pre-create a stray dictionary file as if from a prior run.
        _write_csv(tmp_path / "stale_data_dictionary.csv", ["column_name", "dtype"])
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()
        # No "stale_data_dictionary_data_dictionary.csv" should be produced.
        assert not (tmp_path / "stale_data_dictionary_data_dictionary.csv").exists()

    def test_all_per_class_files_get_dictionaries(self, tmp_path):
        per_class = [
            "building_lifecycle.csv", "building.csv", "roof.csv", "roof_instance.csv",
            "swimming_pool.csv", "solar_panel.csv",
            "building_lifecycle_features.parquet", "building_features.parquet",
            "roof_features.parquet", "roof_instance_features.parquet",
            "swimming_pool_features.parquet", "solar_panel_features.parquet",
        ]
        for name in per_class:
            (tmp_path / name).touch()
            if name.endswith(".csv"):
                _write_csv(tmp_path / name, ["aoi_id"])
            else:
                _write_parquet(tmp_path / name, ["aoi_id"])
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()
        for name in per_class:
            stem = name.rsplit(".", 1)[0]
            dd = tmp_path / f"{stem}_data_dictionary.csv"
            assert dd.exists(), f"missing dictionary for {name}"


# ---------------------------------------------------------------------------
# 9. Atomic write
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_does_not_clobber_existing_dictionary_on_failure(self, tmp_path, monkeypatch):
        """If row generation throws mid-file, the previous dictionary stays intact."""
        # Pre-existing dictionary content.
        existing_dd = tmp_path / "rollup_data_dictionary.csv"
        existing_dd.write_text("column_name,description\nlegacy_col,prior content\n")
        _write_csv(tmp_path / "rollup.csv", ["aoi_id"])

        # Force the write helper to fail mid-flight.
        from nmaipy import data_dictionary_generator as ddg

        def _boom(path, rows):
            raise OSError("simulated mid-write failure")

        monkeypatch.setattr(ddg.DataDictionaryGenerator, "_write_csv", _boom)
        # Should not raise (failures are caught per-file in generate_and_save).
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()

        # Existing dictionary content is preserved.
        assert existing_dd.read_text().startswith("column_name,description\nlegacy_col,prior content")


# ---------------------------------------------------------------------------
# 10. JSON round-trip and validation
# ---------------------------------------------------------------------------


class TestJsonConfig:
    def test_round_trip_edit_appears_in_output(self, tmp_path, monkeypatch):
        """Editing the JSON, reloading, and regenerating reflects the change."""
        # Load real config, mutate one description, write it to a temp file, point loader at it.
        import importlib.resources as resources

        with resources.files("nmaipy.data").joinpath("column_metadata.json").open() as fh:
            payload = json.load(fh)
        payload["exact_matches"]["aoi_id"]["description"] = "EDITED IN TEST"
        custom_path = tmp_path / "custom_metadata.json"
        custom_path.write_text(json.dumps(payload))

        # Force loader to use the custom path.
        reload_metadata()
        from nmaipy import column_metadata as col_mod

        col_mod.load_metadata.cache_clear()
        original = col_mod.load_metadata

        def _patched(json_path=None):
            return original(str(custom_path))

        monkeypatch.setattr(col_mod, "load_metadata", _patched)

        meta = col_mod.lookup_column("aoi_id")
        assert "EDITED IN TEST" in meta.description

    def test_malformed_json_raises_friendly_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not: valid json")
        with pytest.raises(json.JSONDecodeError):
            cm.load_metadata.cache_clear()
            cm.load_metadata(str(bad))

    def test_missing_required_key_raises_friendly_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"exact_matches": {}}))  # missing 'patterns'
        with pytest.raises(ValueError, match="patterns"):
            cm.load_metadata.cache_clear()
            cm.load_metadata(str(bad))

    def test_invalid_pattern_regex_raises_friendly_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"exact_matches": {}, "patterns": [{"regex": "(unclosed"}]}))
        with pytest.raises(ValueError, match="regex is invalid"):
            cm.load_metadata.cache_clear()
            cm.load_metadata(str(bad))


# ---------------------------------------------------------------------------
# 11. Output schema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    def test_dictionary_csv_has_expected_columns_in_order(self, tmp_path):
        _write_csv(tmp_path / "rollup.csv", ["aoi_id"])
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()
        dd = pd.read_csv(tmp_path / "rollup_data_dictionary.csv")
        assert list(dd.columns) == _DD_COLUMNS
