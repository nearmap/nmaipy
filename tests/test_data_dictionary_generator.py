"""Tests for the data dictionary generator and column metadata loader.

Covers the cases enumerated in the INDS-2080 plan (NearmapAIExporter outputs only;
the standalone Roof Age exporter is intentionally out of scope):
- Known-column lookup
- Scope-aware pattern matcher (parcel-scope vs primary-feature scope)
- Generic suffix patterns
- Unknown columns -> sentinel
- Order preservation
- Registry-driven file discovery (gated on tabular_file_format)
- Per-class file coverage
- Source attribution (input data / base ai model / score model / roof age model)
- JSON round-trip (PM editability)
- JSON validation (malformed JSON / missing keys)
- Confidence-null caveat in pattern descriptions
- Cross-file structural consistency for shared columns
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
from nmaipy.data_dictionary_generator import _DD_COLUMNS, DataDictionaryGenerator


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


def _write_export_config(path: Path, *, tabular_file_format: str = "csv", country: str = "us") -> None:
    """Write a minimal export_config.json that the generator reads for format/country."""
    payload = {"parameters": {"tabular_file_format": tabular_file_format, "country": country}}
    path.write_text(json.dumps(payload))


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

    @pytest.mark.parametrize(
        "name, dtype, expect_min",
        [
            ("tile_area_sqft", "float", "0"),
            ("metal_area_sqm", "float", "0"),
        ],
    )
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
        assert meta.dtype == "binary"
        assert meta.allowed_values == "Y | N"
        assert meta.min == "N" and meta.max == "Y"

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
    def test_only_tabular_ai_files_get_dictionaries(self, tmp_path):
        """Operational/error/geometry files are not in the registry — they get no dictionary."""
        _write_export_config(tmp_path / "export_config.json")
        _write_csv(tmp_path / "rollup.csv", ["aoi_id"])
        _write_parquet(tmp_path / "roof_features.parquet", ["aoi_id", "tile_area_sqft"])
        _write_csv(tmp_path / "latency_stats.csv", ["chunk_id", "p50"])
        _write_csv(tmp_path / "feature_api_errors.csv", ["aoi_id", "status_code"])
        _write_csv(tmp_path / "roof_age_errors.csv", ["aoi_id", "status_code"])
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()

        existing = sorted(
            p.name for p in tmp_path.iterdir() if p.suffix == ".csv" and p.name.endswith("_data_dictionary.csv")
        )
        # Only the tabular AI file (rollup.csv) gets a dictionary. The geoparquet
        # companion (roof_features.parquet) and the operational files do not.
        assert existing == ["rollup_data_dictionary.csv"]

    def test_one_dictionary_per_logical_class(self, tmp_path):
        """When CSV is the configured format, parquet variants get no parallel dictionary."""
        _write_export_config(tmp_path / "export_config.json", tabular_file_format="csv")
        # CSV variant is the configured tabular format → gets a dictionary.
        _write_csv(tmp_path / "building.csv", ["aoi_id"])
        # Geoparquet companion → no dictionary (geometry kind, not tabular AI).
        _write_parquet(tmp_path / "building_features.parquet", ["aoi_id", "tile_area_sqft"])
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()

        existing = sorted(p.name for p in tmp_path.iterdir() if p.name.endswith("_data_dictionary.csv"))
        assert existing == ["building_data_dictionary.csv"]

    def test_per_class_files_in_configured_format_get_dictionaries(self, tmp_path):
        """Each per-class tabular file in the configured format gets exactly one dictionary."""
        _write_export_config(tmp_path / "export_config.json", tabular_file_format="csv")
        per_class = [
            "building_lifecycle.csv",
            "building.csv",
            "roof.csv",
            "roof_instance.csv",
            "swimming_pool.csv",
            "solar_panel.csv",
        ]
        for name in per_class:
            _write_csv(tmp_path / name, ["aoi_id"])
        # Geoparquet companions exist alongside but should not get dictionaries.
        for name in [
            "building_lifecycle_features.parquet",
            "building_features.parquet",
            "roof_features.parquet",
            "roof_instance_features.parquet",
            "swimming_pool_features.parquet",
            "solar_panel_features.parquet",
        ]:
            _write_parquet(tmp_path / name, ["aoi_id"])
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()
        for name in per_class:
            stem = name.rsplit(".", 1)[0]
            assert (tmp_path / f"{stem}_data_dictionary.csv").exists(), f"missing dictionary for {name}"
        # Geoparquet companions get no dictionary.
        for name in ["roof_features", "building_features"]:
            assert not (
                tmp_path / f"{name}_data_dictionary.csv"
            ).exists(), f"unexpected dictionary for geometry file {name}"

    def test_parquet_format_picks_parquet_variants(self, tmp_path):
        """When tabular_file_format='parquet', only the parquet variants get dictionaries."""
        _write_export_config(tmp_path / "export_config.json", tabular_file_format="parquet")
        _write_parquet(tmp_path / "rollup.parquet", ["aoi_id"])
        _write_parquet(tmp_path / "building.parquet", ["aoi_id"])
        # A stray CSV variant exists alongside but isn't the configured format.
        _write_csv(tmp_path / "rollup.csv", ["aoi_id"])
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()
        existing = sorted(p.name for p in tmp_path.iterdir() if p.name.endswith("_data_dictionary.csv"))
        assert existing == ["building_data_dictionary.csv", "rollup_data_dictionary.csv"]


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

    def test_does_not_dictionary_unregistered_files(self, tmp_path):
        """Registry-driven discovery skips anything not in output_files."""
        _write_csv(tmp_path / "rollup.csv", ["aoi_id"])
        # A file with no registry entry — should be ignored, not produce a dictionary.
        _write_csv(tmp_path / "some_random_export.csv", ["foo"])
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()
        assert not (tmp_path / "some_random_export_data_dictionary.csv").exists()


# ---------------------------------------------------------------------------
# 12. Cross-file structural consistency
# ---------------------------------------------------------------------------


SHARED_COLUMNS = [
    "aoi_id",
    "feature_id",
    "is_primary",
    "confidence",
    "roof_age_installation_date",
    "roof_age_trust_score",
]


class TestCrossFileConsistency:
    """Columns appearing in multiple output files must have identical structural metadata.

    Descriptions are allowed to vary (scope/class-label phrasing). Anything that
    would surprise a downstream consumer reading two dictionaries side-by-side
    (dtype, min, max, unit, allowed_values) must be identical.
    """

    def test_shared_columns_have_identical_structural_metadata(self, tmp_path):
        _write_export_config(tmp_path / "export_config.json", tabular_file_format="csv")
        # Each column appears in every per-class file plus the rollup.
        common = SHARED_COLUMNS + ["primary_roof_confidence"]
        _write_csv(tmp_path / "rollup.csv", common)
        _write_csv(tmp_path / "roof.csv", common)
        _write_csv(tmp_path / "building.csv", common)
        _write_csv(tmp_path / "roof_instance.csv", common)
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()

        # Load each generated dictionary, build {column → list of (file, structural-tuple)}.
        observations: dict[str, list[tuple[str, tuple]]] = {}
        for dd in tmp_path.glob("*_data_dictionary.csv"):
            df = pd.read_csv(dd, keep_default_na=False)
            for _, row in df.iterrows():
                col = row["column_name"]
                if col not in SHARED_COLUMNS:
                    continue
                structural = (row["dtype"], row["min"], row["max"], row["allowed_values"])
                observations.setdefault(col, []).append((dd.name, structural))

        # Every shared column should have appeared in 2+ dictionaries; check parity.
        for col, entries in observations.items():
            assert len(entries) >= 2, f"shared column {col!r} only seen in one file: {entries}"
            unique = {struct for _, struct in entries}
            assert len(unique) == 1, f"structural metadata for {col!r} differs across files: {entries}"


# ---------------------------------------------------------------------------
# 13. Pass-through user input columns
# ---------------------------------------------------------------------------


class TestUserInputColumns:
    """Arbitrary columns from the input AOI file should be labelled as user-supplied
    rather than rendering as the unknown sentinel. Triggers when the input file
    is reachable via the ``aoi_file`` recorded in ``export_config.json``.
    """

    def _write_aoi_config(self, dirpath: Path, aoi_file: Path) -> None:
        payload = {
            "parameters": {
                "tabular_file_format": "csv",
                "country": "us",
                "aoi_file": str(aoi_file),
            }
        }
        (dirpath / "export_config.json").write_text(json.dumps(payload))

    def test_input_columns_described_as_user_provided(self, tmp_path):
        # Input file with columns the metadata layer has never heard of.
        aoi_path = tmp_path / "input_aois.csv"
        _write_csv(aoi_path, ["aoi_id", "external_id", "force_new", "address"])
        # Rollup carries the same columns through plus a known one.
        _write_csv(
            tmp_path / "rollup.csv",
            ["aoi_id", "external_id", "force_new", "address", "tile_area_sqft"],
        )
        self._write_aoi_config(tmp_path, aoi_path)
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()

        dd = pd.read_csv(tmp_path / "rollup_data_dictionary.csv", keep_default_na=False)
        rows = {r["column_name"]: r for _, r in dd.iterrows()}
        for col in ("external_id", "force_new", "address"):
            assert (
                "Input column provided by user" in rows[col]["description"]
            ), f"{col} should be labelled as user-supplied"
            assert rows[col]["source"] == "input data"
        # Known columns are unaffected.
        assert "?" not in rows["tile_area_sqft"]["description"]
        assert rows["aoi_id"]["source"] == "input data"  # already curated, unchanged

    def test_unknown_column_not_in_input_still_sentinel(self, tmp_path):
        # If a column appears in output but not in the input file, it remains
        # an unknown sentinel — the override is gated on the input file's
        # column set, not a blanket fallback.
        aoi_path = tmp_path / "input_aois.csv"
        _write_csv(aoi_path, ["aoi_id"])  # only aoi_id in input
        _write_csv(tmp_path / "rollup.csv", ["aoi_id", "totally_unknown_xyz"])
        self._write_aoi_config(tmp_path, aoi_path)
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()

        dd = pd.read_csv(tmp_path / "rollup_data_dictionary.csv", keep_default_na=False)
        rows = {r["column_name"]: r for _, r in dd.iterrows()}
        assert rows["totally_unknown_xyz"]["description"] == "?"
        assert rows["totally_unknown_xyz"]["source"] == "?"

    def test_missing_input_file_falls_back_gracefully(self, tmp_path):
        # If aoi_file is unreadable, generator should not crash and unknown
        # columns continue to render the sentinel (preserving prior behaviour).
        self._write_aoi_config(tmp_path, tmp_path / "does_not_exist.csv")
        _write_csv(tmp_path / "rollup.csv", ["aoi_id", "external_id"])
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()

        dd = pd.read_csv(tmp_path / "rollup_data_dictionary.csv", keep_default_na=False)
        rows = {r["column_name"]: r for _, r in dd.iterrows()}
        assert rows["external_id"]["description"] == "?"

    def test_parquet_input_columns_recognised(self, tmp_path):
        # Parquet input files exercise a different branch of the header reader.
        aoi_path = tmp_path / "input_aois.parquet"
        _write_parquet(aoi_path, ["aoi_id", "external_id", "policy_number"])
        _write_csv(tmp_path / "rollup.csv", ["aoi_id", "external_id", "policy_number"])
        self._write_aoi_config(tmp_path, aoi_path)
        DataDictionaryGenerator(output_dir=tmp_path).generate_and_save()

        dd = pd.read_csv(tmp_path / "rollup_data_dictionary.csv", keep_default_na=False)
        rows = {r["column_name"]: r for _, r in dd.iterrows()}
        for col in ("external_id", "policy_number"):
            assert "Input column provided by user" in rows[col]["description"]
            assert rows[col]["source"] == "input data"


class TestDefensibleSpacePrimaryScope:
    """Regression for INDS-2080: ``primary_defensible_space_zone_*`` columns
    must resolve via regex to a non-fallback description that includes the
    zone band and the "clear of vegetation" / risk-object semantics — not the
    generic underscore-replacement fallback. See BUG_defensible_space_prefix.md
    for the original report. ``parcels.py`` renames resolved roof scores from
    ``primary_roof_*`` to ``primary_*``; the regexes in ``column_metadata.json``
    accept both forms so historical fixtures keep resolving correctly.
    """

    def test_primary_zone_columns_resolve_non_fallback(self):
        cols = [
            "primary_defensible_space_zone_0_zone_area_sqft",
            "primary_defensible_space_zone_0_defensible_space_area_sqft",
            "primary_defensible_space_zone_0_coverage_ratio",
            "primary_defensible_space_zone_0_risk_object_area_sqft",
            "primary_defensible_space_zone_1_coverage_ratio",
            "primary_defensible_space_zone_2_zone_area_sqft",
        ]
        for col in cols:
            meta = lookup_column(col, area_unit="sqft")
            assert not meta.is_unknown(), f"{col} fell through to sentinel"
            assert (
                "around the primary roof on the parcel" in meta.description
            ), f"{col} missing primary-roof scope phrase: {meta.description!r}"
            assert (
                "from the structure" in meta.description
            ), f"{col} missing zone-band substitution: {meta.description!r}"

    def test_primary_coverage_ratio_clear_of_vegetation_semantics(self):
        meta = lookup_column("primary_defensible_space_zone_0_coverage_ratio", area_unit="sqft")
        assert "clear of vegetation" in meta.description
        assert "around the primary roof on the parcel" in meta.description

    def test_primary_per_class_ratio_resolves(self):
        meta = lookup_column("primary_defensible_space_zone_0_yard_debris_ratio", area_unit="sqft")
        assert not meta.is_unknown()
        assert "yard debris" in meta.description
        assert "around the primary roof on the parcel" in meta.description

    def test_legacy_primary_roof_form_still_resolves(self):
        # Option B alternation: keep the historical form working for older fixtures.
        meta = lookup_column("primary_roof_defensible_space_zone_0_coverage_ratio", area_unit="sqft")
        assert not meta.is_unknown()
        assert "around the primary roof on the parcel" in meta.description

    def test_aggregate_form_still_works(self):
        meta = lookup_column("aggregate_defensible_space_zone_0_coverage_ratio", area_unit="sqft")
        assert not meta.is_unknown()
        assert "aggregated across all roofs on the parcel" in meta.description

    def test_unprefixed_form_still_works_for_per_roof_csv(self):
        meta = lookup_column("defensible_space_zone_0_coverage_ratio", area_unit="sqft", class_label="roof")
        assert not meta.is_unknown()
        # Else branch in column_metadata.py → "around this {class_label}".
        assert "around this roof" in meta.description
