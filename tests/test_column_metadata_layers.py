"""Tests for the three-layer column metadata stack.

Covers the merge of:
- ``column_metadata.json`` (hand-curated nmaipy-derived columns + patterns)
- ``spec_raw_fields.json`` (auto-generated from the Feature API spec)
- ``column_metadata_overrides.json`` (PM elaborations)

The drift checker in ai-offline-export-pipelines/spec_drift produces and
maintains spec_raw_fields.json; here we only assert the loader-side
invariants that nmaipy itself relies on.
"""

from __future__ import annotations

import json
from importlib import resources

import pytest

from nmaipy.column_metadata import lookup_column, reload_metadata


@pytest.fixture(autouse=True)
def _clear_metadata_cache():
    reload_metadata()
    yield
    reload_metadata()


def _load_json_resource(filename: str) -> dict:
    with resources.files("nmaipy.data").joinpath(filename).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_three_data_files_present():
    """All three layers ship in the wheel."""
    assert _load_json_resource("column_metadata.json")["exact_matches"]
    assert _load_json_resource("spec_raw_fields.json")["exact_matches"]
    # Overrides file is allowed to be empty but the file itself must exist.
    _load_json_resource("column_metadata_overrides.json")


def test_no_stale_overrides():
    """Every override key must correspond to an entry in spec_raw_fields.

    Override entries that don't have a raw-field counterpart are a sign the
    underlying spec field was renamed or removed; remediation belongs in the
    spec_drift skill, not silently masking via overrides.
    """
    spec = _load_json_resource("spec_raw_fields.json")["exact_matches"]
    overrides = _load_json_resource("column_metadata_overrides.json").get("exact_matches", {})
    stale = set(overrides) - set(spec)
    assert not stale, (
        f"Overrides reference keys not present in spec_raw_fields.json: {sorted(stale)}. "
        "Run the spec_drift skill in ai-offline-export-pipelines to reconcile."
    )


def test_spec_sourced_field_resolves():
    """Pick a representative spec field added by the generator and confirm it resolves."""
    meta = lookup_column("class_status")
    assert meta.description, "class_status should resolve to a non-empty description from spec_raw_fields"
    assert "production" in meta.description or "prod" in meta.description


def test_override_wins_over_spec():
    """When an override exists for a key also in spec_raw_fields, the override wins.

    Uses ``survey_date``: spec description is terse; override carries the
    rollup-specific "canonical as-of date" elaboration.
    """
    overrides = _load_json_resource("column_metadata_overrides.json").get("exact_matches", {})
    if "survey_date" not in overrides:
        pytest.skip("survey_date not overridden in this build")
    expected = overrides["survey_date"]["description"]
    meta = lookup_column("survey_date")
    assert meta.description == expected


def test_nmaipy_derived_entries_unaffected():
    """Hand-curated nmaipy entries (no spec counterpart) still resolve correctly."""
    # ``aoi_id`` is nmaipy-only; should resolve via column_metadata.json.
    meta = lookup_column("aoi_id")
    assert "Property identifier" in meta.description


def test_spec_example_surfaces_through_override():
    """Per-field overlay: an override that doesn't carry ``example`` inherits it from spec.

    ``survey_date`` is in both ``spec_raw_fields.json`` (with example
    ``"2019-09-27"``) and ``column_metadata_overrides.json`` (with an
    elaborated description but no ``example`` field). The merged
    ColumnMeta should carry both nmaipy's description AND the spec's
    example.
    """
    meta = lookup_column("survey_date")
    # nmaipy elaboration survives.
    assert "canonical 'as-of' date" in meta.description
    # spec example surfaces.
    assert meta.example == "2019-09-27"


def test_spec_example_absent_when_spec_missing():
    """Columns with no spec counterpart have an empty example."""
    meta = lookup_column("aoi_id")
    assert meta.example == ""
