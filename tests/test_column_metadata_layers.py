"""Tests for the three-layer column metadata stack.

Covers the merge of:
- ``column_metadata.json`` (hand-curated nmaipy-derived columns + patterns)
- ``spec_raw_fields.json`` (auto-generated from the Feature API spec)
- ``column_metadata_overrides.json`` (PM elaborations)

spec_raw_fields.json is generated from the Feature API OpenAPI spec by a
spec-drift tool; here we only assert the loader-side invariants that nmaipy
itself relies on.
"""

from __future__ import annotations

from importlib import resources
import json

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
    underlying spec field was renamed or removed; remediation belongs in
    regenerating spec_raw_fields.json, not silently masking via overrides.
    """
    spec = _load_json_resource("spec_raw_fields.json")["exact_matches"]
    overrides = _load_json_resource("column_metadata_overrides.json").get("exact_matches", {})
    stale = set(overrides) - set(spec)
    assert not stale, (
        f"Overrides reference keys not present in spec_raw_fields.json: {sorted(stale)}. "
        "Regenerate spec_raw_fields.json from the current API spec and reconcile the overrides."
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


@pytest.mark.parametrize(("unit", "unit_name"), [("sqm", "square metres"), ("sqft", "square feet")])
def test_grid_coverage_columns_resolve(unit, unit_name):
    """The rollup's coverage columns resolve to their own entries, the area one in either unit."""
    for name in ["grid_cells", "grid_cells_answered", "answered_area_fraction"]:
        assert lookup_column(name, area_unit=unit).group == "api_status", name
    meta = lookup_column(f"answered_area_{unit}", area_unit=unit)
    assert "denominator" in meta.description
    assert meta.unit == unit_name


def test_model_input_fields_do_not_shadow_roof_component_columns():
    """Score-model input fields must not describe roof component columns of the same bare name.

    The spec's ``*ModelInputFeatures`` schemas (hurricane, wind, wildfire, ...) have fields such as
    ``treeOverhangRatio`` and ``metalPresent``. nmaipy only emits those under a model-input prefix,
    but their bare names equal the base AI model's roof component columns (``tree_overhang_ratio``,
    ``metal_present``), which must keep their component descriptions.
    """
    spec = _load_json_resource("spec_raw_fields.json")["exact_matches"]
    model_input_keys = sorted(
        name
        for name, entry in spec.items()
        if entry.get("_spec_ref", "").split(".", 1)[0].endswith("ModelInputFeatures")
        and name.endswith(("_present", "_ratio"))
    )
    assert "tree_overhang_ratio" in model_input_keys, "Expected the hurricane tree overhang input in the spec layer"
    for name in model_input_keys:
        meta = lookup_column(name, area_unit="sqft", class_label="roof")
        assert meta.source != "score model", f"{name} resolved to a score-model description: {meta.description!r}"
        for scope in ("primary_roof_", ""):
            scoped = lookup_column(f"{scope}{name}", area_unit="sqft")
            assert scoped.source != "score model", f"{scope}{name} resolved to {scoped.description!r}"

    component = lookup_column("flat_present", class_label="roof")
    assert "shingle" not in component.description.lower(), component.description

    model_input = lookup_column("hurricane_vulnerability_model_input_tree_overhang_ratio")
    assert "hurricane" in model_input.description, model_input.description
