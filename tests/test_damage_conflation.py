"""
Tests for Damage Conflation API (ai/damage/v2) flattening and rollup logic.

Uses the real captured Hurricane Milton response fixture
(``tests/data/test_damage_conflation_milton_response.json``) — never mock data,
per the project testing rule.
"""

import json
from pathlib import Path

import pytest

from nmaipy.feature_attributes import flatten_conflated_damage_attributes

# The exact, fixed column set produced by flatten_conflated_damage_attributes.
# Asserting equality (not just membership) catches both missing AND unexpected
# columns if the flattener drifts.
EXPECTED_COLUMNS = {
    "area_sqm",
    "area_sqft",
    "confidence",
    "hilbert_id",
    "damage_event_rating",
    "damage_event_confidence",
    "damage_event_raw_affected",
    "damage_event_raw_destroyed",
    "damage_event_raw_major",
    "damage_event_raw_minor",
    "damage_event_raw_no_damage",
    "damage_event_class_ratios",
    "damage_event_latest_capture_date",
    "damage_pre_event_rating",
    "damage_pre_event_confidence",
    "damage_pre_event_raw_affected",
    "damage_pre_event_raw_destroyed",
    "damage_pre_event_raw_major",
    "damage_pre_event_raw_minor",
    "damage_pre_event_raw_no_damage",
    "damage_pre_event_class_ratios",
    "damage_pre_event_latest_capture_date",
}


@pytest.fixture
def milton_features(data_directory: Path):
    """All features from the real Hurricane Milton conflation response."""
    fixture_path = data_directory / "test_damage_conflation_milton_response.json"
    with open(fixture_path, "r") as f:
        return json.load(f)["features"]


def _feature_with_pre_event(features):
    return next(f for f in features if isinstance(f["properties"]["damage"].get("preEvent"), dict))


def _feature_without_pre_event(features):
    return next(f for f in features if not isinstance(f["properties"]["damage"].get("preEvent"), dict))


def test_flatten_column_set_is_exact(milton_features):
    """Every feature flattens to exactly EXPECTED_COLUMNS — no missing, no extra."""
    for feature in milton_features:
        flat = flatten_conflated_damage_attributes(feature["properties"])
        assert set(flat.keys()) == EXPECTED_COLUMNS, (
            f"column drift: missing={EXPECTED_COLUMNS - set(flat.keys())}, "
            f"extra={set(flat.keys()) - EXPECTED_COLUMNS}"
        )


def test_flatten_event_values_match_source(milton_features):
    """event-block scalar values are copied through verbatim."""
    feature = _feature_with_pre_event(milton_features)
    props = feature["properties"]
    event = props["damage"]["event"]
    flat = flatten_conflated_damage_attributes(props)

    assert flat["area_sqm"] == props["areaSqm"]
    assert flat["area_sqft"] == props["areaSqft"]
    assert flat["confidence"] == props["confidence"]
    assert flat["hilbert_id"] == props["hilbertId"]
    assert flat["damage_event_rating"] == event["rating"]
    assert flat["damage_event_confidence"] == event["confidence"]
    assert flat["damage_event_raw_no_damage"] == event["rawRatings"]["NoDamage"]
    assert flat["damage_event_raw_affected"] == event["rawRatings"]["Affected"]
    # latestCaptureDate lives on preEvent only; event has none.
    assert flat["damage_event_latest_capture_date"] is None


def test_flatten_class_ratios_serialised_to_json(milton_features):
    """classRatios (variable-length list) round-trips through a JSON string column."""
    feature = _feature_with_pre_event(milton_features)
    props = feature["properties"]
    flat = flatten_conflated_damage_attributes(props)

    assert isinstance(flat["damage_event_class_ratios"], str)
    assert json.loads(flat["damage_event_class_ratios"]) == props["damage"]["event"]["classRatios"]


def test_flatten_pre_event_present(milton_features):
    """When preEvent exists, its columns are populated incl. latestCaptureDate."""
    feature = _feature_with_pre_event(milton_features)
    props = feature["properties"]
    pre = props["damage"]["preEvent"]
    flat = flatten_conflated_damage_attributes(props)

    assert flat["damage_pre_event_rating"] == pre["rating"]
    assert flat["damage_pre_event_confidence"] == pre["confidence"]
    assert flat["damage_pre_event_raw_no_damage"] == pre["rawRatings"]["NoDamage"]
    assert flat["damage_pre_event_latest_capture_date"] == pre["latestCaptureDate"]


def test_flatten_pre_event_absent_emits_none(milton_features):
    """When preEvent is absent, all damage_pre_event_* columns are None (stable schema)."""
    feature = _feature_without_pre_event(milton_features)
    flat = flatten_conflated_damage_attributes(feature["properties"])

    pre_event_cols = [c for c in EXPECTED_COLUMNS if c.startswith("damage_pre_event_")]
    assert pre_event_cols, "expected some damage_pre_event_* columns"
    for col in pre_event_cols:
        assert flat[col] is None, f"{col} should be None when preEvent is absent, got {flat[col]!r}"
    # event block is still populated for these features.
    assert flat["damage_event_rating"] is not None


def test_fixture_exercises_both_pre_event_cases(milton_features):
    """Guard: the fixture must contain both preEvent-present and preEvent-absent rows."""
    with_pre = sum(1 for f in milton_features if isinstance(f["properties"]["damage"].get("preEvent"), dict))
    assert 0 < with_pre < len(milton_features), (
        "fixture should have a mix of preEvent-present and preEvent-absent features; "
        f"got {with_pre}/{len(milton_features)}"
    )
