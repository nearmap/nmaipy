"""
Tests for Damage Conflation API (ai/damage/v2) flattening and rollup logic.

Uses the real captured Hurricane Milton response fixture
(``tests/data/test_damage_conflation_milton_response.json``) — never mock data,
per the project testing rule.
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from nmaipy import parcels
from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS
from nmaipy.damage_conflation_api import DamageConflationApi
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


# ----------------------------------------------------------------- rollup tests
def _parse_slice(milton_response, aoi_id, sl):
    """Parse a slice of the fixture as if returned for a single AOI (keeps top-level metadata)."""
    api = DamageConflationApi(event_id="e", api_key="t")
    resp = {**milton_response, "features": milton_response["features"][sl]}
    return api._parse_response(resp, aoi_id)


@pytest.fixture
def rollup_inputs(milton_response):
    """Two AOIs with features, one empty (queried, no buildings), one errored."""
    f_p1 = _parse_slice(milton_response, "p1", slice(0, 5))
    f_p2 = _parse_slice(milton_response, "p2", slice(5, 14))
    features_gdf = gpd.GeoDataFrame(pd.concat([f_p1, f_p2], ignore_index=True), crs=API_CRS)

    aoi_gdf = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1)] * 4,
        crs=API_CRS,
        index=pd.Index(["p1", "p2", "p_empty", "p_error"], name=AOI_ID_COLUMN_NAME),
    )
    successful = {"p1", "p2", "p_empty"}  # p_error not in set
    return aoi_gdf, features_gdf, successful, f_p1, f_p2


def test_conflation_rollup_one_row_per_aoi(rollup_inputs):
    aoi_gdf, features_gdf, successful, _, _ = rollup_inputs
    rollup = parcels.conflation_rollup(aoi_gdf, features_gdf, country="us", successful_aoi_ids=successful)
    assert set(rollup.index) == {"p1", "p2", "p_empty", "p_error"}
    assert len(rollup) == 4


def test_conflation_rollup_counts(rollup_inputs):
    aoi_gdf, features_gdf, successful, f_p1, f_p2 = rollup_inputs
    rollup = parcels.conflation_rollup(aoi_gdf, features_gdf, country="us", successful_aoi_ids=successful)

    assert rollup.loc["p1", "n_buildings"] == len(f_p1)
    assert rollup.loc["p2", "n_buildings"] == len(f_p2)
    # The five per-rating counts partition the buildings exactly.
    rating_count_cols = ["n_no_damage", "n_affected", "n_minor", "n_major", "n_destroyed"]
    assert rollup.loc["p1", rating_count_cols].sum() == len(f_p1)


def test_conflation_rollup_empty_aoi_is_zero_not_null(rollup_inputs):
    aoi_gdf, features_gdf, successful, _, _ = rollup_inputs
    rollup = parcels.conflation_rollup(aoi_gdf, features_gdf, country="us", successful_aoi_ids=successful)

    assert rollup.loc["p_empty", "query_succeeded"]
    assert rollup.loc["p_empty", "n_buildings"] == 0
    assert pd.isna(rollup.loc["p_empty", "primary_feature_id"])


def test_conflation_rollup_errored_aoi_is_nulled(rollup_inputs):
    aoi_gdf, features_gdf, successful, _, _ = rollup_inputs
    rollup = parcels.conflation_rollup(aoi_gdf, features_gdf, country="us", successful_aoi_ids=successful)

    assert not rollup.loc["p_error", "query_succeeded"]
    # Errored AOI nulls out counts (distinguishes "no data" from "zero damaged").
    assert pd.isna(rollup.loc["p_error", "n_buildings"])
    assert pd.isna(rollup.loc["p_error", "primary_feature_id"])


def test_conflation_rollup_primary_is_largest(rollup_inputs):
    aoi_gdf, features_gdf, successful, f_p1, _ = rollup_inputs
    rollup = parcels.conflation_rollup(aoi_gdf, features_gdf, country="us", successful_aoi_ids=successful)

    # default "largest" -> primary is the largest-area_sqm building in the AOI
    expected = f_p1.loc[f_p1["area_sqm"].idxmax()]
    assert rollup.loc["p1", "primary_feature_id"] == expected["feature_id"]
    # country=us -> only the country-correct area unit is carried (sqft), sqm is dropped
    assert rollup.loc["p1", "primary_area_sqft"] == expected["area_sqft"]
    assert "primary_area_sqm" not in rollup.columns
    assert rollup.loc["p1", "primary_damage_event_rating"] == expected["damage_event_rating"]
    # the preEvent block is carried onto the primary too
    assert "primary_damage_pre_event_rating" in rollup.columns


def test_conflation_rollup_carries_event_metadata(rollup_inputs, milton_response):
    aoi_gdf, features_gdf, successful, _, _ = rollup_inputs
    rollup = parcels.conflation_rollup(aoi_gdf, features_gdf, country="us", successful_aoi_ids=successful)
    assert (rollup["event_uuid"] == milton_response["eventUuid"]).all()


@pytest.fixture
def milton_response(data_directory: Path):
    fixture_path = data_directory / "test_damage_conflation_milton_response.json"
    with open(fixture_path, "r") as f:
        return json.load(f)
