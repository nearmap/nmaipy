"""Tests for per-version class returnability used to null out rollup columns.

Availability fixtures are loaded from a real `/classes.json` payload cached at
`tests/data/classes_availability_sample.json`. This is intentionally live data
rather than hand-constructed mocks — future API shape changes should surface
here as test failures, which is informative. Regenerate via the
`test_gen_classes_availability_sample` generator below (uncomment the skip).

A few synthetic fixtures remain only for cases not currently observed in real
data — e.g. a class with both alpha and beta statuses at the same
systemVersion. Those are marked as hypothetical.
"""

import json
from pathlib import Path

import pytest

from nmaipy.feature_api import FeatureApi, is_class_returnable_at_version

CLASSES_SAMPLE_PATH = Path(__file__).parent / "data" / "classes_availability_sample.json"


@pytest.mark.skip("Comment out this line if you wish to regen the test data")
@pytest.mark.live_api
def test_gen_classes_availability_sample():
    """Fetch a fresh /classes.json snapshot and overwrite the cached sample.

    Matches the `test_gen_data_*` pattern elsewhere in the suite — not run by
    default; uncomment the skip when the cached sample needs refreshing
    (e.g. after a catalog change broke downstream assertions).
    """
    api = FeatureApi()
    df = api.get_feature_classes()
    api.cleanup()
    # Reconstruct a minimal payload of the shape /classes.json returns. Only
    # the fields we actually consume are preserved — keeps the fixture small
    # and resistant to incidental schema churn.
    classes = [
        {
            "id": class_id,
            "description": row["description"],
            "type": row.get("type", "Feature"),
            "availability": row.get("availability") or [],
        }
        for class_id, row in df.iterrows()
    ]
    CLASSES_SAMPLE_PATH.write_text(json.dumps({"classes": classes}, indent=2))
    assert CLASSES_SAMPLE_PATH.exists()


@pytest.fixture(scope="module")
def real_classes():
    """Real /classes.json payload — maps class description to availability list."""
    data = json.loads(CLASSES_SAMPLE_PATH.read_text())
    return {c["description"]: c.get("availability") or [] for c in data["classes"] if c.get("type") == "Feature"}


class TestRealClassesProdAndAlpha:
    """Exercise `is_class_returnable_at_version` against real availability data.
    These assertions reflect the catalog state at the time the sample was cached;
    they'll break if a class's gating changes, which is intentional.
    """

    def test_swimming_pool_prod_at_gen6(self, real_classes):
        # Swimming Pool is prod under gen6 → returnable regardless of flags.
        avail = real_classes["Swimming Pool"]
        assert is_class_returnable_at_version(avail, "gen6-glowing_moon-1.0") is True
        assert is_class_returnable_at_version(avail, "gen6-glowing_moon-1.0", alpha=False, beta=False) is True

    def test_hvac_alpha_only_in_gen6(self, real_classes):
        # HVAC is alpha at both gen6-glowing_lantern and gen6-glowing_moon.
        avail = real_classes["HVAC"]
        assert is_class_returnable_at_version(avail, "gen6-glowing_moon-1.0", alpha=False) is False
        assert is_class_returnable_at_version(avail, "gen6-glowing_moon-1.0", alpha=True) is True
        # Beta flag alone does not unlock an alpha-gated class.
        assert is_class_returnable_at_version(avail, "gen6-glowing_moon-1.0", alpha=False, beta=True) is False

    def test_power_pole_absent_from_gen6(self, real_classes):
        # Power Pole exists only in gen4 — absent at gen6 regardless of flags.
        avail = real_classes["Power Pole"]
        assert is_class_returnable_at_version(avail, "gen6-glowing_moon-1.0", alpha=True, beta=True) is False
        # But available at gen4 with --alpha.
        assert is_class_returnable_at_version(avail, "gen4-building_storm-2.3", alpha=True) is True
        assert is_class_returnable_at_version(avail, "gen4-building_storm-2.3", alpha=False) is False

    def test_wrong_exact_version_within_same_gen(self, real_classes):
        # Class's gen6 entry is for glowing_moon; querying an unlisted gen6
        # sub-version (gen6-nonexistent-1.0) is an exact-match miss — not
        # returnable even with all flags set.
        avail = real_classes["HVAC"]
        assert is_class_returnable_at_version(avail, "gen6-nonexistent-1.0", alpha=True, beta=True) is False


class TestHypotheticalMixedStatuses:
    """Edge cases not currently observed in /classes.json but valid per the API
    contract (a class can in principle carry multiple status entries under the
    same exact systemVersion during a promotion). These guard the logic without
    relying on a shape the API doesn't produce today.
    """

    def test_alpha_and_beta_at_same_version_respects_alpha_flag(self):
        # Regression guard: a prefix-based predecessor dropped this case due to
        # beta-first priority, ignoring that alpha=True satisfied the alpha entry.
        availability = [
            {"systemVersion": "gen6-glowing_moon-1.0", "perspective": "Vert", "status": "alpha"},
            {"systemVersion": "gen6-glowing_moon-1.0", "perspective": "Vert", "status": "beta"},
        ]
        assert is_class_returnable_at_version(availability, "gen6-glowing_moon-1.0", alpha=True, beta=False) is True
        assert is_class_returnable_at_version(availability, "gen6-glowing_moon-1.0", alpha=False, beta=True) is True
        assert is_class_returnable_at_version(availability, "gen6-glowing_moon-1.0", alpha=False, beta=False) is False

    def test_prod_and_alpha_at_same_version_resolves_to_returnable(self):
        # Prod access always wins regardless of additional alpha entries.
        availability = [
            {"systemVersion": "gen6-glowing_moon-1.0", "perspective": "Vert", "status": "alpha"},
            {"systemVersion": "gen6-glowing_moon-1.0", "perspective": "Vert", "status": "prod"},
        ]
        assert is_class_returnable_at_version(availability, "gen6-glowing_moon-1.0") is True


class TestFailOpen:
    """Fail-open behaviour for malformed / missing inputs — prefers phantom
    columns to silent drops."""

    def test_none_system_version_is_returnable(self, real_classes):
        assert is_class_returnable_at_version(real_classes["HVAC"], None, alpha=False) is True

    def test_empty_system_version_is_returnable(self, real_classes):
        assert is_class_returnable_at_version(real_classes["HVAC"], "") is True

    def test_non_list_availability_is_returnable(self):
        assert is_class_returnable_at_version(None, "gen6-glowing_moon-1.0") is True
        assert is_class_returnable_at_version({"gen6": "alpha"}, "gen6-glowing_moon-1.0") is True
        assert is_class_returnable_at_version("garbage", "gen6-glowing_moon-1.0") is True

    def test_malformed_entries_skipped(self):
        availability = [
            "not a dict",
            {"status": "prod"},  # no systemVersion — skipped
            {"systemVersion": "gen6-glowing_moon-1.0", "status": "prod"},  # valid
        ]
        assert is_class_returnable_at_version(availability, "gen6-glowing_moon-1.0") is True

    def test_empty_availability_list_is_absent(self):
        # Empty list is not fail-open: it's a legitimate assertion of no availability.
        assert is_class_returnable_at_version([], "gen6-glowing_moon-1.0") is False
