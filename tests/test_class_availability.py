"""Tests for classifying and filtering feature classes by availability under the active system version."""
import pytest

from nmaipy.feature_api import class_returnable_status


# Availability fixture shapes observed from classes.json. Each availability value
# is a list of {systemVersion, perspective, status} entries, where status is one
# of "prod", "alpha", "beta".
HVAC_AVAILABILITY = [
    {"systemVersion": "gen6-glowing_lantern-1.0", "perspective": "Vert", "status": "alpha"},
    {"systemVersion": "gen6-glowing_moon-1.0", "perspective": "Vert", "status": "alpha"},
]

POWER_POLE_AVAILABILITY = [
    {"systemVersion": "gen4-building_storm-2.3", "perspective": "Vert", "status": "alpha"},
    {"systemVersion": "gen4-lightning_bolt-1.3", "perspective": "Vert", "status": "alpha"},
]

SWIMMING_POOL_AVAILABILITY = [
    {"systemVersion": "gen4-building_storm-2.3", "perspective": "Vert", "status": "prod"},
    {"systemVersion": "gen5-tranquil_sea-1.0", "perspective": "Vert", "status": "prod"},
    {"systemVersion": "gen6-glowing_moon-1.0", "perspective": "Vert", "status": "prod"},
]

BETA_ONLY_GEN6 = [
    {"systemVersion": "gen6-glowing_moon-1.0", "perspective": "Vert", "status": "beta"},
]

MIXED_PROD_ALPHA_GEN6 = [
    {"systemVersion": "gen6-glowing_lantern-1.0", "perspective": "Vert", "status": "alpha"},
    {"systemVersion": "gen6-glowing_moon-1.0", "perspective": "Vert", "status": "prod"},
]


class TestClassReturnableStatus:
    def test_alpha_gated_without_flag(self):
        assert class_returnable_status(HVAC_AVAILABILITY, "gen6-", alpha=False) == "alpha_gated"

    def test_alpha_gated_with_flag_is_returnable(self):
        assert class_returnable_status(HVAC_AVAILABILITY, "gen6-", alpha=True) == "returnable"

    def test_beta_gated_without_flag(self):
        assert class_returnable_status(BETA_ONLY_GEN6, "gen6-", beta=False) == "beta_gated"

    def test_beta_gated_with_flag_is_returnable(self):
        assert class_returnable_status(BETA_ONLY_GEN6, "gen6-", beta=True) == "returnable"

    def test_absent_from_gen6(self):
        # Power Pole has only gen4 entries — no gen6 availability at any tier.
        assert class_returnable_status(POWER_POLE_AVAILABILITY, "gen6-", alpha=True, beta=True) == "absent"

    def test_absent_still_absent_regardless_of_flags(self):
        assert class_returnable_status(POWER_POLE_AVAILABILITY, "gen6-") == "absent"

    def test_ga_class_returnable(self):
        assert class_returnable_status(SWIMMING_POOL_AVAILABILITY, "gen6-") == "returnable"

    def test_prefix_switches_classification(self):
        # Power Pole is absent in gen6 but returnable (as alpha) in gen4 with --alpha.
        assert class_returnable_status(POWER_POLE_AVAILABILITY, "gen4-", alpha=True) == "returnable"
        assert class_returnable_status(POWER_POLE_AVAILABILITY, "gen4-", alpha=False) == "alpha_gated"

    def test_mixed_statuses_prod_wins(self):
        # If any entry under the prefix is prod, the class is returnable.
        assert class_returnable_status(MIXED_PROD_ALPHA_GEN6, "gen6-") == "returnable"

    def test_none_prefix_is_unknown(self):
        assert class_returnable_status(HVAC_AVAILABILITY, None) == "unknown"

    def test_empty_prefix_is_unknown(self):
        assert class_returnable_status(HVAC_AVAILABILITY, "") == "unknown"

    def test_non_list_availability_is_unknown(self):
        assert class_returnable_status(None, "gen6-") == "unknown"
        assert class_returnable_status({"gen6": "alpha"}, "gen6-") == "unknown"

    def test_malformed_entries_ignored(self):
        # Entries that aren't dicts or lack a string systemVersion are skipped.
        availability = [
            "not a dict",
            {"perspective": "Vert", "status": "prod"},  # no systemVersion
            {"systemVersion": 42, "status": "prod"},  # non-string systemVersion
            {"systemVersion": "gen6-glowing_moon-1.0", "perspective": "Vert", "status": "prod"},
        ]
        assert class_returnable_status(availability, "gen6-") == "returnable"

    def test_missing_status_falls_through(self):
        # Entry matches prefix but has no recognized status — returns "unknown".
        availability = [{"systemVersion": "gen6-glowing_moon-1.0", "status": "unknown_tier"}]
        assert class_returnable_status(availability, "gen6-") == "unknown"
