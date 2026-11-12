"""``snake_case`` matches ``stringcase.snakecase``, which produced nmaipy's column names before."""

import pytest

from nmaipy._naming import snake_case


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("roofSpotlightIndex", "roof_spotlight_index"),
        ("clippedAreaSqft", "clipped_area_sqft"),
        ("NoDamage", "no_damage"),
        ("preEvent", "pre_event"),
        ("feature_id", "feature_id"),
        ("Has3D", "has3_d"),
        ("foo_Bar", "foo__bar"),
        ("Foo Bar", "foo__bar"),
        ("x-y.z\tw", "x_y_z_w"),
        ("", ""),
        ("École", "école"),
        (None, "none"),
        (0, "0"),
    ],
)
def test_snake_case_matches_stringcase(value: object, expected: str) -> None:
    assert snake_case(value) == expected, f"snake_case({value!r})"
