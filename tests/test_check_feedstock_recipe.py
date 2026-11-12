import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_feedstock_recipe", ROOT / "scripts" / "check_feedstock_recipe.py")
assert SPEC is not None
assert SPEC.loader is not None
check_feedstock_recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_feedstock_recipe)

RECIPE_5_5_2 = (ROOT / "tests" / "data" / "feedstock_recipe_5.5.2.yaml").read_text(encoding="utf-8")


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"{old!r} is in the recipe {text.count(old)} times, not once"
    return text.replace(old, new)


PREPARED = _replace_once(
    _replace_once(
        _replace_once(RECIPE_5_5_2, 'python_min: "3.12"', 'python_min: "3.13"'),
        "    - pip\n",
        "    - pip\n    - hatchling >=1.27\n",
    ),
    "number: 0",
    "number: 1",
)


@pytest.mark.parametrize(
    ("recipe", "requires_python", "expected"),
    [
        (RECIPE_5_5_2, ">=3.13", [["3.12", "3.13"], ["hatchling"]]),
        (PREPARED, ">=3.13", []),
        (
            _replace_once(PREPARED, "    - wheel\n", "    - if: unix\n      then: wheel\n"),
            ">=3.13",
            [["requirements.host"]],
        ),
        (PREPARED, ">=3.13,<4", [["requires-python"]]),
    ],
    ids=["recipe-5.5.2", "prepared", "host-if-mapping", "requires-python-upper-bound"],
)
def test_check(recipe: str, requires_python: str, expected: list[list[str]]) -> None:
    pyproject = {"build-system": {"requires": ["hatchling>=1.27"]}, "project": {"requires-python": requires_python}}
    problems = check_feedstock_recipe.check(recipe, pyproject)
    assert len(problems) == len(expected), f"expected problems naming {expected}, got {problems}"
    for problem, words in zip(problems, expected, strict=True):
        assert all(word in problem for word in words), f"expected a problem naming {words}, got {problem!r}"
