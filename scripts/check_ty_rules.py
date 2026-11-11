"""Check that `[tool.ty.rules]` in pyproject.toml lists exactly the rules the installed ty knows.

ty applies its own default level to any rule the table leaves out, so a ty release that adds a rule could
switch it on unnoticed. This check fails until the table is updated.
"""

import json
from pathlib import Path
import subprocess
import sys
import tomllib

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def main() -> int:
    """Print any mismatch between the table and ty's rules; return 1 if there is one, else 0."""
    known = known_rules()
    problems = mismatches(listed_rules(PYPROJECT), known)
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print(f"[tool.ty.rules] lists all {len(known)} ty rules")
    return 0


def listed_rules(pyproject: Path) -> set[str]:
    """Return the rule names in the `[tool.ty.rules]` table of `pyproject`."""
    return set(tomllib.loads(pyproject.read_text())["tool"]["ty"]["rules"])


def known_rules() -> set[str]:
    """Return the names of all rules the installed ty reports."""
    result = subprocess.run(
        ["ty", "explain", "rule", "--output-format", "json"], capture_output=True, text=True, check=True
    )
    return {rule["name"] for rule in json.loads(result.stdout)}


def mismatches(listed: set[str], known: set[str]) -> list[str]:
    """Return one message for rules ty knows that are not listed, and one for listed rules ty does not know."""
    problems = []
    if absent := sorted(known - listed):
        problems.append(f"[tool.ty.rules] does not list {len(absent)} ty rule(s): {', '.join(absent)}")
    if extra := sorted(listed - known):
        problems.append(f"[tool.ty.rules] lists {len(extra)} rule(s) ty does not know: {', '.join(extra)}")
    return problems


if __name__ == "__main__":
    sys.exit(main())
