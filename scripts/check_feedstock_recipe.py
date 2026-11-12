"""Check that the conda-forge feedstock recipe can build the nmaipy version in this checkout.

The bot that updates the feedstock after a PyPI release changes the recipe's version, hash and run dependencies, but
never its host requirements or `context.python_min`. A change to `[build-system].requires` or `requires-python` in
`pyproject.toml` therefore needs a feedstock pull request before the release. This check fails if the recipe does not
have the build backend in its host requirements and the `requires-python` floor as `python_min`, and it fails on any
recipe it cannot read.
"""

from pathlib import Path
import sys
import tomllib
import urllib.request

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
import yaml

RECIPE_URL = "https://raw.githubusercontent.com/conda-forge/nmaipy-feedstock/main/recipe/recipe.yaml"
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
FIX = "Open a feedstock pull request first, see VERSIONING.md § conda-forge feedstock."


def main() -> int:
    """Check the live feedstock recipe against pyproject.toml; print each problem and return 1 on one, else 0."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    with urllib.request.urlopen(RECIPE_URL, timeout=30) as response:
        recipe_text = response.read().decode("utf-8")
    problems = check(recipe_text, pyproject)
    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        return 1
    print(f"the recipe at {RECIPE_URL} matches the build backend and the Python floor of pyproject.toml")
    return 0


def check(recipe_text: str, pyproject: dict) -> list[str]:
    """Return one message per problem that stops the recipe building this pyproject; empty when it is ready."""
    recipe = yaml.safe_load(recipe_text)
    if not isinstance(recipe, dict):
        return [_problem("The recipe is not a YAML mapping.")]
    problems = []
    schema_version = recipe.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        problems.append(_problem(f"The recipe has schema_version {schema_version!r}, not 1."))
    python_min = _python_min(recipe, problems)
    host_names = _host_names(recipe, problems)
    floor = _python_floor(pyproject, problems)
    if python_min is not None and floor is not None and python_min != floor:
        problems.append(_problem(f"The recipe has python_min {python_min!r}, but requires-python needs {floor}."))
    if host_names is not None:
        for name in _build_requirement_names(pyproject, problems):
            if name not in host_names:
                problems.append(_problem(f"The recipe has no build requirement {name} in requirements.host."))
    return problems


def _problem(message: str) -> str:
    return f"{message} {FIX}"


def _python_min(recipe: dict, problems: list[str]) -> str | None:
    context = recipe.get("context")
    python_min = context.get("python_min") if isinstance(context, dict) else None
    if not isinstance(python_min, str):
        problems.append(_problem(f"The recipe has context.python_min {python_min!r}, not a string."))
        return None
    return python_min


def _host_names(recipe: dict, problems: list[str]) -> set[str] | None:
    requirements = recipe.get("requirements")
    host = requirements.get("host") if isinstance(requirements, dict) else None
    if not isinstance(host, list) or not all(isinstance(entry, str) and entry.split() for entry in host):
        problems.append(_problem(f"requirements.host of the recipe is {host!r}, not a list of package strings."))
        return None
    return {canonicalize_name(entry.split()[0]) for entry in host}


def _python_floor(pyproject: dict, problems: list[str]) -> str | None:
    requires_python = pyproject.get("project", {}).get("requires-python")
    try:
        specifiers = list(SpecifierSet(requires_python)) if isinstance(requires_python, str) else []
    except InvalidSpecifier:
        specifiers = []
    parts = specifiers[0].version.split(".") if len(specifiers) == 1 and specifiers[0].operator == ">=" else []
    if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
        problems.append(_problem(f"requires-python is {requires_python!r}, not one >=X.Y or >=X.Y.Z specifier."))
        return None
    return ".".join(parts[:2])


def _build_requirement_names(pyproject: dict, problems: list[str]) -> list[str]:
    requires = pyproject.get("build-system", {}).get("requires")
    try:
        if isinstance(requires, list) and requires and all(isinstance(entry, str) for entry in requires):
            return [canonicalize_name(Requirement(entry).name) for entry in requires]
    except InvalidRequirement:
        pass
    problems.append(_problem(f"[build-system].requires is {requires!r}, not a list of requirement strings."))
    return []


if __name__ == "__main__":
    sys.exit(main())
