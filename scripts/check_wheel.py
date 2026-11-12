"""Check that the installed nmaipy wheel imports and carries its data files.

Run it with `python -I` from a virtual environment where only the wheel was installed with pip. It fails if nmaipy
is imported from anywhere but site-packages, if a module needs a package the wheel does not declare, or if a JSON
file in the source tree is missing from the installed package.
"""

import importlib
from pathlib import Path
import pkgutil
import sys
import sysconfig

import fsspec

import nmaipy

SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "nmaipy"


def main() -> int:
    """Import every nmaipy module from the installed wheel and check its data files; return 1 on a failure, else 0."""
    init_file = Path(nmaipy.__file__).resolve()
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    if not init_file.is_relative_to(purelib):
        print(f"nmaipy was imported from {init_file}, not from site-packages {purelib}")
        return 1
    installed = init_file.parent
    modules = import_all_modules()
    fsspec.get_filesystem_class("s3")
    if missing := missing_json_files(SOURCE_PACKAGE, installed):
        print(f"the installed package at {installed} is missing {len(missing)} JSON file(s): {', '.join(missing)}")
        return 1
    print(f"imported {len(modules)} nmaipy modules from {installed}")
    return 0


def import_all_modules() -> list[str]:
    """Import every module in the nmaipy package and return their names."""
    names = [module.name for module in pkgutil.walk_packages(nmaipy.__path__, "nmaipy.")]
    for name in names:
        importlib.import_module(name)
    return names


def missing_json_files(source: Path, installed: Path) -> list[str]:
    """Return the paths, relative to `source`, of the JSON files under `source` that are absent under `installed`."""
    return sorted(
        str(path.relative_to(source))
        for path in source.rglob("*.json")
        if not (installed / path.relative_to(source)).is_file()
    )


if __name__ == "__main__":
    sys.exit(main())
