"""Check that the installed nmaipy wheel imports and carries its data files.

Run it with `python -I` from a virtual environment where only the wheel was installed with pip. It fails if nmaipy
is imported from anywhere but site-packages, if a module-level import is not satisfied by the installed wheel and its
dependencies, or if a .py or .json file in the source tree is missing from the installed package.
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
    assert SOURCE_PACKAGE.is_dir(), f"no source package at {SOURCE_PACKAGE}"
    init_file = Path(nmaipy.__file__).resolve()
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    if not init_file.is_relative_to(purelib):
        print(f"nmaipy was imported from {init_file}, not from site-packages {purelib}")
        return 1
    installed = init_file.parent
    modules = import_all_modules()
    fsspec.get_filesystem_class("s3")
    expected = source_files(SOURCE_PACKAGE)
    if missing := [name for name in expected if not (installed / name).is_file()]:
        print(f"the installed package at {installed} is missing {len(missing)} file(s): {', '.join(missing)}")
        return 1
    print(f"imported {len(modules)} nmaipy modules from {installed}; found all {len(expected)} .py and .json files")
    return 0


def import_all_modules() -> list[str]:
    """Import every module in the nmaipy package and return their names."""
    names = [module.name for module in pkgutil.walk_packages(nmaipy.__path__, "nmaipy.")]
    for name in names:
        importlib.import_module(name)
    return names


def source_files(source: Path) -> list[str]:
    """Return the paths, relative to `source`, of the .py and .json files under `source`."""
    return sorted(str(path.relative_to(source)) for suffix in ("*.py", "*.json") for path in source.rglob(suffix))


if __name__ == "__main__":
    sys.exit(main())
