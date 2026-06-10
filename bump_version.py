#!/usr/bin/env python
"""
Version bumping utility for nmaipy.

Usage:
    python bump_version.py patch  # 3.2.1 -> 3.2.2
    python bump_version.py minor  # 3.2.1 -> 3.3.0
    python bump_version.py major  # 3.2.1 -> 4.0.0
    python bump_version.py 3.2.5  # Set specific version
"""

import sys
import re
from pathlib import Path


def read_current_version():
    """Read current version from __version__.py"""
    version_file = Path("nmaipy/__version__.py")
    content = version_file.read_text()
    match = re.search(r'__version__ = "([^"]+)"', content)
    if match:
        return match.group(1)
    raise ValueError("Could not find version in __version__.py")


def write_version(version):
    """Write the new version into __version__.py.

    Substitutes only the version string in place — the rest of the file
    (e.g. the pre-release-safe __version_info__ parsing) is preserved.
    conda.recipe/meta.yaml needs no update: it reads __version__.py
    directly via load_file_regex.
    """
    version_file = Path("nmaipy/__version__.py")
    content = version_file.read_text()
    new_content, n = re.subn(r'__version__ = "[^"]+"', f'__version__ = "{version}"', content)
    if n != 1:
        raise ValueError(f"Expected exactly one __version__ assignment in {version_file}, found {n}")
    version_file.write_text(new_content)
    print(f"✓ Updated nmaipy/__version__.py to {version}")


def bump_version(bump_type):
    """Bump version based on type (major, minor, patch)"""
    current = read_current_version()
    parts = [int(x) for x in current.split(".")]
    
    if len(parts) != 3:
        raise ValueError(f"Version must be in X.Y.Z format, got: {current}")
    
    major, minor, patch = parts
    
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")
    
    return f"{major}.{minor}.{patch}"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    arg = sys.argv[1]
    
    # Check if it's a specific version
    if re.match(r'^\d+\.\d+\.\d+$', arg):
        new_version = arg
        current = read_current_version()
        print(f"Setting version: {current} → {new_version}")
    # Otherwise it's a bump type
    elif arg in ["major", "minor", "patch"]:
        current = read_current_version()
        new_version = bump_version(arg)
        print(f"Bumping {arg} version: {current} → {new_version}")
    else:
        print(f"Error: Invalid argument '{arg}'")
        print(__doc__)
        sys.exit(1)
    
    write_version(new_version)
    
    print("\n📋 Next steps:")
    print("1. Commit the version changes:")
    print(f"   git add -A && git commit -m 'Bump version to {new_version}'")
    print("2. Create a git tag:")
    print(f"   git tag -a v{new_version} -m 'Release version {new_version}'")
    print("3. Push changes and tag:")
    print("   git push && git push --tags")
    print("4. Create a GitHub release from the tag")


if __name__ == "__main__":
    main()