#!/usr/bin/env python
"""Test that all required dependencies are listed in setup.py."""

import ast
import re
from pathlib import Path

# Read setup.py to get listed requirements
setup_path = Path("setup.py")
with open(setup_path) as f:
    setup_content = f.read()

# Extract required_packages list
required_match = re.search(r'required_packages\s*=\s*\[(.*?)\]', setup_content, re.DOTALL)
if required_match:
    required_str = required_match.group(1)
    required_packages = [pkg.strip().strip('"').strip("'").split('[')[0].split('>')[0].split('=')[0].split('<')[0] 
                        for pkg in required_str.split(',') if pkg.strip()]
    print("Dependencies listed in setup.py:")
    for pkg in sorted(required_packages):
        print(f"  - {pkg}")
else:
    print("Could not parse required_packages from setup.py")
    required_packages = []

print("\n" + "="*50)

# Find all imports in the package
imports = set()
package_dir = Path("nmaipy")
for py_file in package_dir.glob("*.py"):
    with open(py_file) as f:
        content = f.read()
    
    # Find import statements
    import_lines = re.findall(r'^import\s+(\w+)', content, re.MULTILINE)
    from_lines = re.findall(r'^from\s+(\w+)', content, re.MULTILINE)
    
    imports.update(import_lines)
    imports.update(from_lines)

# Standard library modules that don't need to be in setup.py
stdlib_modules = {
    'argparse', 'ast', 'atexit', 'collections', 'concurrent', 'contextlib', 
    'copy', 'datetime', 'enum', 'functools', 'gc', 'gzip', 'hashlib', 'http', 
    'io', 'json', 'logging', 'math', 'os', 'pathlib', 're', 'signal', 'ssl',
    'subprocess', 'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 
    'urllib', 'urllib3', 'uuid', 'warnings', 'weakref', 'zipfile'
}

# Filter out stdlib and internal imports
external_imports = {imp for imp in imports if imp not in stdlib_modules and not imp.startswith('_')}
external_imports.discard('nmaipy')  # Remove self-imports

# Map import names to package names
import_to_package = {
    'dotenv': 'python-dotenv',
}
# Convert imports to package names
external_packages = set()
for imp in external_imports:
    external_packages.add(import_to_package.get(imp, imp))

print("\nExternal packages imported in code:")
for imp in sorted(external_imports):
    print(f"  - {imp}")

print("\n" + "="*50)

# Find missing dependencies
missing = external_packages - set(required_packages)
if missing:
    print("\n⚠️  MISSING from setup.py:")
    for pkg in sorted(missing):
        print(f"  - {pkg}")
else:
    print("\n✅ All imported packages are listed in setup.py")

# Find potentially unused dependencies
imported_base = {pkg.lower() for pkg in external_imports}
required_base = {pkg.lower() for pkg in required_packages}
potentially_unused = required_base - imported_base

if potentially_unused:
    print("\n📦 Potentially unused in setup.py (might be indirect deps):")
    for pkg in sorted(potentially_unused):
        print(f"  - {pkg}")