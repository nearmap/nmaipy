# Versioning Strategy for nmaipy

## Version Format

nmaipy follows [Semantic Versioning](https://semver.org/) (SemVer):

**MAJOR.MINOR.PATCH** (e.g., 3.2.1)

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backwards-compatible manner
- **PATCH**: Backwards-compatible bug fixes

## Current Version

The current version is stored in `nmaipy/__version__.py` and can be accessed:

```python
import nmaipy
print(nmaipy.__version__)  # '3.2.1'
print(nmaipy.__version_info__)  # (3, 2, 1)
```

Or from the command line:
```bash
python -m nmaipy.exporter --version
```

## Version Locations

The version number is maintained in these files:
1. `nmaipy/__version__.py` - Single source of truth
2. `conda.recipe/meta.yaml` - For conda packaging
3. Git tags - For releases (format: `v3.2.1`)

## Releasing a New Version

### 1. Update Version Number

Use the bump_version.py script:

```bash
# For bug fixes (3.2.1 -> 3.2.2)
python bump_version.py patch

# For new features (3.2.1 -> 3.3.0)
python bump_version.py minor

# For breaking changes (3.2.1 -> 4.0.0)
python bump_version.py major

# Or set specific version
python bump_version.py 3.2.5
```

### 2. Commit Changes

```bash
git add -A
git commit -m "Bump version to 3.2.2"
```

### 3. Create Git Tag

```bash
git tag -a v3.2.2 -m "Release version 3.2.2"
```

### 4. Push to GitHub

```bash
git push origin main
git push origin v3.2.2
```

### 5. Create GitHub Release

The GitHub Actions workflow will automatically:
- Create a GitHub release when a tag is pushed
- Build the package
- Verify version consistency

Alternatively, manually create a release on GitHub from the tag.

## Version History

- **3.2.1** (Current) - Added damage classification, improved gridding, security fixes
- **3.2.0** - Previous release
- ...

## Checking Version Compatibility

To check if your code is compatible with a minimum version:

```python
from nmaipy import __version_info__

# Require at least version 3.2.0
if __version_info__ < (3, 2, 0):
    raise RuntimeError("nmaipy 3.2.0 or higher required")
```

## Development Versions

During development between releases, the version in `main` branch should be:
- The last released version if no changes have been made
- The next anticipated version with a dev suffix (e.g., "3.2.2.dev0") if actively developing

## API Stability

- Functions/classes marked as public API (not starting with `_`) follow semantic versioning
- Internal functions (starting with `_`) may change without notice
- Deprecation warnings will be provided for at least one minor version before removal

## Questions?

For questions about versioning or releases, please open an issue on [GitHub](https://github.com/nearmap/nmaipy).