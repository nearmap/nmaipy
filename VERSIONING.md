# Version Management & PyPI Deployment

Quick guide for updating nmaipy versions and deploying to PyPI.

## Version Update Process

### 1. Update Version Number
Edit `nmaipy/__version__.py`:
```python
__version__ = "4.0.1"  # Increment following semver
```

**Semantic Versioning:**
- MAJOR.MINOR.PATCH (e.g., 4.0.1)
- MAJOR: Breaking API changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes, backwards compatible

**Pre-release Versions (PEP 440):**
- Alpha: `4.0.0a1`, `4.0.0a2` - early testing, API may change
- Beta: `4.0.0b1` - feature complete, testing for bugs
- Release Candidate: `4.0.0rc1` - final testing before release

Pre-release versions are not installed by default with `pip install nmaipy`. Users must explicitly request them:
```bash
pip install nmaipy==4.0.0a2  # Specific version
pip install --pre nmaipy     # Latest pre-release
```

### 2. Run Tests
```bash
pytest
```

### 3. Build Package
```bash
# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build
python -m build

# Verify
twine check dist/*
```

### 4. Deploy to PyPI
```bash
# Upload to PyPI
twine upload dist/*

# Quick test
pip install --upgrade nmaipy
python -c "import nmaipy; print(nmaipy.__version__)"
```

### 5. Git Tag & Push
```bash
# Commit version change
git add nmaipy/__version__.py
git commit -m "Release version 4.0.1"

# Tag
git tag -a v4.0.1 -m "Release version 4.0.1"

# Push (use your branch name - main for stable, feature branch for pre-releases)
git push origin main
git push origin v4.0.1
```

### 6. Create GitHub Release
1. Go to https://github.com/nearmap/nmaipy/releases
2. Click "Create a new release"
3. Choose tag v4.0.1
4. Add release notes
5. For pre-releases, check "Set as a pre-release" to prevent it showing as "Latest"
6. Publish release

## Prerequisites

### Install Tools
```bash
pip install build twine
```

### Configure PyPI Authentication
Create `~/.pypirc`:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-<your-token-here>

[testpypi]
username = __token__
password = pypi-<your-test-token>
repository = https://test.pypi.org/legacy/
```

Get tokens from: https://pypi.org/manage/account/token/

## Quick Deploy Script

For convenience, you can run all steps:
```bash
# Update version in nmaipy/__version__.py first, then:
pytest && \
rm -rf dist/ build/ *.egg-info && \
python -m build && \
twine check dist/* && \
twine upload dist/* && \
git add nmaipy/__version__.py && \
git commit -m "Release version $(python -c 'from nmaipy import __version__; print(__version__)')" && \
git tag -a v$(python -c 'from nmaipy import __version__; print(__version__)') -m "Release version $(python -c 'from nmaipy import __version__; print(__version__)')" && \
git push origin main && \
git push origin v$(python -c 'from nmaipy import __version__; print(__version__)')
```

## Testing on TestPyPI (Optional)

Before production release:
```bash
# Upload to test
twine upload --repository testpypi dist/*

# Install from test
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nmaipy
```

## Troubleshooting

- **403 Forbidden**: Check API token is correct and starts with `pypi-`
- **Version exists**: Increment version number, can't reuse versions
- **Build issues**: Ensure `MANIFEST.in` includes necessary files
- **Import errors**: Check all dependencies are listed in `pyproject.toml`