# Version Management & Release

## Recommended: Use the `/publish` Skill

If you have Claude Code, run the `/publish` skill for an automated, guided release:

```
/publish
```

This handles all steps: version validation, tests, build, PyPI upload, tagging, and GitHub release creation.

---

## Manual Release Process

For users without Claude Code, or as a reference for what the skill does.

### Prerequisites

Install build tools:
```bash
pip install build twine
```

Configure PyPI authentication in `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-<your-token-here>
```

Get tokens from: https://pypi.org/manage/account/token/

### Release Steps

#### 1. Update Version Number

Edit `nmaipy/__version__.py`:
```python
__version__ = "4.0.1"  # Increment following semver
```

**Semantic Versioning:**
- MAJOR: Breaking API changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes, backwards compatible

**Pre-release Versions (PEP 440):**
- Alpha: `4.0.0a1` - early testing, API may change
- Beta: `4.0.0b1` - feature complete, testing for bugs
- Release Candidate: `4.0.0rc1` - final testing

#### 2. Commit Version Change

```bash
git add nmaipy/__version__.py
git commit -m "Release version X.Y.Z"
git push origin main
```

#### 3. Run Tests

```bash
pytest
```

All tests must pass before proceeding.

#### 4. Build Package

```bash
rm -rf dist/ build/ *.egg-info
python -m build
twine check dist/*
```

#### 5. Upload to PyPI

```bash
twine upload dist/*
```

Verify:
```bash
pip install --upgrade nmaipy
python -c "import nmaipy; print(nmaipy.__version__)"
```

#### 6. Create and Push Tag

```bash
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
```

#### 7. Create GitHub Release

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --generate-notes
```

For pre-releases (versions with 'a', 'b', or 'rc'), add `--prerelease`:
```bash
gh release create vX.Y.Z --title "vX.Y.Z" --generate-notes --prerelease
```

---

## What Happens After Release

When you push a version tag, the GitHub Actions workflow automatically:
1. Runs the test suite (CI validation)
2. Verifies the tag matches the package version
3. Builds the package
4. Attaches build artifacts (.whl, .tar.gz) to the GitHub release
5. Verifies the package is available on PyPI

This provides an additional validation layer but does not publish to PyPI (that's done locally in step 5 above).

---

## Testing on TestPyPI (Optional)

Before production release:
```bash
twine upload --repository testpypi dist/*

pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ nmaipy
```

## Troubleshooting

- **403 Forbidden**: Check API token is correct and starts with `pypi-`
- **Version exists**: Increment version number, can't reuse versions
- **Build issues**: Ensure `MANIFEST.in` includes necessary files
- **Import errors**: Check all dependencies are listed in `pyproject.toml`
