# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

To set up the environment for nmaipy, use conda with the provided environment.yaml file:

```bash
conda env create -f environment.yaml
conda activate nmaipy
```

Alternatively, you can install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

## Key Commands

### Running Tests

Run all tests:
```bash
pytest
```

Run a specific test file:
```bash
pytest tests/test_parcels.py
```

Run a specific test:
```bash
pytest tests/test_parcels.py::test_function_name
```

### Code Formatting

Format code with black:
```bash
black nmaipy tests
```

Sort imports with isort:
```bash
isort nmaipy tests
```

### Export Data

The project provides a command-line tool to export data from Nearmap AI APIs:

```bash
# Set up API key
export API_KEY=your_api_key_here

# Run the exporter
python nmaipy/exporter.py \
    --aoi-file "path/to/aoi.geojson" \
    --output-dir "data/outputs" \
    --country au \
    --processes 4 \
    --system-version-prefix "gen6-" \
    --packs building vegetation \
    --include-parcel-geometry \
    --save-features
```

## Code Architecture

### Core Components

1. **exporter.py**: Main command-line tool for exporting data from Nearmap AI APIs
   - Uses parallel processing to handle large exports efficiently
   - Supports chunking to manage memory usage for large exports
   - Creates both rollup summary data and detailed feature exports

2. **feature_api.py**: Client for interacting with Nearmap AI API endpoints
   - Handles authentication and API requests
   - Provides caching to reduce API calls
   - Supports different API endpoints and versions

3. **parcels.py**: Functions to process property boundaries and features
   - Reads parcel data from different file formats
   - Filters features within parcels 
   - Creates summary statistics (rollups) for features within parcels

4. **constants.py**: Contains important constants used throughout the project
   - Feature class IDs
   - CRS definitions
   - Default filtering parameters

### Data Flow

1. User provides a GeoJSON or CSV file with AOIs (Areas of Interest)
2. The exporter divides work into chunks for parallel processing
3. For each AOI, the feature API fetches relevant AI features
4. These features are filtered based on intersection with AOI boundaries
5. For large AOIs, the system can grid them and merge results
6. Results are summarized into rollup statistics and/or saved as feature data
7. Output is saved in CSV or Parquet format for further analysis

### API Authentication

API access requires a Nearmap API key which should be set as an environment variable:
```bash
export API_KEY=your_api_key_here
```

Alternatively, the API key can be provided as a command-line argument:
```bash
python nmaipy/exporter.py --api-key your_api_key_here [other arguments]
```

## Publishing to PyPI

When preparing a new release of nmaipy for PyPI:

### 1. Pre-release Checklist
- Update version in `nmaipy/__version__.py`
- Ensure all tests pass: `pytest`
- Update CHANGELOG if one exists
- Commit all changes

### 2. Build and Test
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
python -m build

# Verify the build
twine check dist/*
```

### 3. Test on TestPyPI (Optional but Recommended)
```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nmaipy
```

### 4. Publish to PyPI
```bash
# Upload to production PyPI
twine upload dist/*

# Verify installation works
pip install nmaipy
python -c "import nmaipy; print(f'nmaipy {nmaipy.__version__} installed successfully')"
```

### 5. Post-release Tasks
```bash
# Tag the release
git tag -a v<VERSION> -m "Release version <VERSION>"
git push origin v<VERSION>

# Create GitHub release with changelog
# Update README if needed
```

### Required Tools
```bash
pip install build twine
```

### Authentication
Configure `~/.pypirc` with PyPI API tokens (see PUBLISHING.md for details).