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

The project provides command-line tools to export data from Nearmap AI APIs:

#### Feature API Export
```bash
# Set up API key
export API_KEY=your_api_key_here

# Run the feature exporter
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

#### Roof Age API Export (US Only)
```bash
# Set up API key
export API_KEY=your_api_key_here

# Run the roof age exporter
python -m nmaipy.roof_age_exporter \
    --aoi-file "path/to/us_properties.geojson" \
    --output-dir "data/roof_age_outputs" \
    --country us \
    --threads 10 \
    --output-format both
```

## Code Architecture

### Core Components

1. **exporter.py**: Main command-line tool for exporting data from Nearmap AI Feature API
   - Uses parallel processing to handle large exports efficiently
   - Supports chunking to manage memory usage for large exports
   - Creates both rollup summary data and detailed feature exports

2. **roof_age_exporter.py**: Command-line tool for exporting data from Nearmap Roof Age API
   - Specialized exporter for roof age predictions (US only)
   - Parallel processing of multiple AOIs
   - Outputs roof geometries with installation dates and confidence scores
   - Follows similar patterns to exporter.py for consistency

3. **feature_api.py**: Client for interacting with Nearmap AI Feature API
   - Handles authentication and API requests
   - Provides caching to reduce API calls
   - Supports different API endpoints and versions
   - Uses shared infrastructure from api_common.py

4. **roof_age_api.py**: Client for interacting with Nearmap Roof Age API
   - Simpler API surface than Feature API (no packs, classes, system versions)
   - Supports both AOI and address-based queries
   - Returns GeoJSON with roof polygons and age predictions
   - Built on shared BaseApiClient from api_common.py

5. **api_common.py**: Shared infrastructure for all API clients
   - BaseApiClient with session management, caching, and retry logic
   - RetryRequest class with exponential backoff
   - APIKeyFilter for secure logging (removes API keys from logs)
   - Error handling classes (APIError, AIFeatureAPIError, RoofAgeAPIError)
   - Reusable across different Nearmap API products

6. **parcels.py**: Functions to process property boundaries and features
   - Reads parcel data from different file formats
   - Filters features within parcels
   - Creates summary statistics (rollups) for features within parcels

7. **constants.py**: Contains important constants used throughout the project
   - Feature class IDs
   - CRS definitions
   - Default filtering parameters
   - Roof Age API configuration and field names

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
python -m nmaipy.roof_age_exporter --api-key your_api_key_here [other arguments]
```

### API Client Architecture

The library uses a modular architecture with shared infrastructure:

#### Shared Components (api_common.py)
- **BaseApiClient**: Base class providing common functionality:
  - Session management with connection pooling
  - Request retry logic with exponential backoff
  - File-based caching (with optional gzip compression)
  - API key handling and secure logging

- **RetryRequest**: Configurable retry strategy for HTTP requests
  - Handles transient errors (429, 500, 502, 503, 504)
  - Exponential backoff with configurable min/max delays
  - Retries on connection errors and timeouts

- **Error Classes**: Hierarchical error handling
  - APIError (base class)
  - AIFeatureAPIError (for Feature API)
  - RoofAgeAPIError (for Roof Age API)

#### API-Specific Clients
- **FeatureApi**: Complex API with packs, classes, system versions
  - Supports bulk requests, address queries, gridding for large AOIs
  - Feature filtering and rollup calculations
  - Date range and survey resource queries

- **RoofAgeApi**: Simpler API focused on roof age predictions
  - AOI or address-based queries
  - Returns GeoJSON with roof polygons and installation dates
  - US-only currently, may expand to other regions

#### Design Principles
- **Code Reuse**: Common infrastructure factored into api_common.py
- **Separation of Concerns**: API-specific logic in separate modules
- **Consistency**: Similar patterns across different API clients
- **Extensibility**: Easy to add new API clients using BaseApiClient

## Version Management & Deployment

See `VERSIONING.md` for instructions on:
- Updating version numbers
- Building and testing packages
- Deploying to PyPI
- Creating releases