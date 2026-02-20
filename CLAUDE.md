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

Run all tests (excluding live API tests):
```bash
pytest -m "not live_api"
```

Run all tests including live API (requires `API_KEY` env var):
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

#### Test Markers

`pytest.ini` defines four markers:
- `live_api` — requires live Nearmap API connection and `API_KEY` env var
- `integration` — integration tests (may be slow)
- `unit` — unit tests
- `slow` — long-running tests

Deselect markers with `-m`, e.g.: `pytest -m "not live_api and not slow"`

### Code Formatting

Format code with black:
```bash
black nmaipy tests
```

Sort imports with isort:
```bash
isort nmaipy tests
```

Note: `ruff` is also available in the conda environment for linting.

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

#### Unified Feature + Roof Age Export (US Only)
```bash
python nmaipy/exporter.py \
    --aoi-file "path/to/us_properties.geojson" \
    --output-dir "data/outputs" \
    --country us \
    --packs building \
    --roof-age \
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
    --processes 4 \
    --output-format both
```

#### S3 Output (v4.2+)
```bash
python nmaipy/exporter.py \
    --aoi-file "path/to/aoi.geojson" \
    --output-dir "s3://my-bucket/results/" \
    --country us \
    --packs building
```

AWS credentials are resolved from environment variables or `~/.aws/credentials`.

## Code Architecture

### Exporter Layer (user-facing entry points)

1. **exporter.py**: Main command-line tool and `NearmapAIExporter` class
   - Unified exporter for Feature API and Roof Age API
   - Uses parallel processing to handle large exports efficiently
   - Supports chunking to manage memory usage for large exports
   - Creates rollup summary data, per-class feature files, and combined GeoParquet
   - `AOIExporter` is a backward-compatible alias for `NearmapAIExporter`

2. **base_exporter.py**: `BaseExporter` abstract base class for all exporters
   - Defines common interface: `process_chunk()` and `get_chunk_output_file()` abstract methods
   - Chunk splitting with `split_into_chunks()` and cache-aware resume
   - `ProcessPoolExecutor` parallel processing with API warmup ramp-up
   - `BrokenProcessPool` retry (up to 3 attempts) with diagnostic logging (memory/CPU/FDs)
   - Shared progress counters via `multiprocessing.Manager` with dynamic tqdm display
   - S3 local staging directory management (creates `tempfile.mkdtemp()` when output is S3)
   - `_save_config()` for writing `export_config.json` to output

3. **roof_age_exporter.py**: `RoofAgeExporter` class and standalone CLI
   - Specialized exporter for roof age predictions (US only)
   - Inherits from `BaseExporter`
   - Outputs roof geometries with installation dates and confidence scores

### API Client Layer

4. **api_common.py**: Shared infrastructure for all API clients
   - `BaseApiClient`: Session management with connection pooling, request retry via `urllib3.Retry`, file-based caching (with optional gzip compression), API key handling, secure logging via `APIKeyFilter`, latency histogram tracking per-request
   - `GriddedApiClient(BaseApiClient)`: Extends with automatic AOI gridding for large areas, semaphore-based concurrency control, `should_grid_aoi()` size check
   - `RetryRequest(Retry)`: Configurable retry strategy; handles 429/500/502/503/504, exponential backoff (2-10s)
   - Error hierarchy: `APIError` (base) → `AIFeatureAPIError`, `RoofAgeAPIError`, `APIRequestSizeError` (413/504 triggers gridding); `APIGridError` for gridding failures
   - Latency stats: `collect_latency_stats_from_apis()`, `combine_chunk_latency_stats()`, `compute_global_latency_stats()` — histogram-based P50/P90/P95/P99 aggregation across chunks

5. **feature_api.py**: `FeatureApi(GriddedApiClient)` — client for Nearmap AI Feature API
   - Pack and class enumeration (`get_packs()`, `get_feature_classes()`)
   - Bulk feature requests with auto-gridding on oversized AOIs
   - Rollup calculations (`get_rollup_df()`, `get_rollup_df_bulk()`)
   - Feature geometry requests (`get_features_gdf()`, `get_features_gdf_bulk()`)
   - Date range and survey resource queries

6. **roof_age_api.py**: `RoofAgeApi(BaseApiClient)` — client for Nearmap Roof Age API
   - AOI-based queries (`get_roof_age_by_aoi()`) with pagination
   - Address-based queries (`get_roof_age_by_address()`)
   - Bulk processing (`get_roof_age_bulk()`) with thread pool
   - Returns GeoDataFrame with roof polygons and age predictions
   - US-only currently

### Data Processing Layer

7. **parcels.py**: Parcel rollup and feature linking
   - `parcel_rollup()`: Main rollup function — aggregates all features within each AOI into summary statistics with primary feature selection
   - `link_roofs_to_buildings()`: Spatial association of roofs to parent buildings
   - `link_roof_instances_to_roofs()`: Links Roof Age API results to Feature API roof geometries by IoU
   - `calculate_child_feature_attributes()`: Computes attributes for child features (e.g. roof characteristics on a roof)
   - Re-exports `read_from_file` from `aoi_io` for backward compatibility

8. **primary_feature_selection.py**: Primary feature selection algorithms
   - `select_primary()`: Dispatcher for selection method
   - `select_primary_by_largest()`: Selects feature with largest area in parcel
   - `select_primary_by_nearest()`: Selects feature nearest to target point, prefers non-small features
   - `select_primary_optimal()`: Tries nearest first, falls back to largest

9. **feature_attributes.py**: Attribute flattening utilities
   - `flatten_building_attributes()`: Flattens nested building API attributes to flat dict
   - `flatten_roof_attributes()`: Flattens roof attributes including materials, spotlight index
   - `flatten_building_lifecycle_damage_attributes()`: Flattens damage classification scores
   - `flatten_roof_instance_attributes()`: Flattens Roof Age API result attributes
   - `calculate_roof_age_years()`: Computes roof age in years from installation/as-of dates

10. **geometry_utils.py**: Geometry processing utilities
    - `create_grid()` / `split_geometry_into_grid()`: Grid generation for large AOIs
    - `clip_features_to_polygon()`: Clips features to AOI boundary, recalculates areas
    - `combine_features_from_grid()`: Deduplicates and merges features from gridded queries
    - `polygon_to_coordstring()`: Converts Shapely polygon to API coordinate string

### Infrastructure Layer

11. **storage.py**: S3/local filesystem abstraction (new in v4.2)
    - Unified API for both local and `s3://` paths via `fsspec`/`s3fs`
    - Fork-safe: uses per-PID `fsspec` filesystem instances with `skip_instance_cache=True` to avoid sharing botocore connections across forked worker processes
    - Key functions: `is_s3_path()`, `join_path()`, `ensure_directory()`, `open_file()`, `write_parquet()`, `read_json()`, `write_json()`, `glob_files()`, `upload_file()`, `file_exists()`, `remove_file()`
    - `write_parquet()` routes S3 writes through `open_file()` rather than passing S3 URIs to `df.to_parquet()`, because pyarrow's native S3FileSystem is not fork-safe

12. **aoi_io.py**: AOI file reading and format handling
    - `read_from_file()`: Reads GeoJSON, CSV/TSV/PSV (WKT geometry), GeoPackage, Parquet/GeoParquet
    - Handles CRS transformations and validates unique AOI IDs
    - Generates sequential IDs if `aoi_id` column is missing

13. **constants.py**: Project-wide constants
    - Feature class UUIDs (buildings, roofs, vegetation, surfaces, pools, damage, etc.)
    - CRS definitions per country (`AREA_CRS` dict for au/ca/nz/us)
    - API configuration: `DEFAULT_URL_ROOT`, `ROOF_AGE_URL_ROOT`, timeouts, retry limits
    - Column name constants: `AOI_ID_COLUMN_NAME`, date columns, survey resource columns
    - Gridding config: `GRID_SIZE_DEGREES`, `MAX_AOI_AREA_SQM_BEFORE_GRIDDING`

14. **log.py**: Logger configuration
    - `get_logger()` / `configure_logger()`: Shared `nmaipy` logger with `APIKeyFilter`

15. **cgroup_memory.py**: Container resource introspection
    - Reads cgroup v1/v2 memory and CPU limits
    - `get_memory_info_cgroup_aware()` / `get_cpu_info_cgroup_aware()`: Prefers cgroup over psutil
    - Used by `BaseExporter` for progress display and `BrokenProcessPool` crash diagnostics

### Output Layer

16. **readme_generator.py**: `ReadmeGenerator` class
    - Auto-generates a data dictionary `README.md` in export output directories
    - Discovers files present, generates tables of file descriptions and column patterns
    - Reads `export_config.json` to detect country (for area units) and selection method
    - Works with both local and S3 output directories

### Legacy/Internal

17. **coverage_utils.py**: Legacy coverage API utilities
    - Survey coverage point queries and threading
    - Not integrated with current `BaseApiClient` architecture (uses its own `requests.Session`)

18. **reference_code.py**: Minimal utility
    - `is_building_small()`: Checks building area < 30 sqm

### Configuration Files

- **config.json** (repo root): Per-feature-class filtering thresholds, keyed by class UUID. Fields: `min_size` (m²), `min_confidence`, `min_fidelity`, `min_area_in_parcel` (m²), `min_ratio_in_parcel`. Applied during parcel rollup processing in `parcels.py`.
- **pyproject.toml**: Package metadata, dependencies, build config. Python >=3.12 required.
- **environment.yaml** / **environment-minimal.yaml**: Conda environment specifications.

### Data Flow

1. User provides a GeoJSON, CSV, GeoPackage, or Parquet file with AOIs (Areas of Interest)
2. The exporter loads the AOI file via `aoi_io.read_from_file()` and divides work into chunks
3. Chunks are processed in parallel via `ProcessPoolExecutor` with API warmup ramp-up
4. For each AOI, the Feature API (and optionally Roof Age API) fetches relevant AI features
5. Features are filtered based on intersection with AOI boundaries and `config.json` thresholds
6. For large AOIs (>1 sq km), the system automatically grids them and merges/deduplicates results
7. Results are summarized into rollup statistics and/or saved as feature data per chunk
8. Chunks are merged into final output files (rollup CSV/Parquet, per-class files, GeoParquet)
9. For S3 output, per-process local staging is used for formats requiring local I/O before uploading
10. A `README.md` data dictionary is auto-generated in `final/` by `ReadmeGenerator`

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

For S3 output, AWS credentials are resolved from environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) or `~/.aws/credentials`.

## Version Management & Deployment

See `VERSIONING.md` for instructions on:
- Updating version numbers
- Building and testing packages
- Deploying to PyPI
- Creating releases
