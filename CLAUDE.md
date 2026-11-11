# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

The development environment is managed by [pixi](https://pixi.sh) from the `[tool.pixi.*]` tables in `pyproject.toml`, locked in `pixi.lock`. Every package, including nmaipy itself (built by pixi-build from `[project.dependencies]`), comes from conda-forge; nothing is installed from PyPI. The old `nmaipy` conda environment is no longer used.

From a fresh clone:

```bash
conda deactivate                                # if the old nmaipy conda env is active
curl -fsSL https://pixi.sh/install.sh | sh      # install pixi (0.83 or newer); then open a new shell
pixi install                                    # create the default environment in .pixi/
pixi run test                                   # run the offline test suite
```

On Windows, install pixi with `powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"`. Once nothing else needs it, the old environment can be removed with `conda env remove -n nmaipy`.

Run any command inside an environment with `pixi run <cmd>` (e.g. `pixi run python nmaipy/exporter.py ...`), or start a shell with it activated with `pixi shell`. Pick a non-default environment with `-e <env>`.

Environments:
- `default` — Python 3.14 with test, build (`python-build`, `twine`) and notebook (`ipykernel`, `matplotlib-base`) tools
- `py312`, `py313`, `py314` — the test environment on each supported Python version
- `lint` — `ruff` only
- `typecheck` — `ty`, with the test and notebook dependencies it reads
- `docs` — `mystmd`, for the documentation site

## Key Commands

### Running Tests

Run all tests (excluding live API tests; the `test` task runs `pytest -m "not live_api"`):
```bash
pixi run test
```

Run them on a specific Python version:
```bash
pixi run -e py312 test
```

Run all tests including live API (requires `API_KEY` env var):
```bash
pixi run pytest
```

Run a specific test file:
```bash
pixi run pytest tests/test_parcels.py
```

Run a specific test:
```bash
pixi run pytest tests/test_parcels.py::test_function_name
```

#### End-to-End Testing

End-to-end tests against the live API are a critical way to validate new features and bug fixes — unit tests alone are not sufficient for catching issues in the full data pipeline. Sample datasets for this purpose live in `tests/data/`:

- `tests/data/test_parcels_2.csv` — ~100 NJ residential parcels, good general-purpose US test set
- `tests/data/test_parcels.csv` — ~19 Sydney parcels, good general-purpose AU test set

To run an end-to-end test (requires `API_KEY`), choose parameters appropriate to the feature being tested — packs, flags, country, etc. will vary by context. Example:
```bash
pixi run python nmaipy/exporter.py \
    --aoi-file tests/data/test_parcels_2.csv \
    --output-dir data/my_test_output \
    --country us \
    --processes 4 \
    --packs building \
    --save-features
```

Output directly to a subfolder under `data/` so files can be reviewed in the IDE. The `data/` directory is gitignored, so generated outputs won't be committed. Check `data/my_test_output/final/` for rollup and per-class files.

#### Test Markers

`pytest.ini` defines four markers:
- `live_api` — requires live Nearmap API connection and `API_KEY` env var
- `integration` — integration tests (may be slow)
- `unit` — unit tests
- `slow` — long-running tests

Deselect markers with `-m`, e.g.: `pixi run pytest -m "not live_api and not slow"`

#### Golden fixtures

`tests/test_golden_exports.py` runs the full exporter offline against an in-process synthetic fake of the Feature and Roof Age APIs, compares every file in `final/` by value against stored outputs, and pins the `export_config.json` keys and the API cache layout. See `tests/data/golden/README.md`.
```bash
python -m tests.golden.regenerate          # re-baseline expected outputs after an intended change (offline)
python -m tests.golden.drift               # check the fake against the live API (needs API_KEY)
python -m tests.golden.compare LEFT RIGHT  # diff two exports (e.g. old versus new code on the same input)
```

### Code Formatting

Fix lint findings that ruff can fix safely, then format and sort imports with ruff:
```bash
pixi run format
```

Check without changing files. This runs ty first, then ruff. The Lint workflow runs the same check on every pull request and on pushes to main:
```bash
pixi run lint
```

Suppress a finding only on its line, always with the code and a reason: `# noqa: <code> -- <reason>` for ruff, `# ty: ignore[<rule>] -- <reason>` for ty.

The commits that only reformat the code or apply ruff's syntax-only fixes are listed in `.git-blame-ignore-revs`. Run `git config blame.ignoreRevsFile .git-blame-ignore-revs` once so that `git blame` skips them.

### Documentation Site

The documentation site is built with MyST from `docs/` (`docs/index.md` includes `README.md`; the theme overrides live in `docs/_static/custom.css`). Build it into `docs/_build/html`:
```bash
pixi run docs
```

Preview it with `python -m http.server -d docs/_build/html`. The `Docs` workflow (`.github/workflows/docs.yml`) builds the site on pull requests and deploys it to GitHub Pages from `main` when the repository variable `PAGES_ENABLED` is `true`.

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

#### Damage Conflation API Export
```bash
# Set up API key
export API_KEY=your_api_key_here

# Run the damage conflation exporter for a catastrophe event.
# Always emits per-building damage polygons; add --rollup for a per-AOI summary
# (one row per AOI: your input columns, rating counts + the primary building's
# attributes — most useful when AOIs are property-sized; add --parcel-mode there
# to drop neighbours' buildings that barely clip the parcel). Large AOIs are handled by pagination,
# not gridding, so an AOI up to a full event boundary works in one go.
python -m nmaipy.damage_conflation_exporter \
    --aoi-file "path/to/aoi.geojson" \
    --output-dir "data/damage_outputs" \
    --event-id "2f510853-5d55-50f4-9102-2c02de08190e" \
    --country us \
    --processes 4 \
    --output-format both \
    --rollup
```

The `--event-id` is the catastrophe event's UUID (obtained from the Coverage API
`eventId` survey tag; auto-discovery is not yet built into nmaipy). Output in `final/`:
`damage_buildings.{parquet,csv}` (per-building, always) and, with `--rollup`,
`damage_rollup.{parquet,csv}` (per-AOI); plus operational files `damage_metadata.csv`
(per-AOI query metadata) and `damage_errors.csv` (per-AOI failures: status_code + message,
written only when there are errors).

#### S3 Output (v4.2+)
```bash
python nmaipy/exporter.py \
    --aoi-file "path/to/aoi.geojson" \
    --output-dir "s3://my-bucket/results/" \
    --country us \
    --packs building
```

AWS credentials are resolved from environment variables or `~/.aws/credentials`.

## Pull Requests, Commits and Review Comments

nmaipy is a public, standalone package. Everything that becomes part of the repository or its
history (code, comments, docstrings, log strings, test data, commit messages, PR titles and
descriptions, review comments, release notes) must stand on its own:

- Do not reference specific applications, pipelines, deployment tooling or other harnesses that
  consume nmaipy, by name or by description. State the need in terms of nmaipy's own API
  ("callers of `AOIExporter` need to widen the retry budget"), not who the caller is.
- Do not include internal ticket ids, project or dataset names, customer or partner names,
  AOI/parcel counts or regions from a known workload, storage paths, or anything else that
  identifies an internal or commercial use of the package. Describe the issue generically
  ("a large export with spatially sorted input over a coverage gap").
- Tests use synthetic or bundled example data, never excerpts of production outputs.
- When editing an existing PR's description or replying to a review, the same rules apply.

If a change only makes sense in the context of a downstream system, explain it in terms of
nmaipy's behaviour and leave the downstream motivation out.

## Column Naming Conventions

All output column names in nmaipy are **programmatically derived** from the API's `description` field using `.lower().replace(" ", "_")`. This applies everywhere: per-component columns, dominant summary columns, rollup prefixes, etc.

**Do not introduce hardcoded friendly name mappings** (e.g. `"Roof material" → "material"`, `"Roof types" → "shape"`). If the API description is `"Roof material"`, the column prefix is `roof_material_`. If it's `"Roof types"`, the prefix is `roof_types_`. Friendly/display renames are a downstream application concern, not nmaipy's.

This consistency matters because:
- It keeps the mapping from API response to output columns predictable and auditable
- New API descriptions automatically get correct column names without code changes
- The `flatten_*_attributes()` functions, `readme_generator.py`, and `exporter.py` column partitioning all rely on this convention

When adding new columns derived from API descriptions, always use the programmatic pattern rather than inventing abbreviated or "cleaned" names.

## Feature ID Semantics (multiparcel row identity)

`feature_id` values originate in the raw vector tiles, and features can be large — a single roof, building, or building lifecycle polygon can span parcel boundaries or grid cells. Their stability rules:

- **Stable within a survey**: the same physical feature from the same survey ID has the same `feature_id` no matter which AOI or query returned it. This is what makes gridding deduplication (`combine_features_from_grid`) and cross-AOI matching work.
- **Not stable across surveys/dates**: the same physical feature captured on different survey dates gets different IDs. Never match or deduplicate by `feature_id` across dates.
- **Multiparcel consequence (parcelMode)**: two AOIs that intersect the same feature from the same survey each get a row with the *same* `feature_id`, but per-parcel values differ between the rows. The API returns that parcel's `clippedGeometry` / `clippedArea*` and computes the includes (RSI, hurricane score, defensible space) on **that parcel's clip**. Its `attributes` (component areas and ratios, …) stay whole-feature, so nmaipy's flattening recomputes them per clip from child features (see "parcelMode response semantics" below). The row identity in any multi-AOI frame is therefore `(aoi_id, feature_id)`, never `feature_id` alone.

Code implications (see PR #210 for the bug class this prevents):
- Any chunk-level (multi-AOI) cross-row structure must be keyed by `(aoi_id, feature_id)` (e.g. `roof_attrs_cache`) or scoped per AOI as `{aoi_id: {feature_id: …}}` (`_build_parent_lookup_by_aoi`, `roof_to_building_lookup`), with each consumer taking its row's own parcel sub-dict.
- `build_parent_lookup` (flat, fid-keyed) is only valid for a single AOI's features.
- Parent-chain traversal and IoU linkage never cross parcels — per-AOI scoping preserves that invariant by construction, and a chain that is broken within a parcel must yield null rather than borrow another parcel's clip value.
- Deduplication by bare `feature_id` is only correct within one AOI and one survey (the gridded-query merge case).

## parcelMode response semantics

Verified against the live Feature API by querying the bundled `tests/data/test_parcels_2.csv` parcels, plus synthetic halves of some of them split through a roof, with `parcelMode=true` and `false` (`building`, `roof_char`, `vegetation` and `surfaces` packs; includes `roofSpotlightIndex,defensibleSpace,hurricaneScore`):

- **Discrete versus connected classes. This holds in both modes, unchanged by parcelMode.**
  - **Discrete classes** (roof, building, roof components, driveway, …) are whole objects that can extend past the AOI. `geometry` is the whole outline, and `areaSqm` == `unclippedAreaSqm`, the whole object's area.
  - **Connected classes** (vegetation, surfaces, roads, …; most classes) have no whole-object extent. `geometry` is already clipped to the AOI and `areaSqm` == `clippedAreaSqm`. `unclippedAreaSqm` is the whole connected patch, extending beyond the AOI.
  - For both kinds, `clippedAreaSqm` is the area inside the query AOI.
  - nmaipy uses no list of either kind, since the API's classes change. Code that needs the kind reads it from these areas. (`CONNECTED_CLASS_IDS` is deprecated until 6.0, served by `constants.__getattr__` with a warning.)
  - For gridded AOIs, every cell repeats a discrete object's whole area but reports only its own piece of a connected one. `combine_features_from_grid` therefore keeps one copy of the former and sums the latter. It tells them apart by whether a feature's `areaSqm` equals its `clippedAreaSqm` in every cell, never by `unclippedAreaSqm` (a connected patch's is the whole forest or lawn, however little of it is in the AOI).
- **parcelMode adds or removes no features.** For the same AOI both modes return exactly the same set (43,885 features over 485 random lots in 12 US and AU metros, plus the bundled parcels); it only flags and clips them. For some classes, `belongsToParcel` is `false` for a feature below the class's (unpublished) minimum confidence even when it lies wholly inside the parcel, so `payload_gdf` drops it in parcelMode while parcelMode-off and gridded results keep it. Some classes (e.g. Pole) come back with no flag, and are kept with `belongs_to_parcel` null.
- **What parcelMode adds**:
  - `belongsToParcel`: whether the feature meaningfully belongs within the parcel, as assessed by the API's geospatial algorithms. A feature judged not to belong, typically a neighbour's roof that appears to cross the boundary because of parcel/imagery offset or parallax, comes back `false`. It is not a simple share-of-area threshold: a roof split down the middle belongs to both halves. Connected features are nearly all `true`; the Lawn Grass patches that came back `false` were low-confidence ones. `payload_gdf` drops `false` features, so `belongs_to_parcel` is true on every row answered with parcelMode on (null for unflagged classes such as Pole), and null for gridded AOIs' rows (grid cells run with parcelMode off) and when parcelMode is off.
  - `clippedGeometry`, only for discrete features the API judges to be **multiparcel** (genuinely shared with another parcel). Connected features never get one, because their geometry is already clipped. `payload_gdf` swaps it into `geometry` and sets `multiparcel_feature`.
  - A belonging feature that only slightly overhangs the boundary gets no `clippedGeometry`. It is returned and analysed whole, though its `clippedAreaSqm` is still the in-parcel area.
- **`attributes` are never recomputed for the clip.** A feature's `attributes` (roof material, shape and overhang components with their `areaSqm`, `ratio` and `confidence`; pitch; building attributes) are identical in both modes and in every parcel that returns the feature. A component `ratio` is component area divided by whole-feature area.
- **Includes are per-parcel for multiparcel features.** `roofSpotlightIndex` (value and confidence), `hurricaneScore` and `defensibleSpace` differ between the parcels sharing a multiparcel roof, and differ from the parcelMode-off response. For other roofs they are whole-roof and match parcelMode off. For example, `hurricaneScore.modelInputFeatures.treeOverhangRatio` is the clip's ratio for a multiparcel roof and the whole roof's ratio otherwise.
- **A returned roof brings all its children, whole.** Every feature whose `parentId` is the roof (material, shape, Tree Overhang, …) is returned with its full `geometry`, even if it lies entirely outside the AOI. This happens in both modes, and every parcel returning the roof gets the same child set. parcelMode only flags them:
  - children of a multiparcel roof that lie wholly outside the parcel are `belongsToParcel=false` with `clippedAreaSqm=0`, so `payload_gdf` drops them;
  - children of a multiparcel roof that cross the boundary get `clippedGeometry`;
  - children of a slight-overhang roof all belong and stay whole;
  - children of a non-belonging roof are all `false`.

How nmaipy uses this: `flatten_roof_attributes` recomputes components through `calculate_child_feature_attributes` whenever `clipped_area < 0.99 × unclipped_area`.
- Each component's `_present`, `_area_*`, `_ratio` and `_confidence` is recomputed as Σ area(child ∩ stored roof geometry) / area(stored roof geometry), in the equal-area CRS. It uses that AOI's children, matched by class id.
- For a multiparcel roof the stored geometry is the clip, so the result is the per-parcel value. It matches the API's own per-parcel `treeOverhangRatio` in `hurricaneScore` exactly.
- For a slight-overhang roof the stored geometry is the whole outline, so the result reproduces the whole-roof value.
- Gridded AOIs run with parcelMode off, so the stored outline is clipped to the AOI first: by `_clip_roof_geoms_to_aoi` for the per-class files and inline in `parcel_rollup` (`clip_roof_to_parcel`) for the rollup. The includes stay whole-roof there, so the recomputed ratio no longer matches them.
- The Roof Age link (`exporter._link_roof_instances`) compares a gridded AOI's roofs by that same clipped outline, since Roof Age instances are cut per lot. The exported roof keeps its whole outline.
- Grid cells run with parcelMode off by design. Gridding is meant for very large areas, where discrete objects crossing the outer boundary matter little, while parcelMode would treat each cell as a parcel and cut features at the internal grid lines. An AOI whose realtime scores the API skips is first requested once more, unchanged, after `REALTIME_SKIP_RETRY_DELAY_SECONDS` (the retry bypasses the cache, which holds the skipped response). If it still skips and spans more than one `REALTIME_SKIP_GRID_SIZE_DEGREES` cell, it is gridded at that size, since each cell is a request the scorer can handle (the large-parcel, many-roofs case). Both steps are `retry_on_realtime_skip` / `--no-retry-on-realtime-skip`; `regrid_on_skip` is a deprecated alias. Those cells are pinned to the first response's survey; if the grid fails, the AOI keeps its response (skipped scores null), and a successful grid keeps its parcel-level aggregate.
- Includes are passed through unchanged.

## Customer-facing knowledge skill

`.agents/skills/nearmap-ai-apis/` is a Q&A skill for customers and solution engineers about API and output semantics. It lives in `.agents/skills`, where Codex and Gemini CLI find it, and `.claude/skills/nearmap-ai-apis` is a symlink to it for Claude Code. Keep it tool-neutral: Agent Skills frontmatter only (`tests/test_agent_skills.py` checks it) and no instructions that assume one agent's tools. Its `references/` files restate behaviour documented here and in the code (with function names, not line numbers). When a change alters behaviour those files describe (area handling, parcelMode, rollup nulls, primary selection, roof age linking, damage rollups, …), update the matching reference file in the same PR. The same public-content rules apply as for the rest of the repository.

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
   - `combine_chunk_files(prefix, num_chunks, geo=)`: shared chunk consolidation for simple one-file-per-chunk exporters (RoofAge, DamageConflation) — **invalidates the s3fs listing before the existence sweep** so present chunks aren't missed on S3 output (the main `exporter.py` keeps its own streaming/parallel merge)
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
   - Feature geometry requests (`get_features_gdf()`, `get_features_gdf_bulk()`)
   - Date range and survey resource queries
   - Rollups are computed locally via `parcels.parcel_rollup()` from features.json responses. The Feature API's former rollup endpoints are deprecated and unused.

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
   - `conflation_rollup()`: Per-AOI rollup of Damage Conflation features (same primary-selection/`primary_*`/null-out conventions as `parcel_rollup`, single-class)
   - Re-exports `read_from_file` from `aoi_io` for backward compatibility

8. **primary_feature_selection.py**: Primary feature selection algorithms
   - `select_primary()`: Dispatcher for selection method
   - `select_primary_by_largest()`: Selects feature with largest area in parcel
   - `select_primary_by_nearest()`: Selects feature nearest to target point, prefers non-small features
   - `select_primary_optimal()`: Tries nearest first, falls back to largest

9. **feature_attributes.py**: Attribute flattening utilities
   - `flatten_building_attributes()`: Flattens nested building API attributes to flat dict
   - `flatten_roof_attributes()`: Flattens roof attributes including materials, spotlight index
   - `flatten_building_lifecycle_damage_attributes()`: Flattens Feature API per-survey damage classification scores
   - `flatten_conflated_damage_attributes()`: Flattens Damage Conflation API (ai/damage/v2) per-building `damage.event`/`damage.preEvent` ratings into a fixed `damage_event_*`/`damage_pre_event_*` column set
   - `flatten_roof_instance_attributes()`: Flattens Roof Age API result attributes
   - `calculate_roof_age_years()`: Computes roof age in years from installation/as-of dates

10. **geometry_utils.py**: Geometry processing utilities
    - `create_grid()` / `split_geometry_into_grid()`: Grid generation for large AOIs
    - `clip_features_to_polygon()`: Clips features to AOI boundary, recalculates areas
    - `combine_features_from_grid()`: Deduplicates and merges features from gridded queries
    - `polygon_to_coordstring()`: Converts Shapely polygon to API coordinate string

### Damage Conflation Layer (ai/damage/v2)

- **damage_conflation_api.py**: `DamageConflationApi(BaseApiClient)` — client for the Damage Conflation API
  - Event-scoped (`event_id`); POST `/events/{event_id}/latest.geojson` per AOI or address
  - `get_damage_by_aoi()` / `get_damage_by_address()` with `nextCursor` pagination (no gridding — pagination scales to a full event boundary)
  - `get_damage_bulk()` thread-pool fan-out; event-scoped cache layout `damageconflation/<event_id>/`
- **damage_conflation_exporter.py**: `DamageConflationExporter(BaseExporter)` and standalone CLI (`python -m nmaipy.damage_conflation_exporter`)
  - Requires `--event-id`; always emits per-building `damage_buildings.*`, opt-in `--rollup` emits per-AOI `damage_rollup.*` (with `--primary-decision largest|optimal`); also writes operational `damage_metadata.csv` and `damage_errors.csv`
  - Chunk consolidation uses the shared `BaseExporter.combine_chunk_files()` (invalidates the s3fs listing before the sweep, so present chunks aren't missed on S3 output)

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
    - Auto-generates a customer-readable `README.md` in export output directories
    - Lists files via the `output_files` registry; column descriptions come from `column_metadata.lookup_column()`
    - Reads `export_config.json` to detect country (for area units) and selection method
    - Works with both local and S3 output directories

17. **column_metadata.py**: Column metadata loader and lookup
    - Single source of truth for column descriptions, used by both the README generator and the data dictionary generator
    - `lookup_column()`: Resolves a column name (including scope prefixes like `primary_roof_`) to its metadata
    - Reads from `nmaipy/data/column_metadata.json` (+ `column_metadata_overrides.json`), which the PM edits when descriptions need refining

18. **output_files.py**: Registry of files the exporter produces
    - `FileSpec` entries describe each output file; `kind` (`ai_data`/`geometry`/`operational`/`config`) drives whether a file gets a data dictionary and/or a README listing
    - Consumed by `readme_generator` and `data_dictionary_generator` so output file metadata lives in one place

19. **data_dictionary_generator.py**: Per-file data dictionaries
    - Generates `<filename>_data_dictionary.csv` next to every tabular `ai_data` output in `final/`
    - Columns looked up via `column_metadata.lookup_column()`; user input columns are recognised by header-matching against the input AOI file

### Legacy/Internal

20. **coverage_utils.py**: Legacy coverage API utilities + event discovery
    - Survey coverage point queries and threading
    - Not integrated with current `BaseApiClient` architecture (uses its own `requests.Session`)
    - **Post-cat event discovery** (`discover_event(lat, lon) -> (event_id, boundary)`, plus `latest_event_id_at_point` / `event_boundary`): given a point, resolves the latest ImpactResponse `postCatEventId` (newest `postCatEventDate` wins; others logged) and unions the event's survey footprints into a shapely (Multi)Polygon — the operator then subdivides that boundary into AOIs and runs the Damage Conflation exporter with the discovered event id. Deployed-API notes (differ from the public doc): the id is on the `postCatEventId` tag (not `eventId`); the point query takes `{lon},{lat}` in the URL path; and the whole-event footprint comes from one tag-filtered query `/coverage/v2/surveys?include=postCatEventId:<id>` (the "Filter Surveys" `include=<type>:<name>` grammar — `filter=`/`tags=` are rejected), no spatial search.

21. **reference_code.py**: Minimal utility
    - `is_building_small()`: Checks building area < 30 sqm

### Configuration Files

- **pyproject.toml**: Package metadata and dependencies, the hatchling build config, and the pixi workspace, package and environments. Python >=3.12 required.
- **pixi.lock**: Locked conda-forge solve for every pixi environment. Regenerate with `pixi lock` after changing `pyproject.toml`; never edit by hand.

### Data Flow

1. User provides a GeoJSON, CSV, GeoPackage, or Parquet file with AOIs (Areas of Interest)
2. The exporter loads the AOI file via `aoi_io.read_from_file()` and divides work into chunks
3. Chunks are processed in parallel via `ProcessPoolExecutor` with API warmup ramp-up
4. For each AOI, the Feature API (and optionally Roof Age API) fetches relevant AI features
5. Features are filtered based on intersection with AOI boundaries
6. For large AOIs (>1 sq km), the system automatically grids them and merges/deduplicates results
7. Results are summarized into rollup statistics and/or saved as feature data per chunk. The chunk's rollup file is its completion marker for resume and the merge, so `process_chunk` writes it last and only when every other chunk file was written; a failed write is recorded, never raised, and leaves the chunk to be re-processed. `NearmapAIExporter.__init__` refuses to start when `chunks/` holds rollups written under a different config (only `_CONFIG_KEYS_SAFE_TO_CHANGE` may differ), before it overwrites `export_config.json`
8. Chunks are merged into final output files (rollup CSV/Parquet, per-class files, GeoParquet)
9. For S3 output, per-process local staging is used for formats requiring local I/O before uploading
10. A customer-readable `README.md` is auto-generated in `final/` by `ReadmeGenerator`, plus a `<filename>_data_dictionary.csv` next to each tabular AI-data output by `data_dictionary_generator`

### Resource tuning (`--processes`, `--chunk-size`, `--merge-read-workers`)

These are the operator-facing dials for RAM and consolidation speed. The export has two stages — parallel chunk processing (API fetch) and the single-process closeout (combined `features.parquet` stream + per-class merge):

- **`--processes`** (default `4`; must be `>= 1`) sets the parallel worker count for chunk processing AND derives the `features.parquet` streaming prefetch buffer:
  - Chunk-processing peak ≈ `processes × per-chunk feature tables` resident at once (one per worker).
  - `features.parquet`-streaming peak ≈ `(round(1.5 × processes) + 1) × dense-chunk-size` resident at once — the prefetch read-ahead window plus the one chunk currently being reconciled/written (see `_resolve_prefetch_workers` in `exporter.py`). Those per-chunk tables hold **all** classes, so this stays tied to `--processes` for RAM safety.
  - Both stages scale linearly with this dial. If memory pressure shows in either, drop `--processes`.
- **`--chunk-size`** (default `500`) sets how many AOIs each worker processes per chunk. Larger chunks → larger per-chunk feature tables → bigger per-process working set AND bigger per-table footprint at the streaming/merge stages, but **fewer** chunk files (fewer footer reads, fewer output row groups, less per-file overhead in the merge). Smaller chunks shrink memory at the cost of more chunk-boundary overhead and more files to combine. For very large national exports, a bigger `--chunk-size` can noticeably cut merge overhead if RAM allows.
- **`--merge-read-workers`** (default `0` = auto) sets the read-ahead concurrency for the **per-class merge** only (consolidation phase, `_resolve_merge_prefetch_workers`). Auto-sizes to `max(round(1.5 × processes), read-workers)` — i.e. the full S3/local read concurrency (24 on S3, 8 local) — because per-class chunks are a single class and small, so the merge can read at full concurrency without the lone parquet writer starving on I/O. Memory here ≈ `merge-read-workers × per-class-chunk-size` resident, far below the old read-all-then-concat peak. Raise it on a large-RAM box to read further ahead; lower it only if per-class chunks are unusually large (e.g. a big `--chunk-size` on the building layer).

Memory monitoring reports the working set (`memory.current − inactive_file`), matching kubelet / OOM-killer semantics. The displayed figure ignores reclaimable page cache. See `nmaipy/cgroup_memory.py` and the "Container memory display" notes in the v5.0.16 release. Each consolidation stage now shows a tqdm progress bar (`Streaming chunks` for `features.parquet`; `<Class> tabular`/`<Class> geo` for the per-class merge), so a long closeout no longer looks hung.

Operationally: start with the defaults, watch the working-set figure in the tqdm postfix during a real export, and tune `--processes` first (linear effect on both stages, no surprises) before reaching for `--chunk-size`. If the per-class merge is the bottleneck on a roomy box, raise `--merge-read-workers`.

For the retry / backoff / gridding strategy these dials sit on top of, see `RetryRequest` and `GriddedApiClient` in `nmaipy/api_common.py`. (A standalone engineering doc, `docs/retry-and-gridding.md`, is drafted but deliberately held back from the repo until reviewed — see commit 256f1b4.)

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

All API clients also accept `bearer_token=` — a short-lived Nearmap identity JWT sent as an `Authorization: Bearer` header instead of the `?apikey=` query parameter (e.g. for running nmaipy in a sandboxed environment). In bearer mode the api key is deliberately zeroed so a missed code path fails loudly rather than falling back to a long-lived env key; when both credentials are supplied the bearer token wins (shorter-lived credential preferred) with a warning. The token is captured at construction and never refreshed, so it suits short jobs only — the exporter CLIs intentionally have no `--bearer-token` option because exports can outlive a short-lived token.

For S3 output, AWS credentials are resolved from environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) or `~/.aws/credentials`.

## Version Management & Deployment

See `VERSIONING.md` for instructions on:
- Updating version numbers
- Building and testing packages
- Deploying to PyPI
- Creating releases
