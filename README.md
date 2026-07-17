# nmaipy - Nearmap AI Python Library

Extract building footprints, vegetation, damage assessments, roof condition, and other AI features from Nearmap's aerial imagery using simple Python code.

**Supported countries:** `au` (Australia), `us` (United States), `nz` (New Zealand), `ca` (Canada)

## Quick Start

### 1. Install

```bash
pip install nmaipy
```

Requires Python 3.12 or later.

### 2. Set your API key

```bash
export API_KEY=your_api_key_here
```

Contact your account administrator if you don't have one.

Short-lived bearer tokens are also supported at the library level: pass `bearer_token=` to any API client (`FeatureApi`, `RoofAgeApi`, `DamageConflationApi`, or the `coverage_utils` functions) to authenticate with an `Authorization: Bearer` header instead of an API key — useful when running nmaipy in a sandboxed environment where a long-lived key can't be exposed. The token is captured at construction and never refreshed, so keep jobs shorter than its lifetime. For that reason the exporter CLIs deliberately have no `--bearer-token` option: exports can run for hours, longer than a short-lived token lasts.

### 3. Run your first extraction

The repository ships with a 10-property smoke-test script (`run_10_test.py`) that extracts building features and roof age predictions for 10 US properties. It's the fastest way to verify your setup end-to-end:

```bash
python run_10_test.py
```

Output is written to `data/outputs/quick_test/final/`. Open `rollup.csv`, browse the rollup row per property, then have a look at `README.md` in the same directory — every export now ships an auto-generated, customer-readable data dictionary explaining every column.

The script itself is short — read it (`run_10_test.py`) to see the full call signature.

## Common Use Cases

### Population-scale analysis with mesh blocks / census blocks

`nmaipy` doesn't only consume parcel polygons. **Mesh blocks (AU), census blocks (US/CA), suburbs, statistical areas — anything you can express as polygons** can be passed in as AOIs. This gives you a **contiguous** map of AI features you can directly merge with demographic, economic, or other reference data on the same geographic units.

Pull a representative time-window (e.g. 12 months) for the latest snapshot:

```python
from nmaipy.exporter import NearmapAIExporter

exporter = NearmapAIExporter(
    aoi_file='melbourne_mesh_blocks.geojson',  # or census blocks, suburbs, etc.
    output_dir='melbourne_2025',
    country='au',
    packs=['building', 'vegetation', 'surfaces', 'solar'],
    since='2025-01-01',
    until='2025-12-31',
    save_features=True,            # per-class GeoParquet with feature geometries
    include_parcel_geometry=True,  # keep block boundaries for GIS joins
    processes=8,
)
exporter.run()
```

**For change detection**, repeat the same call with a different `since` / `until` window and diff the resulting rollups. Each AOI gets a `survey_date` and `system_version` so you know exactly which capture / model version produced each row — useful when comparing vintages.

> **Tip:** if your input has a `since` or `until` column (string-typed `YYYY-MM-DD`), it overrides the bulk values per row, so you can mix windows in a single export.

### Disaster Response

Damage classification is a separate pack family. For a post-event sweep, request the buildings together with damage so you get the building footprint + the damage attributes on the same parcels:

```python
exporter = NearmapAIExporter(
    aoi_file='affected_areas.geojson',
    output_dir='damage_assessment',
    country='us',
    packs=['building', 'damage_classifications', 'damage'],   # geometry + damage classes
    since='2024-07-08',             # date range of the event
    until='2024-07-11',
    rapid=True,                     # Permit use of rapid post-catastrophe imagery (e.g. during an event)
    include=['damage'],             # Adds the damage classification score (FEMA-style damage classification levels on a 5-point scale)
    save_features=True,
)
exporter.run()
```

`rapid=True` enables consideration of post-catastrophe rapid-response surveys (when available). The damage pack has variants (`damage_postcat`, `damage_non_postcat`) for damage-related feature classes; `damage_classifications` is the richer pack used here, exposing ten damage feature classes (Roof, Missing Roof Tile or Shingle, Vegetation Debris, Junk and Wreckage, Building Structural Loss, Roof with Temporary Repair, Structural Damage, Building Damage by Tree, Building with Structural Damage, Building lifecycle). The damage classification *score* is a separate quantity, retrieved by passing `include=['damage']` — see the auto-generated README in your export's `final/` directory for what each emitted column contains.

#### Finding a catastrophe event from a point

For ImpactResponse workflows you often know roughly *where* an event hit but not its event id. Given a lat/lon, `nmaipy.coverage_utils.discover_event` resolves the **latest** post-catastrophe event covering that point and assembles its footprint — in memory, no files:

```python
from nmaipy.coverage_utils import discover_event

event_id, boundary = discover_event(lat=27.742889, lon=-82.754961)
# event_id: str UUID (e.g. Hurricane Milton); boundary: shapely (Multi)Polygon in EPSG:4326
```

You then subdivide `boundary` into your own AOIs (e.g. census/mesh blocks) and run the exporters over them with the discovered `event_id`. Pass `since`/`until` to disambiguate a location hit by more than one event (e.g. St Pete Beach sits under both Hurricanes Milton and Helene — the latest wins, and the others are logged). The pieces are also available separately: `latest_event_id_at_point(lat, lon)` and `event_boundary(event_id)` (the boundary is one tag-filtered Coverage query — `include=postCatEventId:<id>` — unioned, no spatial search).

### Roof Age Analysis (US Only)

Predict roof installation dates using AI analysis of historical imagery, combined with building permits and climate data. Recommended approach — unified with the Feature API in one export:

```python
exporter = NearmapAIExporter(
    aoi_file='properties.geojson',
    output_dir='unified_results',
    country='us',
    packs=['building', 'building_structures'],
    roof_age=True,                # add Roof Age API predictions
    save_features=True,
)
exporter.run()
```

You can also use the standalone exporter if you only need roof age data (no Feature API calls):

```python
from nmaipy.roof_age_exporter import RoofAgeExporter

exporter = RoofAgeExporter(
    aoi_file='properties.geojson',
    output_dir='roof_age_results',
    country='us',
    threads=10,
    output_format='both',         # GeoParquet and CSV
)
exporter.run()
```

**Dataset selection:** pass `roof_age_dataset='latest'` (default — pointer maintained by the Nearmap team), `'A.0'`, `'A.1'`, or any raw resource UUID. **Historical "as-of" queries:** combine with `since` / `until` to ask "what was this roof like in the past?" — these drive the API's `sinceAsOfDate` / `untilAsOfDate` body parameters. Note: `--until` / `--since` are not supported on the `A.0` dataset and are rejected client-side with a clear error.

Each roof carries:
- Predicted installation date
- Confidence score (trust signal)
- Evidence type and number of imagery captures analysed
- Timeline of all imagery used

Useful for insurance underwriting, property valuation, maintenance planning, and real-estate due diligence.

### Damage Conflation (Event-Scoped Damage)

For a catastrophe event (hurricane, wildfire), the Damage Conflation API returns a single *conflated* damage rating per building — the highest-confidence assessment fused across every capture in the event lifecycle, rather than one survey's classification. Query it standalone with an event id:

```python
from nmaipy.damage_conflation_exporter import DamageConflationExporter

exporter = DamageConflationExporter(
    aoi_file='properties.geojson',
    output_dir='damage_results',
    event_id='2f510853-5d55-50f4-9102-2c02de08190e',  # e.g. Hurricane Milton
    country='us',
    output_format='both',         # GeoParquet and CSV
    rollup=True,                  # also emit a per-AOI summary
)
exporter.run()
```

Always emits per-building damage polygons (`damage_event_*` / `damage_pre_event_*` rating, confidence and rawRatings scores); `rollup=True` adds a per-AOI summary (rating counts + the primary building). Large AOIs — up to a full event boundary — are paged, not gridded. The `event_id` comes from the Coverage API's `eventId` survey tag.

### Wildfire Risk & Defensible Space

Assess wildfire vulnerability with defensible space analysis around structures — per-zone metrics and per-class risk object breakdowns (vegetation, neighbouring roofs, yard debris) across three concentric zones (0-indexed, matching the CalFire convention):

```python
exporter = NearmapAIExporter(
    aoi_file='properties.geojson',
    output_dir='wildfire_risk',
    country='us',
    packs=['building'],
    include=['defensibleSpace', 'wildfireScore'],
    save_features=True,
)
exporter.run()
```

Output includes per-zone metrics (zone area, defensible-space area, coverage ratio) and per-class risk-object breakdowns for both the primary roof and the aggregate parcel.

### Roof Condition (RSI) with Structural Damage Fallback

Extract Roof Spotlight Index scores that automatically resolve the best RSI per roof — using the roof's own score when available, or falling back to the Building Lifecycle score when structural damage is present:

```python
exporter = NearmapAIExporter(
    aoi_file='properties.geojson',
    output_dir='rsi_results',
    country='us',
    packs=['building', 'building_structures'],
    include=['roofSpotlightIndex'],
    save_features=True,
)
exporter.run()
```

The `building_structures` pack provides the Building Lifecycle features needed for the fallback. Without it, RSI is only available from roofs that have no structural damage.

### 3D-first exports with 2D fallback (`prefer3d`)

When you want **maximum 3D attribute coverage** in a single unified export — without losing AOIs that only have 2D coverage in your date window — use `prefer3d=True`. The exporter runs the bulk request as if `only3d=True`, then transparently retries each AOI that returned no 3D survey with `only3d=False`:

```python
exporter = NearmapAIExporter(
    aoi_file='properties.geojson',
    output_dir='unified_3d_results',
    country='us',
    packs=['building', 'building_char', 'roof_char'],
    prefer3d=True,
    save_features=True,
)
exporter.run()
```

The output rollup carries a `mesh_date` column — non-empty when the row used a 3D survey, empty when it fell back to 2D. Mutually exclusive with `only3d`. No-op for AOIs pinned to a `survey_resource_id` (the survey is already chosen).

## Discovering available AI Features

Packs and feature classes are not hard-coded — your account's available packs (and their classes) are returned by the API. List them programmatically:

```python
from nmaipy.feature_api import FeatureApi

api = FeatureApi()  # uses API_KEY env var

# Dict of pack code -> list of feature class UUIDs your account can query
packs = api.get_packs()
print(sorted(packs.keys()))

# DataFrame of feature classes (rich metadata: description, type, perspective, etc.)
classes = api.get_feature_classes()
print(classes[["description", "type"]].head())

# Filter to a specific pack
building_classes = api.get_feature_classes(packs=["building"])
```

Commonly available packs (this list is not exhaustive):

| Pack | Description |
|------|-------------|
| `building` | Building footprints and heights |
| `building_char` | Detailed building characteristics |
| `roof_char` | Detailed roof characteristics (material, shape, etc.) |
| `vegetation` | Trees and vegetation coverage |
| `surfaces` | Ground surface materials |
| `pools` | Swimming pool detection |
| `solar` | Solar panel detection |
| `damage` / `damage_postcat` / `damage_non_postcat` | Damage classification (post-catastrophe and lifecycle) |

For the full catalog and access details, use the snippet above or check the [Nearmap help site](https://help.nearmap.com).

## Input Data Formats

`nmaipy` accepts AOIs in:

- **GeoJSON** — standard geospatial polygons
- **GeoPackage** (`.gpkg`) — OGC standard
- **Parquet / GeoParquet** — efficient columnar format for large datasets
- **CSV / TSV / PSV** — text format with a `geometry` column containing WKT polygons

Your file should contain polygons representing the areas you want to analyse — parcels, mesh blocks, census blocks, suburbs, statistical areas, custom regions, etc. An `aoi_id` column identifies each row; if absent, sequential IDs are generated. Optional per-row `since` / `until` (string-typed) and `survey_resource_id` columns override bulk values per-AOI.

## Output Data

The exporter writes results to `{output_dir}/final/`:

| File | Description |
|------|-------------|
| `rollup.csv` or `.parquet` | One row per AOI with summary statistics (counts, areas, primary-feature attributes, scores, metadata) |
| `rollup_data_dictionary.csv` | Customer-readable data dictionary for every column in `rollup` |
| `{class}.csv` | Per-class attribute tables (e.g. `roof.csv`, `building.csv`, `swimming_pool.csv`, `solar_panel.csv`, `roof_instance.csv`, `building_lifecycle.csv`) |
| `{class}_data_dictionary.csv` | Data dictionary for each per-class file |
| `{class}_features.parquet` | Per-class GeoParquet with feature geometries (when `save_features=True`) |
| `features.parquet` | All features combined as GeoParquet (when `save_features=True`) |
| `feature_api_errors.csv` | AOIs where the Feature API returned errors (with status code and message) |
| `roof_age_errors.csv` | AOIs where the Roof Age API returned errors (US only) |
| `classes_availability.json` | Per-class availability metadata for the resolved system version |
| `latency_stats.csv` | API latency diagnostics (P50, P90, P95, P99) |
| `export_config.json` | Full record of export parameters and `nmaipy` version, used by the fast-path re-invocation detector |
| `README.md` | **Auto-generated, customer-readable README** — column tables grouped by section, class hierarchy diagram, primary-feature selection explainer, full data dictionary for the export |

A `{output_dir}/chunks/` directory holds intermediate per-chunk results during processing and enables resume after interruption.

For column-level documentation, open `final/README.md` from your export — it always reflects exactly what's in the files (including any 3D / prefer3d / pack-specific columns).

## S3 Output Support

Pass an `s3://` URI as `output_dir` to write directly to Amazon S3:

```python
exporter = NearmapAIExporter(
    aoi_file='properties.geojson',
    output_dir='s3://my-bucket/nmaipy-results/',
    country='us',
    packs=['building'],
)
exporter.run()
```

AWS credentials are resolved from environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) or `~/.aws/credentials`. The `cache_dir` parameter also accepts S3 URIs, though local caching is faster for iterative development.

## Working with Large Areas

Any AOI larger than 1 km² is **automatically gridded** — split into ~200 m cells, processed in parallel, then recombined and deduplicated. No configuration needed for the common case:

```python
exporter = NearmapAIExporter(
    aoi_file='large_region.geojson',
    output_dir='large_area_results',
    country='us',
    packs=['building'],
    processes=16,
)
exporter.run()
```

By default, gridded AOIs require **every cell to succeed** (`aoi_grid_min_pct=100`). Use a lower value to allow partial coverage at the cost of mixing survey dates within an AOI; set `aoi_grid_inexact=True` to silence the multi-date warning.

## Performance Tips

1. **Parallel processing**: set `processes` to roughly the number of CPU cores available.
2. **Tune chunk size**: `chunk_size` (default 500) groups AOIs per parallel work unit. Smaller = finer parallelism and cheaper resume; larger = lower overhead.
3. **Cache API responses**: `cache_dir` persists API responses to a directory. Re-runs with the same parameters reuse the cache. Defaults to `{output_dir}/cache/`.
4. **Filter by date**: `since` / `until` restrict to specific time periods and reduce data volume.
5. **Fast-path re-invocation**: if `final/README.md` and `export_config.json` already exist and the config matches the current call, the exporter returns in ~1s instead of rebuilding. Pure perf knobs (`processes`, `threads`, `chunk_size`, `cache_dir`) are ignored when comparing configs.

## Command Line Interface

Every Python API option has a CLI equivalent. See `python nmaipy/exporter.py --help` for the full list. Highlights:

```bash
python nmaipy/exporter.py \
    --aoi-file "parcels.geojson" \
    --output-dir "results" \
    --country us \
    --packs building vegetation \
    --save-features \
    --roof-age
```

Key options:

| Flag | What it does |
|------|--------------|
| `--packs` | AI packs to extract (`building`, `vegetation`, `surfaces`, `pools`, `solar`, `damage`, etc.) |
| `--include` | Additional calculated includes (`roofSpotlightIndex`, `defensibleSpace`, `hurricaneScore`, `windScore`, `hailScore`, `wildfireScore`, `windHailRisk`) |
| `--roof-age` | Include Roof Age API data (US only) |
| `--roof-age-dataset` | `latest` (default), `A.0`, `A.1`, or any raw resource UUID |
| `--prefer3d` | Prefer 3D surveys, fall back to 2D per AOI when no 3D coverage exists in the window. Mutually exclusive with `--only3d` |
| `--only3d` | Restrict every query to 3D surveys (no fallback) |
| `--system-version-prefix` | Restrict to a specific AI generation (e.g. `gen6-`) |
| `--system-version` | Pin to an exact version (e.g. `gen6-glowing_grove-1.0`) |
| `--rapid` | Consider rapid post-catastrophe surveys (damage workflows) |
| `--since` / `--until` | Filter by survey date range (`YYYY-MM-DD`). Per-AOI columns override per-row. With `--roof-age`, drives `sinceAsOfDate` / `untilAsOfDate` for historical roof state |
| `--save-features` | Save per-class GeoParquet files with feature geometries |
| `--tabular-file-format` | `csv` (default) or `parquet` for rollup and per-class attribute files |
| `--primary-decision` | Feature selection method: `largest_intersection`, `nearest`, or `optimal` |
| `--cache-dir` / `--no-cache` | Persist API responses to a directory / disable caching |
| `--max-retries` | Maximum API retry attempts (default 10) |
| `--api-key` | Override the `API_KEY` env var |

### Standalone Roof Age Export (US Only)

If you only need roof age data with no Feature API calls:

```bash
python -m nmaipy.roof_age_exporter \
    --aoi-file "us_properties.geojson" \
    --output-dir "roof_age_results" \
    --country us \
    --processes 4 \
    --output-format both
```

Accepts the same `--roof-age-dataset`, `--until`, `--since` flags as the unified exporter (with the same A.0-dataset rejection rule). See `python -m nmaipy.roof_age_exporter --help` for all options.

### Standalone Damage Conflation Export

Event-scoped, building-level conflated damage for a catastrophe event (requires `--event-id`):

```bash
python -m nmaipy.damage_conflation_exporter \
    --aoi-file "us_properties.geojson" \
    --output-dir "damage_results" \
    --event-id "2f510853-5d55-50f4-9102-2c02de08190e" \
    --country us \
    --processes 4 \
    --output-format both \
    --rollup
```

Outputs in `final/`:

| File | When | Contents |
|------|------|----------|
| `damage_buildings.{parquet,csv}` | always | One row per building — `damage_event_*` / `damage_pre_event_*` rating, confidence and rawRatings, plus geometry |
| `damage_rollup.{parquet,csv}` | `--rollup` | One row per input AOI — rating counts + the primary building (chosen by `--primary-decision largest|optimal`) |
| `damage_metadata.csv` | always | Per-AOI query metadata (event id/name, model version) for AOIs that returned a response |
| `damage_errors.csv` | on failures | Per-AOI failures (`status_code` + `message`) — e.g. a `403` for an AOI outside the event footprint |

See `python -m nmaipy.damage_conflation_exporter --help` for all options.

## Examples

- **`run_10_test.py`** — the 10-property smoke test described in Quick Start.
- **`examples.py`** — working examples covering building / vegetation extraction, damage assessment, multi-pack urban planning, vegetation analysis, pool detection, large-area gridding, time-series, and unified roof age.
- **`data/examples/`** — sample AOI files:
  - `sydney_parcels.geojson` — Sydney CBD, AU
  - `us_parcels.geojson` — Austin, TX
  - `large_area.geojson` — 2 km × 2 km Melbourne (triggers auto-gridding)
- **`notebooks/Coverage_Checker.ipynb`** — Jupyter notebook for coverage checking.

## Requirements

- **Python 3.12+**
- **Nearmap API key** (contact Nearmap for access)
- **Memory**: assume ~1 GB per process for large exports. The default `processes=N` (CPU count) on a 16-core box with `chunk_size=500` typically uses 10–20 GB.
- **AWS credentials** (only if writing to S3)

## Getting Help

- **Issues / feature requests**: [GitHub](https://github.com/nearmap/nmaipy/issues)
- **API key & access**: contact Nearmap
- **Column-level documentation**: open `final/README.md` inside your export's output directory

## License

See `LICENSE` for details.
