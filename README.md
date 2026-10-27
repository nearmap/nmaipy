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

Output is written to `data/outputs/quick_test/final/`. Open `rollup.csv`, browse the rollup row per property, then have a look at `README.md` in the same directory — every export ships an auto-generated, customer-readable data dictionary explaining every column.

The script itself is short — read it (`run_10_test.py`) to see the full call signature.

## Common Use Cases

> **Running as a script:** the exporters start their worker processes with the `spawn` method, which re-imports your script in every worker. In a `.py` file, call `run()` under `if __name__ == "__main__":` (as the disaster-response examples do); notebooks and the REPL don't need it.

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

**Dataset selection:** pass `roof_age_dataset='latest'` (default — pointer maintained by the Nearmap team), `'A.0'`, `'A.1'`, `'A.1Q2'`, or any raw resource UUID. **Historical "as-of" queries:** combine with `since` / `until` to ask "what was this roof like in the past?" — these drive the API's `sinceAsOfDate` / `untilAsOfDate` body parameters. Note: `since` / `until` are not supported on the `A.0` dataset and are rejected client-side with a clear error.

Each roof carries:
- Predicted installation date
- Confidence score (trust signal)
- Evidence type and number of imagery captures analysed
- Timeline of all imagery used

Useful for insurance underwriting, property valuation, maintenance planning, and real-estate due diligence.

### Disaster Response

After a catastrophe (hurricane, tornado, wildfire, flood), Nearmap flies **post-catastrophe (post-cat) surveys** over the affected area. Three kinds of damage data come from them:

- [**Damage Classifications**](#damage-classifications) — a FEMA-style damage rating for each building, from a single survey (Feature API).
- [**AI Layers on Post-Catastrophe Surveys**](#ai-layers-on-post-catastrophe-surveys) — debris, tarps, missing shingles, structural damage and other AI layers from a post-cat survey (Feature API).
- [**Damage Conflation**](#damage-conflation) — one rating per building fused across every capture of the event, retrieved for a whole event at once (Damage Conflation API).

Three rules apply to both Feature API workflows:

- **Packs are either post-cat or standard**, and a request only returns features from surveys of the matching kind. The post-cat packs are `damage`, `damage_classifications`, `postcat` and `postcat_others`. They combine with each other, but mixing one with any standard pack (e.g. `building`) is rejected with `400 INCOMPATIBLE_PACKS`.
- **Use small AOIs**: individual property boundaries, or the smallest statistical unit of the country — census blocks (US) or mesh blocks (AU).
- **Set `since` to the event date** so only surveys flown after the event are used. AOIs outside post-cat coverage fail with `404 no survey resources found` (listed in `feature_api_errors.csv`); post-cat AI coverage can stop slightly short of the imagery footprint, so expect some of these along the edge of an event.

The examples use Hurricane Isaias (event date 2026-10-09, Florida Panhandle). To find an event's id and footprint from a point inside the damage zone, see [Damage Conflation](#damage-conflation).

#### Damage Classifications

The Feature API's damage classification is a **per-survey, real-time rating**: every `Building lifecycle` feature gets a FEMA 5-tier class — `Undamaged`, `Affected`, `Minor`, `Major` or `Destroyed` — with a confidence for each tier, computed from that one survey's imagery. Request the `damage` pack (Building lifecycle on post-cat surveys) and add `include=['damage']`:

```python
from nmaipy.exporter import NearmapAIExporter

if __name__ == "__main__":
    exporter = NearmapAIExporter(
        aoi_file='aois.geojson',        # property boundaries or census / mesh blocks, inside the event footprint
        output_dir='isaias_damage_classifications',
        country='us',
        packs=['damage'],               # Building lifecycle features from post-cat surveys
        include=['damage'],             # adds the 5-tier damage rating to each Building lifecycle feature
        since='2026-10-09',             # event date: only surveys flown after it
        rapid=True,                     # also use rapid post-cat surveys when available
    )
    exporter.run()
```

The rollup gets the primary building's rating — `primary_building_lifecycle_damage_class`, its confidence, and one confidence column per tier — and `building_lifecycle.csv` carries every building's 5-tier, 3-tier (`UndamagedOrAffected` / `Minor` / `MajorOrDestroyed`) and 2-tier confidences.

- `rapid=True` also lets the API use surveys from Nearmap's rapid AI pipelines (e.g. during an event). They deliver only Building lifecycle and the damage rating, so `rapid` works only when Building lifecycle is the sole class requested (the `damage` pack); anything else is rejected with `400 UNSUPPORTED_CLASS_RAPID`.
- For a pre-event baseline, request `damage_non_postcat` — the same class and rating from standard surveys — with `until` before the event.
- To get the rating alongside the post-cat AI layers below, add `include=['damage']` to that export instead (without `rapid`).

#### AI Layers on Post-Catastrophe Surveys

Post-cat surveys get the full Nearmap AI suite, from the same processing as standard surveys. What changes is access: **packs control which layers you can read from post-cat versus standard surveys**, and the pack names don't make that obvious. The standard `debris` pack reads standard surveys only: with `since` set to the event date it fails with `404 no survey resources found`, and without `since` it silently returns debris from the latest *pre-event* survey. To read debris, tarps, missing shingles, exposed underlayment, temporary repairs or structural damage from a post-cat survey, use the `postcat` pack:

```python
from nmaipy.exporter import NearmapAIExporter

if __name__ == "__main__":
    exporter = NearmapAIExporter(
        aoi_file='aois.geojson',           # property boundaries or census / mesh blocks, inside the event footprint
        output_dir='isaias_postcat_layers',
        country='us',
        packs=['postcat'],                 # post-cat AI layers; never mixed with standard packs such as 'building'
        since='2026-10-09',                # event date: only surveys flown after it
        save_features=True,                # per-class GeoParquet with feature geometries
    )
    exporter.run()
```

Which pack to ask for:

| You want | Standard surveys | Post-cat surveys |
|----------|------------------|------------------|
| Building lifecycle + damage rating (`include=['damage']`) | `damage_non_postcat` | `damage` |
| Debris (Vegetation Debris, Junk and Wreckage, Tyres) | `debris` | `postcat` |
| Buildings and roofs | `building` | `postcat` |

`postcat` (29 classes) also covers roof damage (Missing Roof Tile or Shingle, Exposed Underlayment, Exposed Roof Deck, Tarpaulin/Shade Cloth, Roof with Temporary / Permanent Repair), structural damage (Structural Damage, Building with Structural Damage, Building Structural Loss, Building Damage by Tree), swimming pools and Building lifecycle. `damage_classifications` is a 10-class, damage-focused subset of it, and `postcat_others` (100+ classes) holds the long tail — Roof Debris, Yard Debris, roof objects, vegetation, surfaces and more. `FeatureApi().get_feature_classes(packs=['postcat'])` lists the classes your account can query.

#### Damage Conflation

The Damage Conflation API is the **multi-date** counterpart of the damage classification above: instead of one survey's rating, it returns one *conflated* rating per building — the highest-confidence assessment across every capture in the event's lifecycle. Queries are scoped by event id rather than by date or survey, so there is no survey to choose. Large AOIs are paged, so areas bigger than census blocks work too — up to the whole event footprint — though large AOIs take longer to return. This script goes from one point to conflated damage for the whole event, one census block at a time:

```python
"""Pull conflated building damage for a whole catastrophe event, starting from one point."""

import geopandas as gpd

from nmaipy.coverage_utils import discover_event
from nmaipy.damage_conflation_exporter import DamageConflationExporter

LAT, LON = 30.15465, -85.76716  # Panama City Beach, FL: inside the Hurricane Isaias (2026) damage zone
# US Census 2020 blocks for Florida (TIGER/Line 2022, ~190 MB), read straight from census.gov
BLOCKS_URL = "https://www2.census.gov/geo/tiger/TIGER2022/TABBLOCK20/tl_2022_12_tabblock20.zip"
AOI_FILE = "isaias_blocks.parquet"


def main():
    # 1. Point -> the latest post-catastrophe event covering it, plus that event's footprint
    #    (the union of its post-cat surveys). since/until bracket the event, so a later event
    #    at the same spot can't take over.
    event_id, footprint = discover_event(lat=LAT, lon=LON, since="2026-10-09", until="2026-12-31")
    # event_id == "442c3974-5b6c-5335-a6e4-83b5b9afc8b7"  (Hurricane Isaias, event date 2026-10-09)

    # 2. Footprint -> one AOI per census block, clipped to the surveyed area. Block boundaries
    #    mostly run down the middle of roads; an arbitrary grid would cut through buildings and
    #    return each one in every AOI it touches. The bbox read is in the file's CRS (NAD83),
    #    hence the small buffer.
    minx, miny, maxx, maxy = footprint.bounds
    blocks = gpd.read_file(
        BLOCKS_URL, columns=["GEOID20"], bbox=(minx - 0.01, miny - 0.01, maxx + 0.01, maxy + 0.01)
    ).to_crs("EPSG:4326")
    blocks = gpd.clip(blocks, footprint, keep_geom_type=True)
    blocks = blocks[~blocks.is_empty].rename(columns={"GEOID20": "aoi_id"})
    blocks.to_parquet(AOI_FILE)

    # 3. One conflated damage rating per building, fused across every capture of the event.
    DamageConflationExporter(
        aoi_file=AOI_FILE,
        output_dir="isaias_damage",
        event_id=event_id,
        country="us",
        output_format="both",  # GeoParquet + CSV
        rollup=True,  # adds a per-block rollup (n_buildings, n_major, ...) alongside the per-building rows
        processes=8,
        threads=1,  # the conflation API rate-limits well below the Feature API; keep concurrency low
        chunk_size=100,
        max_retries=20,  # default 10; AOIs still failing after this are listed in damage_errors.csv
    ).run()


if __name__ == "__main__":  # required: nmaipy's worker pool uses the "spawn" start method
    main()
```

`discover_event` picks the **latest** post-catastrophe event covering the point and logs any others — this point also sits under Hurricane Michael (2018) and a 2024 severe-weather event — so pass `since`/`until` to pin a specific one (they also bound the captures that make up the footprint). The event id is the Coverage API's `postCatEventId` survey tag. The two steps are also available separately: `latest_event_id_at_point(lat, lon)` and `event_boundary(event_id)` (one tag-filtered Coverage query — `include=postCatEventId:<id>` — unioned, no spatial search).

The exporter always emits per-building damage polygons (`damage_event_*` / `damage_pre_event_*` rating, confidence and rawRatings scores); `rollup=True` adds a per-AOI summary (rating counts + the primary building, which is most meaningful when AOIs are property-sized). Ratings use the same five tiers, except that the lowest is `NoDamage` (the Feature API's `Undamaged`). Unlike the Feature API, an AOI outside the event footprint is not an error: it succeeds with no buildings. A building that straddles two AOIs is returned for each, with the same `feature_id` and rating, so deduplicate on `feature_id` when counting across AOIs — census blocks keep this rare. The Isaias footprint covers ~14,700 blocks, so slice `blocks` (e.g. `blocks.head(10)`) for a quick first run. See [Standalone Damage Conflation Export](#standalone-damage-conflation-export) for the CLI and the files written to `final/`.

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
| `pool` | Swimming pool detection |
| `solar` | Solar panel detection |
| `debris` | Vegetation debris, junk and wreckage, tyres (standard surveys) |
| `damage` / `damage_non_postcat` | Building lifecycle, for the damage rating (`include=['damage']`) on post-cat / standard surveys |
| `postcat` / `postcat_others` / `damage_classifications` | AI layers on post-catastrophe surveys (see [Disaster Response](#disaster-response)) |

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

Any AOI larger than 1 km² is **automatically gridded** — split into 0.005° (~550 m) cells, processed in parallel, then recombined and deduplicated. No configuration needed for the common case:

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

Every Python API option has a CLI equivalent. See `python -m nmaipy.exporter --help` for the full list. Highlights:

```bash
python -m nmaipy.exporter \
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
| `--packs` | AI packs to extract (`building`, `vegetation`, `surfaces`, `pool`, `solar`, `damage`, etc.) |
| `--include` | Additional calculated includes (`roofSpotlightIndex`, `roofConditionConfidenceStats`, `damage`, `defensibleSpace`, `hurricaneScore`, `windScore`, `hailScore`, `wildfireScore`, `windHailRiskScore`) |
| `--roof-age` | Include Roof Age API data (US only) |
| `--roof-age-dataset` | `latest` (default), `A.0`, `A.1`, `A.1Q2`, or any raw resource UUID |
| `--prefer3d` | Prefer 3D surveys, fall back to 2D per AOI when no 3D coverage exists in the window. Mutually exclusive with `--only3d` |
| `--only3d` | Restrict every query to 3D surveys (no fallback) |
| `--system-version-prefix` | Restrict to a specific AI generation (e.g. `gen6-`) |
| `--system-version` | Pin to an exact version (e.g. `gen6-glowing_grove-1.0`) |
| `--rapid` | Also consider rapid post-catastrophe surveys (Building lifecycle / `damage` pack only) |
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

Event-scoped, building-level conflated damage for a catastrophe event (requires `--event-id`). Resolve the event id from a point inside the damage zone, then export your AOIs (e.g. properties) inside its footprint:

```bash
# Resolves to 442c3974-5b6c-5335-a6e4-83b5b9afc8b7 (Hurricane Isaias, event date 2026-10-09)
EVENT_ID=$(python -c "from nmaipy.coverage_utils import latest_event_id_at_point; print(latest_event_id_at_point(30.15465, -85.76716, since='2026-10-09', until='2026-12-31'))")

python -m nmaipy.damage_conflation_exporter \
    --aoi-file "us_properties.geojson" \
    --output-dir "damage_results" \
    --event-id "$EVENT_ID" \
    --country us \
    --processes 4 \
    --threads 1 \
    --max-retries 20 \
    --output-format both \
    --rollup
```

Outputs in `final/`:

| File | When | Contents |
|------|------|----------|
| `damage_buildings.{parquet,csv}` | always | One row per building — `damage_event_*` / `damage_pre_event_*` rating, confidence and rawRatings, plus geometry |
| `damage_rollup.{parquet,csv}` | `--rollup` | One row per input AOI — your input file's columns (geometry excluded), then rating counts + the primary building (chosen by `--primary-decision largest|optimal`) |
| `damage_metadata.csv` | always | Per-AOI query metadata (event id/name, model version) for AOIs that returned a response |
| `damage_errors.csv` | on failures | Per-AOI requests that still failed after `--max-retries` (`status_code` + `message`). An AOI outside the event footprint is not a failure: it succeeds with no buildings (`query_succeeded=True`, `n_buildings=0` in the rollup) |
| `latency_stats.csv` | always | Per-chunk API latency summary (request count, percentiles, retries) |
| `damage_conflation_export_config.json` | always | The export's parameters and `nmaipy` version |

The conflation API is map-style: every building intersecting an AOI is returned, including neighbours' buildings that only clip a parcel boundary. For property-sized AOIs, `--parcel-mode` drops buildings that barely intersect their AOI (kept when at least 50% of the building lies inside it, or its in-AOI footprint is at least 30 sqm so multi-parcel structures such as townhouse rows survive). It is a client-side approximation of the Feature API's `parcelMode`; nothing is clipped. Off by default.

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
