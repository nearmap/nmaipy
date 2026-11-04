# Nearmap AI Export

This folder contains AI-generated property data from Nearmap aerial imagery.


---

## Files in This Export

| File Name | Description |
|-----------|-------------|
| `building.csv` | Per-building data with feature attributes |
| `building_features.parquet` | Building polygons with geometry (GeoParquet) |
| `classes_availability.json` | Available feature classes per AOI |
| `features.parquet` | All detected features with geometry (GeoParquet) |
| `latency_stats.csv` | API call timing statistics |
| `rollup.csv` | Property-level summary with one row per parcel containing aggregated statistics |
| `roof.csv` | Per-roof data with feature attributes |
| `roof_features.parquet` | Roof polygons with geometry (GeoParquet) |

## Feature Classes in This Export

Column names use snake_case versions of class names:

| Class Name | Column Prefix |
|------------|---------------|
| Building | `building_` |
| Roof | `roof_` |

## Class Hierarchy & Relationships

nmaipy connects structural classes so that primary-feature columns and per-class files can be cross-referenced. The canonical hierarchy is shown below; only the layers actually requested in your export will have corresponding files in `final/`.

```
  Building Lifecycle  (stable building identity, linked across surveys)
                        │  parent_id
                        ▼
  Building            (building footprint)
                        │  spatial IoU
                        ▼
  Roof                (roof footprint)
                        │  spatial IoU
                        ▼
  Roof Instance       (roof clipped to parcel — the unit roof age is reported on)
```

**How the layers are linked:**

- **Building Lifecycle ↔ Building** is a 1-hop traversal of the API's `parent_id`.
- **Building ↔ Roof** and **Roof ↔ Roof Instance** use spatial Intersection over Union (IoU), not `parent_id`. The roof's API `parentId` points to a deprecated Building class, so nmaipy re-links via geometry, with an IoU-based threshold to assign a parent.
- Each parent feature gets a `primary_child_*_id` (the child with the highest IoU); each child gets a `parent_*_id`.
- When the primary roof has structural damage that masks its polygon, RSI and similar scores fall back through Roof → Building → Building Lifecycle.

## Column Naming Patterns

### Rollup Columns (rollup file)

For each feature class, the following columns are generated:

| Pattern | Example | Type | Min | Max | Unit | Description |
|---------|---------|------|-----|-----|------|-------------|
| `{class}_present` | `building_present` | Y/N | — | — | — | Feature was detected |
| `{class}_count` | `building_count` | int | 0 | — | — | Number of features detected |
| `{class}_confidence` | `building_confidence` | float (quantised uint8) | 0.0 | 1.0 | — | Combined confidence score indicating likelihood that any features of this class are present |
| `{class}_total_area_sqm` | `building_total_area_sqm` | float | 0 | — | square metres | Total area of all features |
| `{class}_total_clipped_area_sqm` | `building_total_clipped_area_sqm` | float | 0 | — | square metres | Total area clipped to parcel boundary |
| `{class}_total_unclipped_area_sqm` | `building_total_unclipped_area_sqm` | float | 0 | — | square metres | Total unclipped feature area |
| `{class}_fidelity` | `roof_fidelity` | float | 0.0 | 1.0 | — | Quality of the shape of the vectorized footprint polygon (only for structural classes — building, roof) |

### Primary Feature Columns

The **largest intersection area with parcel** feature of each class has additional `primary_` columns.

**How the primary feature is selected (largest_intersection):** Selects the feature with the largest area in the parcel.

| Pattern | Example | Type | Min | Max | Unit | Description |
|---------|---------|------|-----|-----|------|-------------|
| `primary_{class}_area_sqm` | `primary_building_area_sqm` | float | 0 | — | square metres | Area of primary feature |
| `primary_{class}_clipped_area_sqm` | `primary_building_clipped_area_sqm` | float | 0 | — | square metres | Clipped area of primary feature |
| `primary_{class}_confidence` | `primary_building_confidence` | float (quantised uint8) | 0.0 | 1.0 | — | Calibrated confidence measuring likelihood that the primary feature exists |
| `primary_{class}_feature_id` | `primary_building_feature_id` | string | — | — | — | Unique ID of primary feature (does not persist across surveys) |
| `primary_{class}_fidelity` | `primary_roof_fidelity` | float | 0.0 | 1.0 | — | Quality of the shape of the vectorized footprint polygon (only for structural classes — building, roof) |

## Common Columns

These columns appear in most output files:

| Column | Type | Min | Max | Unit | Example | Description |
|--------|------|-----|-----|------|---------|-------------|
| `mesh_date` | date | — | — | — | 2019-09-27 | Date of the 3D mesh used for 3D attributes (height, pitch, etc.). The mesh is usually comprised of a number of overlapping surveys captured at a similar time of year, so this date represents that capture window rather than a single survey. |
| `aoi_id` | string \| int | — | — | — | — | Property identifier. If the input file had an `aoi_id` column it is preserved; otherwise generated from row index (0-indexed). |
| `system_version` | string | — | — | gen<N>-<release_name>-<api_version> | — | Base AI model version used for detection (applies to all columns with source = 'base ai model'). Example: `gen6-glowing_moon-1.0`. See https://help.nearmap.com/kb/articles/1629-generation-6-ai-content for details. |
| `link` | URL | — | — | — | — | URL to view this row in Nearmap MapBrowser. Per-class files (roof, building, etc.) frame the URL on each feature's centroid; the rollup frames it on the parcel. |

## Address & Query Columns

These columns are present based on the query mode used:

| Column | Type | Min | Max | Unit | Example | Description |
|--------|------|-----|-----|------|---------|-------------|
| `query_aoi_lat` | float | -90.0 | 90.0 | degrees | — | Representative latitude of the query AOI geometry. |
| `query_aoi_lon` | float | -180.0 | 180.0 | degrees | — | Representative longitude of the query AOI geometry. |

## Data Notes

- **Coordinates**: WGS84 (EPSG:4326)
- **Areas**: Square meters (sqm)
- **Confidence scores**: Range from 0.0 (low) to 1.0 (high)
- **Dates**: ISO 8601 format (YYYY-MM-DD)
- **GeoParquet files**: Can be opened with QGIS, ArcGIS, or Python (geopandas)
- **`*_skipped` columns**: The Feature API may return `{skipped: true}` for an include section (e.g. RSI, peril scores, defensible space) when it decides not to evaluate that section for a given row. nmaipy surfaces this as a per-include `*_skipped` boolean — `Y` means "API declined to evaluate", `N` means evaluated normally, absent means the include wasn't requested. When a row has `*_skipped = Y`, the corresponding score / area / ratio columns will be null because the API didn't compute them, not because the underlying feature is absent.


---

*Generated by nmaipy*
