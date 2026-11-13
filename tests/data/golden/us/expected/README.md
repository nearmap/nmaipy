# Nearmap AI Export

This folder contains AI-generated property data from Nearmap aerial imagery.


---

## Files in This Export

| File Name | Description |
|-----------|-------------|
| `building.csv` | Per-building data with feature attributes |
| `building_features.parquet` | Building polygons with geometry (GeoParquet) |
| `classes_availability.json` | Available feature classes per AOI |
| `feature_api_errors.csv` | Properties where Feature API calls failed |
| `feature_api_errors.parquet` | Properties where Feature API calls failed |
| `features.parquet` | All detected features with geometry (GeoParquet) |
| `latency_stats.csv` | API call timing statistics |
| `rollup.csv` | Property-level summary with one row per parcel containing aggregated statistics |
| `roof.csv` | Per-roof data with feature attributes |
| `roof_age_errors.csv` | Properties where Roof Age API calls failed |
| `roof_age_errors.parquet` | Properties where Roof Age API calls failed |
| `roof_features.parquet` | Roof polygons with geometry (GeoParquet) |
| `roof_instance.csv` | Per-roof age instance data with feature attributes |
| `roof_instance_features.parquet` | Roof age instance polygons with geometry (GeoParquet) |
| `swimming_pool.csv` | Per-swimming pool data with feature attributes |
| `swimming_pool_features.parquet` | Swimming pool polygons with geometry (GeoParquet) |

## Feature Classes in This Export

Column names use snake_case versions of class names:

| Class Name | Column Prefix |
|------------|---------------|
| Building | `building_` |
| Roof | `roof_` |
| Roof Instance | `roof_instance_` |
| Swimming Pool | `swimming_pool_` |

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
| `{class}_total_area_sqft` | `building_total_area_sqft` | float | 0 | — | square feet | Total area of all features |
| `{class}_total_clipped_area_sqft` | `building_total_clipped_area_sqft` | float | 0 | — | square feet | Total area clipped to parcel boundary |
| `{class}_total_unclipped_area_sqft` | `building_total_unclipped_area_sqft` | float | 0 | — | square feet | Total unclipped feature area |
| `{class}_fidelity` | `roof_fidelity` | float | 0.0 | 1.0 | — | Quality of the shape of the vectorized footprint polygon (only for structural classes — building, roof) |

### Primary Feature Columns

The **largest intersection area with parcel** feature of each class has additional `primary_` columns.

**How the primary feature is selected (largest_intersection):** Selects the feature with the largest area in the parcel.

| Pattern | Example | Type | Min | Max | Unit | Description |
|---------|---------|------|-----|-----|------|-------------|
| `primary_{class}_area_sqft` | `primary_building_area_sqft` | float | 0 | — | square feet | Area of primary feature |
| `primary_{class}_clipped_area_sqft` | `primary_building_clipped_area_sqft` | float | 0 | — | square feet | Clipped area of primary feature |
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

## Dominant Roof Material & Shape Columns

These columns summarise the dominant roof material and shape for the primary roof.

**How dominant is computed.** Each roof is decomposed into material and shape components (each with a class, area, and ratio of the roof's total area). The **dominant** component is the one with the highest ratio. For **materials**, the dominant material is reported only when its ratio is at least 0.5; below that threshold the column is `unknown`. For **shapes**, the same "highest ratio wins" rule applies but no ratio threshold is enforced — a shape is dominant if it has the highest ratio *and* a non-zero area; otherwise `unknown`. (Shape ratios can overlap or gap and do not sum to 1, which is why no shape ratio column is emitted.) Deprecated `flat (deprecated)` and `shed` shape classes are excluded from selection. The description is null when a component of that attribute cannot be computed.

### Dominant Material

| Column | Type | Min | Max | Unit | Example | Description |
|--------|------|-----|-----|------|---------|-------------|
| `dominant_roof_material_feature_class` | UUID | — | — | — | — | Feature class UUID of the dominant roof material. Dominant = highest-ratio component; UNKNOWN (null) when ratio < 0.5. |
| `dominant_roof_material_description` | string | — | — | — | — | Snake_case name of the dominant material feature class (`unknown` when ratio < 0.5; null when a material component cannot be computed). Open-ended; consult the Feature API class registry for the full list of possible values. |
| `dominant_roof_material_area_sqft` | float | 0 | — | square feet | — | Area of the dominant material component on the primary roof. |
| `dominant_roof_material_ratio` | float | 0.0 | 1.0 | — | — | Ratio of the dominant material relative to total roof area. |
| `dominant_roof_material_confidence` | float (quantised uint8) | 0.0 | 1.0 | — | — | Confidence in the dominant material classification. |

If no single material has a ratio >= 0.5, the dominant material is reported as `unknown` with null statistics.

### Dominant Shape

| Column | Type | Min | Max | Unit | Example | Description |
|--------|------|-----|-----|------|---------|-------------|
| `dominant_roof_types_feature_class` | UUID | — | — | — | — | Feature class UUID of the dominant roof shape. Dominant = highest-ratio component; UNKNOWN (null) when no detected area. |
| `dominant_roof_types_description` | string | — | — | — | — | Snake_case name of the dominant shape feature class (`unknown` when no detected area; null when a shape component cannot be computed). Open-ended; consult the Feature API class registry for the full list of possible values. Excludes deprecated flat roof and `shed`. |
| `dominant_roof_types_area_sqft` | float | 0 | — | square feet | — | Area of the dominant shape component on the primary roof. |
| `dominant_roof_types_confidence` | float (quantised uint8) | 0.0 | 1.0 | — | — | Confidence in the dominant shape classification. |

If all shape components have zero area, the dominant shape is reported as `unknown` with null statistics.
Shape does not include a ratio column because roof shapes can overlap or gap, so ratios do not sum to 1.

## Roof Age Columns

These columns are present when roof age estimation was requested:

**Dataset and historical query for this export:**

- `roof_age_dataset`: `latest`
- `untilAsOfDate` cutoff: `2025-03-18` (per-AOI `until` column values, when present, override this for each row)

Note: cutoff parameters are not supported on the A.0 dataset. The `model_version` column in each row records the actual model that produced that record — for resilient downstream code, prefer `model_version` over the dataset alias when reasoning about model behaviour.

| Column | Type | Min | Max | Unit | Example | Description |
|--------|------|-----|-----|------|---------|-------------|
| `roof_age_installation_date` | date | — | — | — | — | Estimated roof installation date. |
| `roof_age_as_of_date` | date | — | — | — | — | Reference date for age calculation. When --until is set, this is the requested cutoff (clamped to the dataset's nearest snapshot date). |
| `roof_age_until_date` | date | — | — | — | — | The original requested cutoff (`untilAsOfDate`) before snapshot clamping. Differs from `roof_age_as_of_date` when the requested cutoff fell between dataset snapshots; null when no historical cutoff was applied. |
| `roof_age_years_as_of_date` | float | 0 | — | years | — | Calculated roof age in years (as_of_date − installation_date). |
| `roof_age_trust_score` | int | 0 | 100 | — | — | Reliability of age estimate (higher = more reliable). |
| `roof_age_evidence_type` | int | 0 | 8 | — | — | How age was determined (numeric code; see evidence_type_legend in README.md). |
| `roof_age_evidence_type_description` | string | — | — | — | — | Human-readable evidence type description. |
| `roof_age_before_installation_capture_date` | date | — | — | — | — | Last capture before roof installation. |
| `roof_age_after_installation_capture_date` | date | — | — | — | — | First capture after roof installation. |
| `roof_age_min_capture_date` | date | — | — | — | — | Earliest imagery capture date analyzed. |
| `roof_age_max_capture_date` | date | — | — | — | — | Latest imagery capture date analyzed. |
| `roof_age_number_of_captures` | int | 0 | — | — | — | Number of images analyzed. |
| `roof_age_map_browser_url` | URL | — | — | — | — | URL to view roof age timeline in MapBrowser. |
| `roof_age_model_version` | string | A.0 | — | <major>.<minor> | — | Version of the roof age prediction model used. |
| `roof_age_kind` | string | — | — | — | — | Type of estimate. One of `roof` (roof section) or `parcel` (property-level). Parcel-level estimates are only provided when no imagery is available. |
| `roof_age_relevant_permits` | binary | N | Y | — | — | Whether relevant building permits were used in determining roof age. |
| `roof_age_relevant_permits_details` | JSON | — | — | — | — | Details of relevant building permits (present when permits used in determining roof age). |
| `roof_age_assessor_data` | binary | N | Y | — | — | Whether assessor records (year built) were used in determining roof age. |
| `roof_age_assessor_data_details` | JSON | — | — | — | — | Details of assessor (year built) records (present when records were used in determining roof age). |

**Evidence Type code → description:**

- **Type 0**: No unobscured images of the roof or parcel and no relevant building permits or assessor year-built data. No roof installation date is provided.
- **Type 1**: (Future Release) No unobscured images and no relevant permits/assessor data, but aggregated supporting evidence from the area indicates the likely age of roofs nearby.
- **Type 2**: No unobscured images of the roof or parcel, but relevant building permits and/or assessor year-built data provide an indication of the installation date.
- **Type 3**: A single unobscured image of the roof. Roof change cannot be detected; a machine-learning model predicts the age from the image plus other sources (e.g. climate data). No supporting evidence corroborating the date.
- **Type 4**: A single unobscured image of the roof (ML-predicted age, as in Type 3), with relevant building permits and/or assessor year-built data supporting the predicted installation date.
- **Type 5**: Multiple unobscured images of the roof, no roof change detected between captures. A separate ML model predicts the age from imagery plus other sources. No supporting evidence corroborating the date.
- **Type 6**: Multiple unobscured images of the roof, no roof change detected (ML-predicted age, as in Type 5), with relevant building permits and/or assessor year-built data supporting the predicted installation date.
- **Type 7**: Multiple unobscured images of the roof. AI model detects a roof change between two captures; installation date is set as the mid-point of those captures. No supporting evidence corroborating the date.
- **Type 8**: Multiple unobscured images of the roof, with detected roof change (mid-point installation date, as in Type 7), with relevant building permits and/or assessor year-built data supporting the predicted installation date.

Higher evidence types indicate more robust information sources. **Trust Score (0-100)** quantifies reliability based on the same evidence quality factors (imagery, permits, change detection, capture count).

For more details, see:
- https://help.nearmap.com/kb/articles/1810-nearmap-roof-age
- https://help.nearmap.com/kb/articles/1811-evidence-type-and-trust-score

## Data Notes

- **Coordinates**: WGS84 (EPSG:4326)
- **Areas**: Square feet (sqft)
- **Confidence scores**: Range from 0.0 (low) to 1.0 (high)
- **Dates**: ISO 8601 format (YYYY-MM-DD)
- **GeoParquet files**: Can be opened with QGIS, ArcGIS, or Python (geopandas)
- **`*_skipped` columns**: The Feature API may return `{skipped: true}` for an include section (e.g. RSI, peril scores, defensible space) when it decides not to evaluate that section for a given row. nmaipy surfaces this as a per-include `*_skipped` boolean — `Y` means "API declined to evaluate", `N` means evaluated normally, absent means the include wasn't requested. When a row has `*_skipped = Y`, the corresponding score / area / ratio columns will be null because the API didn't compute them, not because the underlying feature is absent.


---

*Generated by nmaipy*
