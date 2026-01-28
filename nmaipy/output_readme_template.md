# Nearmap AI Export Output Files

This folder contains outputs from the Nearmap AI Feature Exporter (`nmaipy`). Below is a guide to each file and what it contains.

> **Note**: File names are based on your input file name. Examples below assume an input file called `properties.geojson`.

---

## Export Configuration

#### `export_config.json`

**What it is**: Auto-generated configuration file capturing all parameters used for this export. Created at the start of the export, so it's available even if the export fails.

**Use this for**: Reproducibility, debugging, auditing export settings.

```json
{
  "_metadata": {
    "export_started_at": "2024-01-15T10:30:00+00:00",
    "nmaipy_version": "4.0.0",
    "python_version": "3.11.5",
    "platform": "macOS-14.0-arm64"
  },
  "parameters": {
    "aoi_file": "/path/to/properties.geojson",
    "packs": ["building", "roof"],
    "country": "us",
    "save_features": true,
    "roof_age": true,
    ...
  }
}
```

| Field | Description |
|-------|-------------|
| `_metadata.export_started_at` | UTC timestamp when export began |
| `_metadata.nmaipy_version` | Version of nmaipy used |
| `_metadata.python_version` | Python interpreter version |
| `_metadata.platform` | Operating system and architecture |
| `parameters.*` | All command-line arguments and their values |

#### `roof_age_export_config.json`

**What it is**: Configuration file for standalone Roof Age API exports (when using `roof_age_exporter.py` directly).

---

## Output Files Reference

### Main Rollup File

#### `properties_aoi_rollup.csv` (or `.parquet`)

**What it is**: The primary output file with one row per property/AOI containing aggregated statistics for all detected features.

**Use this for**: Property-level analysis, insurance underwriting, portfolio assessment.

| Column | Description |
|--------|-------------|
| `aoi_id` | Your original property identifier |
| `mesh_date` | Date of the aerial imagery used (YYYY-MM-DD) |
| `system_version` | AI model version that produced these results |
| `{class}_present` | Whether this feature was detected: `Y` or `N` |
| `{class}_count` | Number of distinct features of this class |
| `{class}_total_area_sqm` | Total area in square meters (or `_sqft` for US) |
| `{class}_total_clipped_area_sqm` | Area clipped to property boundary |
| `{class}_total_unclipped_area_sqm` | Full feature area (may extend beyond property) |
| `{class}_confidence` | Combined confidence: `1 - product(1 - individual_confidences)` |
| `primary_{class}_*` | Attributes of the largest feature (see below) |
| `geometry` | Property boundary (if `--include-parcel-geometry` was used) |

**Primary feature columns** (for the largest feature of each class):
- `primary_{class}_area_sqm` / `_sqft` - Area of the primary feature
- `primary_{class}_clipped_area_sqm` - Area within property boundary
- `primary_{class}_confidence` - Confidence score (0-1)
- `primary_{class}_feature_id` - Unique feature identifier
- Plus class-specific attributes (materials, heights, etc.)

---

### Feature-Level Files

#### `properties_features.parquet`

**What it is**: GeoParquet file containing every individual AI-detected feature with full geometry. Created when `--save-features` flag is used.

**Use this for**: GIS mapping, spatial analysis, detailed feature inspection.

**Format**: GeoParquet (open with QGIS, ArcGIS, or Python geopandas)

| Column | Description |
|--------|-------------|
| `aoi_id` | Property this feature belongs to |
| `feature_id` | Unique identifier for this feature |
| `class_id` | Feature classification UUID (see Class IDs below) |
| `description` | Human-readable class name (e.g., "Roof", "Swimming pool") |
| `confidence` | Model confidence (0.0-1.0, higher is better) |
| `area_sqm` / `area_sqft` | Feature area |
| `clipped_area_sqm` | Area within property boundary |
| `unclipped_area_sqm` | Total feature area |
| `survey_date` | When imagery was captured |
| `mesh_date` | Processing date |
| `geometry` | Feature polygon (WGS84 / EPSG:4326) |
| `attributes.*` | Flattened attribute fields (varies by class) |

---

### Building-Specific Files

#### `properties_buildings.csv` (or `.parquet`)

**What it is**: One row per building with detailed building-specific attributes. Created when `--save-buildings` flag is used.

**Use this for**: Building inventory, property assessment, construction analysis.

| Column | Description |
|--------|-------------|
| `aoi_id` | Property identifier |
| `feature_id` | Building feature ID |
| `confidence` | Detection confidence |
| `area_sqm` / `area_sqft` | Building footprint area |
| `clipped_area_sqm` | Area within property |
| `Building 3d attributes.height` | Building height in meters |
| `Building 3d attributes.numStories.*` | Story count probabilities |
| `Building style.*` | Architectural style classifications |

#### `properties_building_features.parquet`

**What it is**: Same as buildings file but includes geometry (GeoParquet format).

---

### Per-Class Rollup Files

These files provide class-specific summaries. One CSV per feature class detected in your data.

#### `properties_roof.csv`

**What it is**: One row per property with roof-specific aggregated data.

**Class ID**: `c08255a4-ba9f-562b-932c-ff76f2faeeeb`

| Column | Description |
|--------|-------------|
| `aoi_id` | Property identifier |
| `roof_present` | `Y` if roof detected |
| `roof_count` | Number of roof sections |
| `roof_total_area_sqm` | Total roof area |
| `primary_roof_area_sqm` | Largest roof section area |
| `primary_roof_confidence` | Confidence in primary roof detection |
| RSI columns | Roof Spotlight Index scores (condition indicators) |
| Material columns | Detected roof materials |
| Shape columns | Detected roof shapes (hip, gable, etc.) |
| 3D attribute columns | Height, pitch, number of stories |

#### `properties_roof_features.parquet`

**What it is**: GeoParquet with all individual roof polygons and their attributes.

---

#### `properties_building.csv`

**What it is**: One row per property with building footprint data.

**Class ID**: `1878ccf6-46ec-55a7-a20b-0cf658afb755` (Building - new semantic)

| Column | Description |
|--------|-------------|
| `aoi_id` | Property identifier |
| `building_present` | `Y` if building detected |
| `building_count` | Number of buildings |
| `building_total_area_sqm` | Total building footprint |
| `primary_building_*` | Attributes of largest building |

#### `properties_building_features.parquet`

**What it is**: GeoParquet with all building footprint polygons.

---

#### `properties_swimming_pool.csv`

**What it is**: Pool detection summary per property.

**Class ID**: `0339726f-081e-5a6e-b9a9-42d95c1b5c8a`

| Column | Description |
|--------|-------------|
| `swimming_pool_present` | `Y` if pool detected |
| `swimming_pool_count` | Number of pools |
| `swimming_pool_total_area_sqm` | Total pool surface area |

---

#### `properties_solar_panel.csv`

**What it is**: Solar panel detection summary per property.

**Class ID**: `3680e1b8-8ae1-5a15-8ec7-820078ef3298`

| Column | Description |
|--------|-------------|
| `solar_panel_present` | `Y` if solar panels detected |
| `solar_panel_count` | Number of panel arrays |
| `solar_panel_total_area_sqm` | Total panel coverage |

---

#### `properties_trampoline.csv`

**What it is**: Trampoline detection summary per property.

**Class ID**: `753621ee-0b9f-515e-9bcf-ea40b96612ab`

---

#### `properties_tree_overhang.csv`

**What it is**: Tree canopy overhanging structures, per property.

**Class ID**: `8e9448bd-4669-5f46-b8f0-840fee25c34c`

| Column | Description |
|--------|-------------|
| `tree_overhang_present` | `Y` if overhanging trees detected |
| `tree_overhang_total_area_sqm` | Area of canopy over structures |

---

#### Other Per-Class Files

Depending on which packs were requested, you may also see:

| File | Class ID | Description |
|------|----------|-------------|
| `properties_construction.csv` | `a2a81381-13c6-57dc-a967-af696e45f6c7` | Active construction sites |
| `properties_skylight.csv` | `3f5a737e-6d56-538a-ac26-f2934bbbb695` | Skylights on roofs |
| `properties_boat.csv` | `62a0958e-2139-5688-a776-b88c6049d50e` | Boats on property |
| `properties_car.csv` | `8337e0e1-e171-5292-89cc-99c0da2a4fe4` | Vehicles detected |
| `properties_debris.csv` | `8ab218a7-8173-5f1e-a5cb-bb2cd386a73e` | Roof debris |

Each has a corresponding `*_features.parquet` with geometries.

---

### Roof Age Files (US Only)

#### `properties_roof_instance.csv`

**What it is**: One row per roof section with installation date predictions. Created when `--roof-age` flag is used.

**Class ID**: `f00f1a9e-0001-4000-a000-000000000001` (synthetic class for Roof Age API)

| Column | Description |
|--------|-------------|
| `aoi_id` | Property identifier |
| `roof_age_kind` | `"roof"` (individual section) or `"parcel"` (property-level) |
| `roof_age_installation_date` | Estimated installation date (YYYY-MM-DD) |
| `roof_age_as_of_date` | Reference date for age calculation |
| `roof_age_years_as_of_date` | Calculated age in years (decimal) |
| `roof_age_trust_score` | Confidence in prediction (0.0-1.0) |
| `roof_age_area_sqm` | Area of this roof section |
| `roof_age_evidence_type` | How age was determined |
| `roof_age_evidence_type_description` | Human-readable explanation |
| `roof_age_before_installation_capture_date` | Last image before installation |
| `roof_age_after_installation_capture_date` | First image after installation |
| `roof_age_number_of_captures` | Number of aerial captures analyzed |
| `roof_age_mapbrowser_url` | Link to view in Nearmap MapBrowser |

#### `properties_roof_instance_features.parquet`

**What it is**: GeoParquet with roof age polygons and all attributes.

---

### Roof Condition Classes

If roof condition analysis is included, you may see files for:

| File | Class ID | Description |
|------|----------|-------------|
| `properties_temporary_repair.csv` | `abb1f304-ce01-527b-b799-cbfd07551b2c` | Tarps/temporary repairs |
| `properties_rust.csv` | `526496bf-7344-5024-82d7-77ceb671feb4` | Rust on metal roofs |
| `properties_missing_tiles_or_shingles.csv` | `dec855e2-ae6f-56b5-9cbb-f9967ff8ca12` | Missing roof covering |
| `properties_ponding.csv` | `f41e02b0-adc0-5b46-ac95-8c59aa9fe317` | Standing water on flat roofs |
| `properties_staining.csv` | `319f552f-f4b7-520d-9b16-c8abb394b043` | Staining/discoloration |
| `properties_worn_shingles.csv` | `97a6f930-82ae-55f2-b856-635e2250af29` | Worn/weathered shingles |
| `properties_patching.csv` | `8b30838b-af41-5d1d-bdbd-29e682fe3b00` | Patched areas |
| `properties_structural_damage.csv` | `c0224852-4310-57dd-95fe-42bff1c0a3f0` | Structural damage |

---

### Error Files

#### `properties_feature_api_errors.csv`

**What it is**: Records properties where the Feature API request failed.

| Column | Description |
|--------|-------------|
| `aoi_id` | Property that failed |
| `status_code` | HTTP status code |
| `message` | Error description |

**Common status codes**:
- `404` - No imagery coverage for this location
- `429` - Rate limit exceeded (retry later)
- `500` - Server error (retry later)

#### `properties_feature_api_errors.parquet`

**What it is**: Same as above but includes geometry of the failed area (useful for mapping coverage gaps).

#### `properties_roof_age_errors.csv` / `.parquet`

**What it is**: Records properties where Roof Age API requests failed (same columns as above).

---

## Understanding the Data

### Area Measurements

| Measurement | Use When |
|-------------|----------|
| `clipped_area` | Calculating areas within your property boundary |
| `unclipped_area` | Feature extends beyond property (e.g., shared roof) |
| `total_area` (rollups) | Sum of all features of this class on the property |
| `primary_*_area` | Just the largest feature |

### Confidence Scores

- **0.0 - 0.5**: Low confidence, verify manually
- **0.5 - 0.8**: Moderate confidence
- **0.8 - 1.0**: High confidence

**Rollup confidence** is calculated as `1 - product(1 - individual_confidences)`, so increases with more detected features.

### Trust Score (Roof Age)

The `roof_age_trust_score` indicates reliability of the age estimate:
- Based on evidence quality (permits, imagery change detection)
- Higher scores when multiple evidence sources agree
- Lower scores for older roofs with fewer historical images

---

## File Formats

| Extension | Description | Open With |
|-----------|-------------|-----------|
| `.csv` | Comma-separated values | Excel, Google Sheets, any text editor |
| `.parquet` | Columnar binary format | Python (pandas), R, Spark, DuckDB |
| `.parquet` (GeoParquet) | Parquet with geometry | QGIS, ArcGIS, Python (geopandas) |

**Tip**: GeoParquet files can be identified by having a `geometry` column and metadata indicating spatial data.

---

## Quick Start Analysis

1. **Property overview**: Open `properties_aoi_rollup.csv` in Excel
2. **Check coverage**: Review `properties_feature_api_errors.csv` for missing properties
3. **Map features**: Load `properties_features.parquet` in QGIS
4. **Roof analysis**: Use `properties_roof.csv` for condition/material breakdowns
5. **Risk assessment**: Combine roof age (`roof_instance.csv`) with condition data

---

## Additional Resources

- **nmaipy Documentation**: https://github.com/nearmap/nmaipy
- **Nearmap AI API Docs**: https://docs.nearmap.com/display/ND/AI+Feature+API
- **Nearmap Support**: https://www.nearmap.com/support

---

*Generated by nmaipy - Nearmap AI Python Library*
