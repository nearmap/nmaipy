# nmaipy - Nearmap AI Python Library

Extract building footprints, vegetation, damage assessments, and other AI features from Nearmap's aerial imagery using simple Python code.

## What is nmaipy?

nmaipy (pronounced "en-my-pie") is a Python library that makes it easy for data scientists to access Nearmap's AI-powered geospatial data. Whether you're analyzing a few properties or processing millions of buildings across entire cities, nmaipy handles the complexity so you can focus on your analysis.

**Supported countries:** `au` (Australia), `us` (United States), `nz` (New Zealand), `ca` (Canada)

## Quick Start for Data Scientists

### 1. Install

#### Option A: Install from PyPI
```bash
pip install nmaipy
```

#### Option B: Install from source (for development)
```bash
git clone https://github.com/nearmap/nmaipy.git
cd nmaipy
pip install -e .
```

#### Option C: Using conda

Minimal installation (core features only):
```bash
conda env create -f environment-minimal.yaml
conda activate nmaipy
```

Full installation (includes development and notebook tools):
```bash
conda env create -f environment.yaml
conda activate nmaipy
```

#### Option D: Install into existing conda environment
```bash
conda install -c conda-forge geopandas pandas numpy pyarrow psutil pyproj python-dotenv requests rtree shapely stringcase tqdm fsspec s3fs
pip install -e .
```

#### Additional options

For running notebooks with pip:
```bash
pip install -e ".[notebooks]"
```

For development with pip:
```bash
pip install -e ".[dev]"
```

### 2. Set your API key

```bash
export API_KEY=your_api_key_here
```

### 3. Run your first extraction

```python
from nmaipy.exporter import NearmapAIExporter

# Extract building and vegetation data
exporter = NearmapAIExporter(
    aoi_file='my_parcels.geojson',  # Your areas of interest
    output_dir='results',            # Where to save outputs
    country='au',                     # au, us, nz, or ca
    packs=['building', 'vegetation'], # What features to extract
    processes=4                       # Parallel processing
)

exporter.run()
```

That's it! Your results will be saved as CSV or Parquet files in the output directory.

> **Note:** `AOIExporter` is available as a backward-compatible alias for `NearmapAIExporter`.

## Common Use Cases

### Urban Planning
Extract comprehensive data about buildings, vegetation coverage, and surface materials:

```python
exporter = NearmapAIExporter(
    aoi_file='city_blocks.geojson',
    output_dir='urban_analysis',
    country='au',
    packs=['building', 'vegetation', 'surfaces', 'solar'],
    save_features=True,  # Get individual features, not just summaries
    include_parcel_geometry=True  # Keep boundaries for GIS analysis
)
```

### Disaster Response
Assess damage after natural disasters like hurricanes or floods:

```python
exporter = NearmapAIExporter(
    aoi_file='affected_areas.geojson',
    output_dir='damage_assessment',
    country='us',
    packs=['damage'],
    since='2024-07-08',  # Date range of the event
    until='2024-07-11',
    rapid=True,  # Use rapid post-catastrophe imagery
    save_features=True
)
```

### Environmental Analysis
Study vegetation coverage and tree canopy:

```python
exporter = NearmapAIExporter(
    aoi_file='study_area.geojson',
    output_dir='vegetation_study',
    country='au',
    packs=['vegetation'],
    save_features=True  # Get individual tree polygons
)
```

### Market Research
Find properties with pools or solar panels:

```python
exporter = NearmapAIExporter(
    aoi_file='suburbs.geojson',
    output_dir='market_analysis',
    country='au',
    packs=['pools', 'solar'],
    include_parcel_geometry=True
)
```

### Roof Age Analysis (US Only)
Predict roof installation dates using AI analysis of historical imagery.

**Unified approach** (recommended) - combines Feature API and Roof Age in one export:

```python
exporter = NearmapAIExporter(
    aoi_file='properties.geojson',
    output_dir='unified_results',
    country='us',
    packs=['building'],
    roof_age=True,  # Include Roof Age API data
    save_features=True
)
exporter.run()
```

**Standalone approach** - for roof age data only:

```python
from nmaipy.roof_age_exporter import RoofAgeExporter

exporter = RoofAgeExporter(
    aoi_file='properties.geojson',
    output_dir='roof_age_results',
    country='us',
    threads=10,
    output_format='both'  # Generate both GeoParquet and CSV
)
exporter.run()
```

The Roof Age API uses machine learning to analyze multiple imagery captures over time, combined with building permit data and climate information, to predict when roofs were last installed or significantly renovated. Each roof feature includes:
- Predicted installation date
- Confidence score (trust score)
- Evidence type and number of captures analyzed
- Timeline of all imagery used in analysis

This is valuable for:
- Insurance underwriting and risk assessment
- Property valuation and market analysis
- Maintenance planning and capital budgeting
- Real estate due diligence

## Available AI Features

Some of the more common AI packs are below - there are more and growing, available via API request or on the Nearmap help.nearmap.com page.

| Pack | Description | Example Use Cases |
|------|-------------|-------------------|
| `building` | Building footprints and heights | Urban planning, property analysis |
| `vegetation` | Trees and vegetation coverage | Environmental studies, urban forestry |
| `surfaces` | Ground surface materials | Permeability studies, heat mapping |
| `pools` | Swimming pool detection | Compliance, market research |
| `solar` | Solar panel detection | Renewable energy assessment |
| `damage` | Post-disaster damage classification | Insurance, emergency response |
| `building_characteristics` | Detailed roof types, materials | Detailed property analysis |

## Input Data Formats

nmaipy accepts areas of interest (AOIs) in several formats:

- **GeoJSON**: Standard geospatial format with polygons
- **GeoPackage** (GPKG): OGC standard for geospatial data
- **Parquet / GeoParquet**: Efficient columnar format for large datasets
- **CSV**: Simple format with WKT geometries (also supports TSV and PSV)

Your input file should contain polygon geometries representing the areas you want to analyze (parcels, census blocks, suburbs, etc.).

## Output Data

The exporter writes results to `{output_dir}/final/` with the following structure:

| File | Description |
|------|-------------|
| `{stem}_aoi_rollup.csv` or `.parquet` | One row per AOI with summary statistics (counts, areas, confidences) |
| `{stem}_{class}.csv` | Per-class attribute tables (e.g. `roof.csv`, `building.csv`) |
| `{stem}_{class}_features.parquet` | Per-class GeoParquet with feature geometries (when `save_features=True`) |
| `{stem}_features.parquet` | All features combined as GeoParquet (when `save_features=True`) |
| `{stem}_buildings.csv` or `.parquet` | Per-building detail rows (when `save_buildings=True`) |
| `{stem}_feature_api_errors.csv` | AOIs where the Feature API returned errors |
| `{stem}_roof_age_errors.csv` | AOIs where the Roof Age API returned errors (US only) |
| `{stem}_latency_stats.csv` | API timing diagnostics |
| `export_config.json` | Full record of export parameters and nmaipy version |
| `README.md` | Auto-generated data dictionary describing all output files and columns |

A `{output_dir}/chunks/` directory holds intermediate per-chunk results during processing, enabling resume after interruption.

For detailed column-level documentation, refer to the auto-generated `README.md` inside each export's `final/` directory.

## S3 Output Support

nmaipy can write output directly to Amazon S3. Pass an `s3://` URI as the output directory:

```python
exporter = NearmapAIExporter(
    aoi_file='properties.geojson',
    output_dir='s3://my-bucket/nmaipy-results/',
    country='us',
    packs=['building'],
)
exporter.run()
```

AWS credentials are resolved automatically from environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) or `~/.aws/credentials`. No additional nmaipy configuration is needed.

The `cache_dir` parameter also accepts S3 URIs for cloud-native workflows, though local caching is faster for iterative development.

## Examples

**Quick start** — verify your setup with 10 US properties covering buildings, features, and roof age:
```bash
export API_KEY=your_api_key_here
python run_10_test.py
```

**More examples** — see `examples.py` for complete, working examples covering:
- Basic building/vegetation extraction
- Damage assessment (Hurricane Beryl)
- Urban planning (multi-pack)
- Vegetation analysis
- Pool detection
- Large area gridding
- Time series extraction
- Unified roof age + feature export

Example AOI files are provided in `data/examples/`:
- `sydney_parcels.geojson` — Sydney CBD, Australia
- `us_parcels.geojson` — Austin, Texas, USA
- `large_area.geojson` — 2km x 2km Melbourne area (triggers auto-gridding)

## Working with Large Areas

nmaipy automatically handles large areas by:
- Splitting them into manageable grid cells
- Processing in parallel
- Combining results seamlessly

For areas larger than 1 sq km, the library will automatically use gridding:

```python
exporter = NearmapAIExporter(
    aoi_file='large_region.geojson',
    output_dir='large_area_results',
    country='us',
    packs=['building'],
    aoi_grid_inexact=True,  # Allow mixing survey dates if needed
    processes=16  # Use more processes for speed
)
```

## Performance Tips

1. **Use parallel processing**: Set `processes` to the number of CPU cores available.
2. **Tune chunk size**: `chunk_size` controls how many AOIs are grouped into each parallel work unit (default: 500). Smaller values give finer-grained parallelism and cheaper resume after interruption; larger values reduce overhead.
3. **Cache API responses**: Use `cache_dir` to persist API responses to a directory. On subsequent runs with different parameters (e.g. different packs), cached responses are reused without re-fetching. By default, cache is stored in `{output_dir}/cache/`.
4. **Filter by date**: Use `since` and `until` to restrict to specific time periods, reducing data volume.

## Command Line Interface

### Feature API Export

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
- `--packs`: AI packs to extract (building, vegetation, surfaces, pools, solar, damage, etc.)
- `--roof-age`: Include Roof Age API data (US only)
- `--save-features`: Save per-class GeoParquet files with feature geometries
- `--save-buildings`: Save per-building detail rows
- `--rollup-format`: Output format for rollup file (`csv` or `parquet`, default: `csv`)
- `--cache-dir`: Directory for caching API responses
- `--no-cache`: Disable caching entirely
- `--primary-decision`: Feature selection method (`largest_intersection`, `nearest`, `optimal`)
- `--since` / `--until`: Filter by survey date range
- `--max-retries`: Maximum API retry attempts (default: 10)

Run `python nmaipy/exporter.py --help` for all options.

### Standalone Roof Age Export (US Only)

```bash
python -m nmaipy.roof_age_exporter \
    --aoi-file "us_properties.geojson" \
    --output-dir "roof_age_results" \
    --country us \
    --processes 4 \
    --output-format both
```

Run `python -m nmaipy.roof_age_exporter --help` for all options.

## Getting Help

- **Examples**: See `examples.py` for common use cases
- **Installation**: See `INSTALL.md` for detailed installation options
- **Notebooks**: Check the `notebooks/` directory for Jupyter notebook tutorials
- **Issues**: Report bugs or request features on [GitHub](https://github.com/nearmap/nmaipy)

## Requirements

- Python 3.12+
- Nearmap API key (contact Nearmap for access)
- 4GB+ RAM recommended for large extractions
- AWS credentials for S3 output (optional)

## Advanced: Building a Conda Package

For system administrators who want to create a local conda package:

```bash
conda build conda.recipe
conda install --use-local nmaipy
```

This will create a conda package that can be shared internally or uploaded to a conda channel.

## License

See LICENSE file for details.
