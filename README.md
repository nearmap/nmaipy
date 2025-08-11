# nmaipy - Nearmap AI Python Library

Extract building footprints, vegetation, damage assessments, and other AI features from Nearmap's aerial imagery using simple Python code.

## What is nmaipy?

nmaipy (pronounced "en-my-pie") is a Python library that makes it easy for data scientists to access Nearmap's AI-powered geospatial data. Whether you're analyzing a few properties or processing millions of buildings across entire cities, nmaipy handles the complexity so you can focus on your analysis.

## Quick Start for Data Scientists

### 1. Install

#### Option A: Using pip
```bash
pip install -e .
```

#### Option B: Using conda

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

#### Option C: Install into existing conda environment
```bash
conda install -c conda-forge geopandas pandas numpy pyarrow psutil pyproj python-dotenv requests rtree shapely stringcase tqdm
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
from nmaipy.exporter import AOIExporter

# Extract building and vegetation data
exporter = AOIExporter(
    aoi_file='my_parcels.geojson',  # Your areas of interest
    output_dir='results',            # Where to save outputs
    country='au',                     # au, us, nz, or ca
    packs=['building', 'vegetation'], # What features to extract
    processes=4                       # Parallel processing
)

exporter.run()
```

That's it! Your results will be saved as CSV or Parquet files in the output directory.

## Common Use Cases

### 🏢 Urban Planning
Extract comprehensive data about buildings, vegetation coverage, and surface materials:

```python
exporter = AOIExporter(
    aoi_file='city_blocks.geojson',
    output_dir='urban_analysis',
    country='au',
    packs=['building', 'vegetation', 'surfaces', 'solar'],
    save_features=True,  # Get individual features, not just summaries
    include_parcel_geometry=True  # Keep boundaries for GIS analysis
)
```

### 🌊 Disaster Response
Assess damage after natural disasters like hurricanes or floods:

```python
exporter = AOIExporter(
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

### 🌳 Environmental Analysis
Study vegetation coverage and tree canopy:

```python
exporter = AOIExporter(
    aoi_file='study_area.geojson',
    output_dir='vegetation_study',
    country='au',
    packs=['vegetation'],
    save_features=True  # Get individual tree polygons
)
```

### 🏊 Market Research
Find properties with pools or solar panels:

```python
exporter = AOIExporter(
    aoi_file='suburbs.geojson',
    output_dir='market_analysis',
    country='au',
    packs=['pools', 'solar'],
    include_parcel_geometry=True
)
```

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
- **Parquet**: Efficient columnar format for large datasets  
- **CSV**: Simple format with lat/lon coordinates or WKT geometries

Your input file should contain polygon geometries representing the areas you want to analyze (parcels, census blocks, suburbs, etc.).

## Output Data

Results are saved as CSV or Parquet files containing:

- **Rollups**: Summary statistics per AOI (counts, areas, percentages)
- **Features**: Individual AI features with geometries (when `save_features=True`)
- **Metadata**: Survey dates, data quality metrics

## Examples

Check out `examples.py` for complete working examples of different use cases.

For a minimal example, see `run.py`.

## Working with Large Areas

nmaipy automatically handles large areas by:
- Splitting them into manageable grid cells
- Processing in parallel
- Combining results seamlessly

For areas larger than 1 sq km, the library will automatically use gridding:

```python
exporter = AOIExporter(
    aoi_file='large_region.geojson',
    output_dir='large_area_results',
    country='us',
    packs=['building'],
    aoi_grid_inexact=True,  # Allow mixing survey dates if needed
    processes=16  # Use more processes for speed
)
```

## Performance Tips

1. **Use parallel processing**: Set `processes` to the number of CPU cores
2. **Process in chunks**: Use `chunk_size` for very large datasets
3. **Cache results**: Reuse cached API responses with `cache_dir`
4. **Filter by date**: Use `since` and `until` to get specific time periods

## API Documentation

For detailed API documentation and advanced options, see the [API Reference](docs/api.md).

## Getting Help

- **Examples**: See `examples.py` for common use cases
- **Notebooks**: Check the `notebooks/` directory for Jupyter notebook tutorials
- **Issues**: Report bugs or request features on [GitHub](https://github.com/nearmap/nmaipy)

## Requirements

- Python 3.11+
- Nearmap API key (contact Nearmap for access)
- 4GB+ RAM recommended for large extractions

## Advanced: Building a Conda Package

For system administrators who want to create a local conda package:

```bash
conda build conda.recipe
conda install --use-local nmaipy
```

This will create a conda package that can be shared internally or uploaded to a conda channel.

## License

See LICENSE file for details.