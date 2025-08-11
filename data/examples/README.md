# Example GeoJSON Files

These are small test GeoJSON files for testing nmaipy functionality. They contain example parcels in areas where Nearmap has coverage.

## Files

- **sydney_parcels.geojson** - Small parcels in Sydney CBD, Australia
- **us_parcels.geojson** - Parcels in Austin, Texas, USA  
- **large_area.geojson** - A 2km x 2km area in Melbourne (triggers automatic gridding)

## Usage

These files are referenced in `examples.py` and `run.py` to demonstrate nmaipy functionality.

```python
from nmaipy.exporter import AOIExporter

exporter = AOIExporter(
    aoi_file='data/examples/sydney_parcels.geojson',
    output_dir='data/outputs',
    country='au',
    packs=['building', 'vegetation'],
    processes=4
)
exporter.run()
```

## Note

These are small test areas designed to run quickly. For production use, you would typically use your own parcel/AOI files.