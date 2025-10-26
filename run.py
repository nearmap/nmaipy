#!/usr/bin/env python
"""
Simple Python example for extracting Nearmap AI data.

This script shows data scientists how to use nmaipy to extract
building footprints, vegetation, and other AI features.
"""

import os
from nmaipy.exporter import AOIExporter

# Create an exporter with your parameters
exporter = AOIExporter(
    aoi_file='data/examples/sydney_parcels.geojson',  # Can be GeoJSON, Parquet, or CSV
    output_dir='data/outputs',
    country='au',
    packs=['building', 'vegetation'],  # Options: building, vegetation, surfaces, pools, damage, etc.
    processes=4,
    save_features=True,
    include_parcel_geometry=True,
)

# Run the extraction
if __name__ == "__main__":
    print("Extracting Nearmap AI features...")
    exporter.run()
    print("Done! Check the output directory for results.")