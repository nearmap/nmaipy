#!/usr/bin/env python
"""
Simple Python example for extracting Nearmap AI data.

This script shows data scientists how to use nmaipy to extract
building footprints, vegetation, and other AI features.
"""

import os
os.environ['API_KEY'] = 'your_api_key_here'  # Or set in your environment

from nmaipy.exporter import AOIExporter

# Create an exporter with your parameters
exporter = AOIExporter(
    # Input file with your areas of interest
    aoi_file='data/example_parcels.geojson',  # Can be GeoJSON, Parquet, or CSV
    
    # Where to save the results
    output_dir='data/outputs',
    
    # Which country (au, us, nz, ca)
    country='au',
    
    # What AI features to extract
    packs=['building', 'vegetation'],  # Options: building, vegetation, surfaces, pools, damage, etc.
    
    # Number of parallel processes (speed things up!)
    processes=4,
    
    # Optional: Save individual features, not just summaries
    save_features=True,
    
    # Optional: Include the AOI boundaries in output
    include_parcel_geometry=True,
)

# Run the extraction
if __name__ == "__main__":
    print("Extracting Nearmap AI features...")
    exporter.export()
    print("Done! Check the output directory for results.")