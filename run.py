#!/usr/bin/env python
"""
Simple Python example for extracting Nearmap AI data.

This script shows data scientists how to use nmaipy to extract:
1. Building footprints, vegetation, and other AI features (Feature API)
2. Roof age predictions (Roof Age API - US only)
"""

import os
from nmaipy.exporter import AOIExporter
from nmaipy.roof_age_exporter import RoofAgeExporter

# Example 1: Extract AI features using Feature API
print("=" * 60)
print("Example 1: Extracting AI Features (Feature API)")
print("=" * 60)

feature_exporter = AOIExporter(
    aoi_file='tests/data/test_parcels_2.csv',  # New Jersey parcels (US)
    output_dir='data/outputs/features',
    country='us',
    packs=['building', 'vegetation'],  # Options: building, vegetation, surfaces, pools, damage, etc.
    processes=2,
    save_features=True,
    include_parcel_geometry=True,
)

roof_age_exporter = RoofAgeExporter(
    aoi_file='tests/data/test_parcels_2.csv',  # Same New Jersey parcels
    output_dir='data/outputs/roof_age',
    country='us',
    processes=2,  # Number of parallel processes (matches feature API)
    output_format='both',  # Output both parquet and CSV
)

# Run the feature extraction
if __name__ == "__main__":
    print("\nExtracting Nearmap AI features...")
    feature_exporter.run()
    print("✓ Feature extraction complete! Check data/outputs/features/\n")

    # Example 2: Extract roof age predictions using Roof Age API
    print("=" * 60)
    print("Example 2: Extracting Roof Age Data (Roof Age API - US Only)")
    print("=" * 60)

    print("\nExtracting roof age predictions...")
    roof_age_exporter.run()
    print("✓ Roof age extraction complete! Check data/outputs/roof_age/\n")

    print("=" * 60)
    print("All extractions complete!")
    print("=" * 60)
    print("\nOutput locations:")
    print("  - AI Features: data/outputs/features/")
    print("  - Roof Ages:   data/outputs/roof_age/")
