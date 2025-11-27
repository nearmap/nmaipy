#!/usr/bin/env python
"""
Simple Python example for extracting Nearmap AI data.

This script shows data scientists how to use nmaipy to extract:
1. Building footprints, vegetation, and other AI features (Feature API)
2. Roof age predictions (Roof Age API - US only)

The NearmapAIExporter provides a unified interface for both APIs.
"""

from nmaipy.exporter import NearmapAIExporter


if __name__ == "__main__":
    # Example 1: Extract AI features only (Feature API)
    print("=" * 60)
    print("Example 1: Extracting AI Features Only (Feature API)")
    print("=" * 60)

    feature_only_exporter = NearmapAIExporter(
        aoi_file='tests/data/test_parcels_2.csv',  # New Jersey parcels (US)
        output_dir='data/outputs/features_only',
        country='us',
        packs=['building', 'vegetation'],  # Options: building, vegetation, surfaces, pools, damage, etc.
        processes=2,
        save_features=True,
        include_parcel_geometry=True,
        roof_age=False,  # Feature API only
    )

    print("\nExtracting Nearmap AI features (Feature API only)...")
    feature_only_exporter.run()
    print("Feature extraction complete! Check data/outputs/features_only/\n")

    # Example 2: Extract both AI features and roof age (unified)
    print("=" * 60)
    print("Example 2: Extracting Features + Roof Age (Unified)")
    print("=" * 60)

    unified_exporter = NearmapAIExporter(
        aoi_file='tests/data/test_parcels_2.csv',  # Same New Jersey parcels
        output_dir='data/outputs/unified',
        country='us',
        packs=['building'],  # Default pack
        processes=2,
        save_features=True,
        include_parcel_geometry=True,
        roof_age=True,  # Also query Roof Age API
    )

    print("\nExtracting unified features + roof age...")
    unified_exporter.run()
    print("Unified extraction complete! Check data/outputs/unified/\n")

    print("=" * 60)
    print("All extractions complete!")
    print("=" * 60)
    print("\nOutput locations:")
    print("  - Features Only: data/outputs/features_only/")
    print("  - Unified:       data/outputs/unified/")
    print("\nOutput files include:")
    print("  - {stem}_aoi_rollup.csv - One row per AOI with rollup statistics")
    print("  - {stem}_{class}.csv - One row per feature (e.g., roof.csv, roof_instance.csv)")
    print("  - {stem}_{class}_features.parquet - GeoParquet version with geometry")
    print("  - {stem}_feature_api_errors.csv - Feature API failures (if any)")
    print("  - {stem}_roof_age_errors.csv - Roof Age API failures (if any)")
    print("\nNote: Unified output includes:")
    print("  - feature_api_success column (Y/N per AOI)")
    print("  - roof_age_api_success column (Y/N per AOI)")
