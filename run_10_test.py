#!/usr/bin/env python
"""
Quick test script for property-based exports.

This is a minimal working example that extracts:
- Building features from the Feature API
- Roof age predictions from the Roof Age API (US only)

Usage:
    export API_KEY=your_api_key_here
    python run_10_test.py

Output will be saved to data/outputs/quick_test/
"""

from nmaipy.exporter import NearmapAIExporter

if __name__ == "__main__":
    exporter = NearmapAIExporter(
        # Input: CSV with aoi_id and geometry columns (WKT polygons)
        aoi_file="tests/data/test_parcels_2.csv",
        # Output directory (will be created if it doesn't exist)
        output_dir="data/outputs/quick_test",
        # Country code (us, au, nz, ca) - affects area units and API endpoints
        country="us",
        # AI packs to extract (building, vegetation, surfaces, pools, solar, damage, etc.)
        packs=["building"],
        # Parallel processing settings
        processes=4,
        threads=4,
        # Save individual feature geometries (GeoParquet files)
        save_features=True,
        # Include input parcel boundaries in output
        include_parcel_geometry=True,
        # Include Roof Age API data (US only) - adds roof installation date predictions
        roof_age=True,
        # Primary feature selection: 'largest_intersection', 'nearest', or 'optimal'
        primary_decision="optimal",
    )

    exporter.run()
