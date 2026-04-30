#!/usr/bin/env python
"""
Examples of common nmaipy use cases for data scientists.

These examples show how to extract different types of Nearmap AI features
for various analysis scenarios.
"""

import os

from nmaipy.exporter import NearmapAIExporter

# Make sure you have your API key set
# Option 1: Set in environment before running
# export API_KEY=your_api_key_here
#
# Option 2: Set in script (less secure)
# os.environ['API_KEY'] = 'your_api_key_here'


def example_basic_extraction():
    """
    Basic example: Extract building and vegetation data for parcels.
    """
    exporter = NearmapAIExporter(
        aoi_file="data/examples/sydney_parcels.geojson",
        output_dir="data/outputs/basic",
        country="au",
        packs=["building", "vegetation"],
        processes=4,
    )
    exporter.run()


def example_damage_assessment():
    """
    Extract damage classification data after a natural disaster.
    Perfect for insurance and emergency response analysis.
    """
    exporter = NearmapAIExporter(
        aoi_file="data/examples/us_parcels.geojson",
        output_dir="data/outputs/damage",
        country="us",
        packs=["damage"],
        # Date range for the disaster
        since="2024-07-08",  # Hurricane Beryl
        until="2024-07-11",
        # Use rapid assessment mode for post-catastrophe
        rapid=True,
        # Get the latest imagery
        order="latest",
        # Save all features for detailed analysis
        save_features=True,
        processes=8,
    )
    exporter.run()


def example_urban_planning():
    """
    Extract comprehensive data for urban planning analysis.
    Includes buildings, vegetation, surfaces, and solar panels.
    """
    exporter = NearmapAIExporter(
        aoi_file="data/examples/sydney_parcels.geojson",
        output_dir="data/outputs/urban",
        country="au",
        # Multiple AI packs for comprehensive analysis
        packs=["building", "vegetation", "surfaces", "solar"],
        # Include building characteristics
        include=["building_characteristics"],
        # Save individual features for detailed GIS analysis
        save_features=True,
        include_parcel_geometry=True,
        # Use latest generation AI
        system_version_prefix="gen6-",
        processes=8,
    )
    exporter.run()


def example_vegetation_analysis():
    """
    Focused vegetation analysis for environmental studies.
    """
    exporter = NearmapAIExporter(
        aoi_file="data/examples/sydney_parcels.geojson",
        output_dir="data/outputs/vegetation",
        country="au",
        # Just vegetation data
        packs=["vegetation"],
        # Get individual tree features
        save_features=True,
        # Process in smaller chunks for large areas
        chunk_size=50,
        processes=4,
    )
    exporter.run()


def example_pool_detection():
    """
    Detect swimming pools for compliance or market analysis.
    """
    exporter = NearmapAIExporter(
        aoi_file="data/examples/sydney_parcels.geojson",
        output_dir="data/outputs/pools",
        country="au",
        # Pool detection
        packs=["pools"],
        # Include parcel boundaries for mapping
        include_parcel_geometry=True,
        processes=4,
    )
    exporter.run()


def example_large_area_extraction():
    """
    Handle large areas efficiently with gridding.
    """
    exporter = NearmapAIExporter(
        aoi_file="data/examples/large_area.geojson",
        output_dir="data/outputs/large",
        country="au",
        packs=["building", "vegetation"],
        # Allow combining data from different survey dates
        aoi_grid_inexact=True,
        # Accept partial results (0 = accept any percentage)
        aoi_grid_min_pct=0,
        # Process in chunks
        chunk_size=100,
        # Use more processes for speed
        processes=16,
    )
    exporter.run()


def example_time_series():
    """
    Extract data for a specific time period for change detection.
    """
    exporter = NearmapAIExporter(
        aoi_file="data/examples/sydney_parcels.geojson",
        output_dir="data/outputs/timeseries",
        country="au",
        packs=["building", "vegetation"],
        # Specify date range
        since="2024-01-01",
        until="2024-06-30",
        # Get earliest imagery in the range
        order="earliest",
        save_features=True,
        processes=4,
    )
    exporter.run()


def example_roof_age_unified():
    """
    Extract building features AND roof age predictions in one export (US only).
    This is the recommended approach for combining Feature API and Roof Age API data.
    """
    exporter = NearmapAIExporter(
        aoi_file="data/examples/us_parcels.geojson",
        output_dir="data/outputs/roof_age",
        country="us",  # Roof Age API is US only
        packs=["building"],
        # Include Roof Age API data
        roof_age=True,
        # Save individual features as GeoParquet
        save_features=True,
        processes=4,
    )
    exporter.run()


def example_roof_age_a1():
    """
    Use the A.1 Roof Age dataset instead of the default A.0.

    A.1 uses a refreshed model (different installation dates and trust scores
    vs A.0 for the same parcels) and supports the historical 'untilAsOfDate'
    and 'sinceAsOfDate' parameters, which A.0 does not. Pass either the
    friendly alias 'A.1' or a raw resource UUID (the alias just maps to the
    UUID under the hood).
    """
    exporter = NearmapAIExporter(
        aoi_file="data/examples/us_parcels.geojson",
        output_dir="data/outputs/roof_age_a1",
        country="us",
        packs=["building"],
        roof_age=True,
        # Select the A.1 dataset. Use 'A.0' / 'latest' (default) for the original
        # model, or pass a raw resource UUID to target a newly published dataset
        # without a code change.
        roof_age_dataset="A.1",
        save_features=True,
        processes=4,
    )
    exporter.run()


def example_roof_age_historical_bulk():
    """
    Roof age 'as of' a historical date — same cutoff applied to every AOI.

    Useful for reconstructing roof state at a fixed point in the past — for
    example, what every roof in a portfolio looked like immediately before a
    storm event. Requires A.1+ (A.0 does not support untilAsOfDate).
    """
    exporter = NearmapAIExporter(
        aoi_file="data/examples/us_parcels.geojson",
        output_dir="data/outputs/roof_age_until_2020",
        country="us",
        packs=["building"],
        roof_age=True,
        roof_age_dataset="A.1",
        # Roof state as of 2020-01-01. The Roof Age API receives this as the
        # 'untilAsOfDate' parameter on every per-AOI request.
        until="2020-01-01",
        save_features=True,
        processes=4,
    )
    exporter.run()


def example_roof_age_historical_per_aoi():
    """
    Per-AOI 'untilAsOfDate' driven by a column in the input file.

    Add a string 'until' column to your AOI input (CSV, GeoJSON, etc.) with a
    YYYY-MM-DD value per row. The exporter sends each AOI's value as that
    AOI's untilAsOfDate, mirroring Feature API per-AOI override semantics.
    Use this when each property in your portfolio has its own date of interest
    (e.g. policy inception date, claim date, transaction date). Requires a
    non-A.0 dataset.

    The 'until' column must be string-typed YYYY-MM-DD. Pandas will sometimes
    auto-parse ISO date columns as datetime64; if that happens for your input,
    cast back to string with df['until'] = df['until'].dt.strftime('%Y-%m-%d')
    before exporting (the exporter raises a clear error if it sees a non-string
    dtype).

    Example input file (CSV):
        aoi_id,geometry,until
        parcel_a,POLYGON((...)),2018-06-15
        parcel_b,POLYGON((...)),2022-11-30
        parcel_c,POLYGON((...)),       # blank → falls back to bulk default (no cutoff)
    """
    exporter = NearmapAIExporter(
        aoi_file="data/examples/us_parcels_with_until.csv",
        output_dir="data/outputs/roof_age_per_aoi_until",
        country="us",
        packs=["building"],
        roof_age=True,
        roof_age_dataset="A.1",
        save_features=True,
        processes=4,
    )
    exporter.run()


def example_roof_age_date_range():
    """
    Roof age data restricted to a date range — both 'sinceAsOfDate' and 'untilAsOfDate'.

    Useful for reconstructing the slice of a roof's history that overlaps a window of
    interest — for example, the portfolio's roof state during a policy term. Requires
    a non-A.0 dataset.
    """
    exporter = NearmapAIExporter(
        aoi_file="data/examples/us_parcels.geojson",
        output_dir="data/outputs/roof_age_2018_2020",
        country="us",
        packs=["building"],
        roof_age=True,
        roof_age_dataset="A.1",
        # Roof state restricted to the window [2018-01-01, 2020-12-31]. The
        # Roof Age API receives these as 'sinceAsOfDate' / 'untilAsOfDate' on
        # every per-AOI request.
        since="2018-01-01",
        until="2020-12-31",
        save_features=True,
        processes=4,
    )
    exporter.run()


if __name__ == "__main__":
    print("nmaipy Examples")
    print("-" * 40)
    print("Uncomment the example you want to run:")
    print()

    # Uncomment the example you want to run:

    # example_basic_extraction()
    # example_damage_assessment()
    # example_urban_planning()
    # example_vegetation_analysis()
    # example_pool_detection()
    # example_large_area_extraction()
    # example_time_series()
    # example_roof_age_unified()
    # example_roof_age_a1()
    # example_roof_age_historical_bulk()
    # example_roof_age_historical_per_aoi()
    # example_roof_age_date_range()

    print("\nEdit this file and uncomment an example to run it.")
