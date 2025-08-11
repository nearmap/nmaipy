#!/usr/bin/env python
"""
Examples of common nmaipy use cases for data scientists.

These examples show how to extract different types of Nearmap AI features
for various analysis scenarios.
"""

import os
from nmaipy.exporter import AOIExporter

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
    exporter = AOIExporter(
        aoi_file='data/examples/sydney_parcels.geojson',
        output_dir='data/outputs/basic',
        country='au',
        packs=['building', 'vegetation'],
        processes=4
    )
    exporter.run()


def example_damage_assessment():
    """
    Extract damage classification data after a natural disaster.
    Perfect for insurance and emergency response analysis.
    """
    exporter = AOIExporter(
        aoi_file='data/examples/us_parcels.geojson',
        output_dir='data/outputs/damage',
        country='us',
        packs=['damage'],
        
        # Date range for the disaster
        since='2024-07-08',  # Hurricane Beryl
        until='2024-07-11',
        
        # Use rapid assessment mode for post-catastrophe
        rapid=True,
        
        # Get the latest imagery
        order='latest',
        
        # Save all features for detailed analysis
        save_features=True,
        processes=8
    )
    exporter.run()


def example_urban_planning():
    """
    Extract comprehensive data for urban planning analysis.
    Includes buildings, vegetation, surfaces, and solar panels.
    """
    exporter = AOIExporter(
        aoi_file='data/examples/sydney_parcels.geojson',
        output_dir='data/outputs/urban',
        country='au',
        
        # Multiple AI packs for comprehensive analysis
        packs=['building', 'vegetation', 'surfaces', 'solar'],
        
        # Include building characteristics
        include=['building_characteristics'],
        
        # Save individual features for detailed GIS analysis
        save_features=True,
        include_parcel_geometry=True,
        
        # Use latest generation AI
        system_version_prefix='gen6-',
        
        processes=8
    )
    exporter.run()


def example_vegetation_analysis():
    """
    Focused vegetation analysis for environmental studies.
    """
    exporter = AOIExporter(
        aoi_file='data/examples/sydney_parcels.geojson',
        output_dir='data/outputs/vegetation',
        country='au',
        
        # Just vegetation data
        packs=['vegetation'],
        
        # Get individual tree features
        save_features=True,
        
        # Process in smaller chunks for large areas
        chunk_size=50,
        processes=4
    )
    exporter.run()


def example_pool_detection():
    """
    Detect swimming pools for compliance or market analysis.
    """
    exporter = AOIExporter(
        aoi_file='data/examples/sydney_parcels.geojson',
        output_dir='data/outputs/pools',
        country='au',
        
        # Pool detection
        packs=['pools'],
        
        # Include parcel boundaries for mapping
        include_parcel_geometry=True,
        
        processes=4
    )
    exporter.run()


def example_large_area_extraction():
    """
    Handle large areas efficiently with gridding.
    """
    exporter = AOIExporter(
        aoi_file='data/examples/large_area.geojson',
        output_dir='data/outputs/large',
        country='au',
        
        packs=['building', 'vegetation'],
        
        # Allow combining data from different survey dates
        aoi_grid_inexact=True,
        
        # Accept partial results (0 = accept any percentage)
        aoi_grid_min_pct=0,
        
        # Process in chunks
        chunk_size=100,
        
        # Use more processes for speed
        processes=16
    )
    exporter.run()


def example_time_series():
    """
    Extract data for a specific time period for change detection.
    """
    exporter = AOIExporter(
        aoi_file='data/examples/sydney_parcels.geojson',
        output_dir='data/outputs/timeseries',
        country='au',
        
        packs=['building', 'vegetation'],
        
        # Specify date range
        since='2024-01-01',
        until='2024-06-30',
        
        # Get earliest imagery in the range
        order='earliest',
        
        save_features=True,
        processes=4
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
    
    print("\nEdit this file and uncomment an example to run it.")