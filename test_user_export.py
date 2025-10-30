#!/usr/bin/env python
"""
Test to verify damage data is returned from API for the user's AOI
"""
import sys
import os
sys.path.insert(0, os.path.expanduser('~/Development/datascience/nmaipy'))

from nmaipy.feature_api import FeatureApi
import geopandas as gpd

# Load user's AOI
aoi_file = '/Users/michael.bewley/Development/datascience/michael_bewley_hacks/2025-10_suncorp_qld_hail/data/Ipswich_QLD_hail_Oct25_20251027_flown.geojson'
aoi_gdf = gpd.read_file(aoi_file)

print(f"Loaded {len(aoi_gdf)} AOIs")
print(f"AOI bounds: {aoi_gdf.total_bounds}")

# Get first AOI for testing
first_aoi = aoi_gdf.iloc[0].geometry

# Test with damage pack WITHOUT dates
print("\n=== Test 1: WITHOUT since/until dates ===")
feature_api = FeatureApi(
    api_key=os.environ.get('API_KEY'),
    parcel_mode=True
)

features_gdf, metadata, error = feature_api.get_features_gdf(
    first_aoi,
    region='au',
    packs=['damage'],
    include=['damage'],
    aoi_id='test_aoi'
)

if error:
    print(f"Error: {error}")
else:
    print(f"Features: {len(features_gdf)}")
    print(f"Has damage column: {'damage' in features_gdf.columns}")
    if 'damage' in features_gdf.columns:
        non_null = features_gdf['damage'].notna().sum()
        print(f"Features with damage data: {non_null}/{len(features_gdf)}")
    if metadata:
        print(f"Survey date: {metadata.get('date', 'N/A')}")

# Test WITH dates (Oct 27, 2025 - the hail event date)
print("\n=== Test 2: WITH since/until dates (2025-10-27) ===")
features_gdf2, metadata2, error2 = feature_api.get_features_gdf(
    first_aoi,
    region='au',
    packs=['damage'],
    include=['damage'],
    aoi_id='test_aoi',
    since='2025-10-27',
    until='2025-10-27'
)

if error2:
    print(f"Error: {error2}")
else:
    print(f"Features: {len(features_gdf2)}")
    print(f"Has damage column: {'damage' in features_gdf2.columns}")
    if 'damage' in features_gdf2.columns:
        non_null = features_gdf2['damage'].notna().sum()
        print(f"Features with damage data: {non_null}/{len(features_gdf2)}")
        if non_null > 0:
            # Show sample
            sample_damage = features_gdf2[features_gdf2['damage'].notna()]['damage'].iloc[0]
            print(f"Sample damage keys: {list(sample_damage.keys()) if isinstance(sample_damage, dict) else type(sample_damage)}")
    if metadata2:
        print(f"Survey date: {metadata2.get('date', 'N/A')}")
