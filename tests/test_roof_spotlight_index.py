#!/usr/bin/env python
"""Test that roofSpotlightIndex is correctly handled in flatten_roof_attributes."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import Polygon

from nmaipy.constants import ROOF_ID, AOI_ID_COLUMN_NAME
from nmaipy.feature_api import FeatureApi
from nmaipy.parcels import flatten_roof_attributes

data_directory = Path(__file__).parent / "data"


@pytest.mark.skip("Comment out this line if you wish to generate roofSpotlightIndex test data")
def test_gen_rsi_data(cache_directory: Path):
    """
    Generate test data with roofSpotlightIndex from the API.
    This should be run once to create the test data file.
    """
    # US location likely to have roofSpotlightIndex data
    test_polygon = Polygon([
        [-111.9260, 33.4152],  # Phoenix, AZ area
        [-111.9250, 33.4152],
        [-111.9250, 33.4142],
        [-111.9260, 33.4142],
        [-111.9260, 33.4152]
    ])
    
    test_gdf = gpd.GeoDataFrame(
        [{"aoi_id": "rsi_test_area", "geometry": test_polygon}],
        crs="EPSG:4326"
    ).set_index(AOI_ID_COLUMN_NAME)
    
    api_key = os.getenv("API_KEY")
    if not api_key:
        pytest.skip("API_KEY not found in environment")
    
    # Fetch features with roofSpotlightIndex included
    api = FeatureApi(api_key=api_key, cache_dir=cache_directory)
    
    # Also get the raw payload to save
    print("Fetching raw payload...")
    raw_payload = api.get_features(
        geometry=test_polygon,
        region="us", 
        packs=["building"],
        include=["roofSpotlightIndex"]
        # No date constraints - get the latest available
    )
    
    # Save raw payload
    raw_payload_file = data_directory / "test_rsi_raw_payload.json"
    with open(raw_payload_file, 'w') as f:
        json.dump(raw_payload, f, indent=2)
    print(f"Saved raw payload to {raw_payload_file}")
    
    # Now get the processed GeoDataFrame (without date constraints)
    features_gdf, metadata_df, errors_df = api.get_features_gdf_bulk(
        test_gdf,
        region="us",
        packs=["building"],
        include=["roofSpotlightIndex"]
        # No date constraints - get the latest available
    )
    
    if len(errors_df) > 0:
        print(f"Errors fetching data: {errors_df}")
    
    # Save the features
    outfile = data_directory / "test_features_rsi.csv"
    features_gdf.to_csv(outfile)
    print(f"Saved {len(features_gdf)} features to {outfile}")
    
    # Check if we got roofSpotlightIndex data
    roof_features = features_gdf[features_gdf['class_id'] == ROOF_ID]
    rsi_count = 0
    roof_samples = []
    
    for idx, row in roof_features.iterrows():
        attrs = row.get('attributes')
        if attrs is not None and not (isinstance(attrs, float) and pd.isna(attrs)):
            if isinstance(attrs, str):
                try:
                    attrs = json.loads(attrs.replace("'", '"'))
                except:
                    attrs = eval(attrs) if attrs else []
            if isinstance(attrs, list) and len(attrs) > 0:
                # Save sample of roof with attributes
                roof_sample = {
                    'feature_id': row.get('feature_id'),
                    'attributes': attrs
                }
                roof_samples.append(roof_sample)
                
                for attr in attrs:
                    if isinstance(attr, dict) and attr.get('description') == 'roofSpotlightIndex':
                        rsi_count += 1
                        print(f"Found roofSpotlightIndex: {attr}")
                        # Save a sample with RSI
                        rsi_sample_file = data_directory / "test_rsi_sample.json"
                        with open(rsi_sample_file, 'w') as f:
                            json.dump({
                                'feature_id': row.get('feature_id'),
                                'roofSpotlightIndex_attribute': attr,
                                'all_attributes': attrs
                            }, f, indent=2)
                        print(f"Saved RSI sample to {rsi_sample_file}")
                        break
    
    # Save samples of roofs with attributes
    if roof_samples:
        samples_file = data_directory / "test_roof_attributes_samples.json"
        with open(samples_file, 'w') as f:
            json.dump(roof_samples, f, indent=2)
        print(f"Saved {len(roof_samples)} roof attribute samples to {samples_file}")
    
    print(f"Found {rsi_count} roofs with roofSpotlightIndex out of {len(roof_features)} total roofs")
    if rsi_count == 0:
        print("WARNING: No roofSpotlightIndex data found in API response.")
        print("This might be because:")
        print("  1. The API doesn't have RSI data for this location")
        print("  2. The date doesn't have RSI data available")
        print("  3. The include parameter isn't working as expected")
        print("\nThe test data has been saved but without roofSpotlightIndex.")
    # Don't fail - we can still test that the code handles missing RSI gracefully


def test_roof_spotlight_index_with_real_data():
    """
    Test that roofSpotlightIndex is correctly extracted using real API data.
    
    roofSpotlightIndex is available with gen6 on "Roof" features as a root-level property.
    """
    # We know from the raw payload that we have RSI data
    raw_payload_file = data_directory / "test_rsi_raw_payload.json"
    
    if not raw_payload_file.exists():
        pytest.skip(f"Raw payload file {raw_payload_file} does not exist. Run test_gen_rsi_data first.")
    
    # Load the raw payload to get the actual structure
    with open(raw_payload_file, 'r') as f:
        payload = json.load(f)
    
    # Find a roof feature with roofSpotlightIndex
    roof_with_rsi = None
    for feature in payload['features']:
        if feature.get('classId') == ROOF_ID and 'roofSpotlightIndex' in feature:
            # Found one! Simulate what happens after processing through feature_api
            # The columns get converted to snake_case
            roof_with_rsi = {
                'feature_id': feature['id'],
                'class_id': feature['classId'],
                'roof_spotlight_index': feature['roofSpotlightIndex'],  # snake_case conversion
                'attributes': feature.get('attributes', [])
            }
            # Store the original RSI data for verification
            original_rsi = feature['roofSpotlightIndex']
            break
    
    if not roof_with_rsi:
        pytest.skip("No roofSpotlightIndex found in raw payload data.")
    
    # Test the flatten_roof_attributes function with the processed data
    result = flatten_roof_attributes([roof_with_rsi], country="us")
    
    # Verify roofSpotlightIndex was extracted correctly
    assert "roof_spotlight_index" in result, "roof_spotlight_index should be present in flattened result"
    assert result["roof_spotlight_index"] == original_rsi['value'], \
        f"Expected value {original_rsi['value']}, got {result['roof_spotlight_index']}"
    
    if 'confidence' in original_rsi:
        assert "roof_spotlight_index_confidence" in result, \
            "confidence should be in flattened result"
        assert result["roof_spotlight_index_confidence"] == original_rsi['confidence'], \
            f"Expected confidence {original_rsi['confidence']}, got {result['roof_spotlight_index_confidence']}"
    
    if 'modelVersion' in original_rsi:
        assert "roof_spotlight_index_model_version" in result, \
            "modelVersion should be in flattened result"
        assert result["roof_spotlight_index_model_version"] == original_rsi['modelVersion'], \
            f"Expected modelVersion {original_rsi['modelVersion']}, got {result['roof_spotlight_index_model_version']}"
    
    print(f"✓ Successfully extracted roofSpotlightIndex from real gen6 API data")
    print(f"  Value: {result.get('roof_spotlight_index')}")
    print(f"  Confidence: {result.get('roof_spotlight_index_confidence')}")


def test_handles_missing_rsi_gracefully():
    """Test that the function handles missing roofSpotlightIndex gracefully."""
    
    test_data_file = data_directory / "test_features.csv"  # Original test data without RSI
    
    if not test_data_file.exists():
        pytest.skip(f"Test data file {test_data_file} does not exist.")
    
    # Load the test data
    df = pd.read_csv(test_data_file)
    
    # Get a roof feature (which won't have RSI)
    roof_features = df[df['class_id'] == ROOF_ID]
    
    if len(roof_features) == 0:
        pytest.skip("No roof features in test data")
    
    # Take first roof
    roof = roof_features.iloc[0].to_dict()
    
    # Parse attributes if it's a string
    if pd.notna(roof.get('attributes')):
        attrs = roof['attributes']
        if isinstance(attrs, str):
            try:
                attrs = json.loads(attrs.replace("'", '"'))
            except:
                try:
                    attrs = eval(attrs)
                except:
                    attrs = []
        roof['attributes'] = attrs if isinstance(attrs, list) else []
    else:
        roof['attributes'] = []
    
    # Run flatten_roof_attributes - should not crash even without RSI
    result = flatten_roof_attributes([roof], country="au")
    
    # Verify no RSI fields in result
    assert "roof_spotlight_index" not in result, \
        "roof_spotlight_index should not be present when not in source data"
    assert "roof_spotlight_index_confidence" not in result
    assert "roof_spotlight_index_model_version" not in result
    
    print("Successfully handled missing roofSpotlightIndex")