#!/usr/bin/env python
"""Test that roofConditionConfidenceStats is correctly handled in flatten_roof_attributes."""

import ast
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


@pytest.fixture
def roof_condition_confidence_stats_payload():
    """Load the raw roofConditionConfidenceStats API payload."""
    raw_payload_file = data_directory / "test_roof_condition_confidence_stats_raw_payload.json"
    if not raw_payload_file.exists():
        pytest.skip(f"Raw payload file {raw_payload_file} does not exist. Run test_gen_roof_condition_confidence_stats_data first.")

    with open(raw_payload_file, 'r') as f:
        return json.load(f)


@pytest.fixture
def roof_with_rccs(roof_condition_confidence_stats_payload):
    """
    Get a roof feature with roofConditionConfidenceStats from the API payload.

    Note: confidenceStats are nested in attributes.components, not at root level.
    """
    for feature in roof_condition_confidence_stats_payload['features']:
        if feature.get('classId') == ROOF_ID:
            # Check if this roof has confidenceStats in its attributes
            for attr in feature.get('attributes', []):
                if 'components' in attr:
                    for component in attr['components']:
                        if 'confidenceStats' in component:
                            return feature
    pytest.skip("No roof with confidenceStats in components found in payload")


@pytest.mark.skip("Comment out this line if you wish to generate roofConditionConfidenceStats test data")
def test_gen_roof_condition_confidence_stats_data(cache_directory: Path):
    """
    Generate test data with roofConditionConfidenceStats from the API.
    This should be run once to create the test data file.
    """
    # US location that has RSI data (Phoenix, AZ)
    test_polygon = Polygon([
        [-111.9260, 33.4152],
        [-111.9250, 33.4152],
        [-111.9250, 33.4142],
        [-111.9260, 33.4142],
        [-111.9260, 33.4152]
    ])

    test_gdf = gpd.GeoDataFrame(
        [{"aoi_id": "rccs_test_area", "geometry": test_polygon}],
        crs="EPSG:4326"
    ).set_index(AOI_ID_COLUMN_NAME)

    api_key = os.getenv("API_KEY")
    if not api_key:
        pytest.skip("API_KEY not found in environment")

    # Fetch features with roofConditionConfidenceStats included
    # The API already uses gen6 by default
    api = FeatureApi(api_key=api_key, cache_dir=cache_directory)

    # Get the raw payload to save
    print("Fetching raw payload...")
    raw_payload = api.get_features(
        geometry=test_polygon,
        region="us",
        packs=["roof_cond"],  # Use roof_cond pack for roof condition stats
        include=["roofConditionConfidenceStats"]
    )

    # Save raw payload
    raw_payload_file = data_directory / "test_roof_condition_confidence_stats_raw_payload.json"
    with open(raw_payload_file, 'w') as f:
        json.dump(raw_payload, f, indent=2)
    print(f"Saved raw payload to {raw_payload_file}")

    # Now get the processed GeoDataFrame
    features_gdf, metadata_df, errors_df = api.get_features_gdf_bulk(
        test_gdf,
        region="us",
        packs=["roof_cond"],  # Use roof_cond pack for roof condition stats
        include=["roofConditionConfidenceStats"]
    )

    if len(errors_df) > 0:
        print(f"Errors fetching data: {errors_df}")

    # Save the features
    outfile = data_directory / "test_features_roof_condition_confidence_stats.csv"
    features_gdf.to_csv(outfile)
    print(f"Saved {len(features_gdf)} features to {outfile}")

    # Check if we got roofConditionConfidenceStats data
    roof_features = features_gdf[features_gdf['class_id'] == ROOF_ID]
    rccs_count = 0

    # Check raw payload for roofConditionConfidenceStats
    print("\nChecking raw payload for roofConditionConfidenceStats...")
    for feature in raw_payload.get('features', []):
        if feature.get('classId') == ROOF_ID:
            if 'roofConditionConfidenceStats' in feature:
                rccs_count += 1
                print(f"Found roofConditionConfidenceStats: {feature['roofConditionConfidenceStats']}")
                # Save a sample with RCCS
                rccs_sample_file = data_directory / "test_roof_condition_confidence_stats_sample.json"
                with open(rccs_sample_file, 'w') as f:
                    json.dump({
                        'feature_id': feature.get('id'),
                        'roofConditionConfidenceStats': feature['roofConditionConfidenceStats']
                    }, f, indent=2)
                print(f"Saved RCCS sample to {rccs_sample_file}")
                break

    print(f"Found {rccs_count} roofs with roofConditionConfidenceStats out of {len(roof_features)} total roofs")
    if rccs_count == 0:
        print("WARNING: No roofConditionConfidenceStats data found in API response.")
        print("This might be because:")
        print("  1. The API doesn't have RCCS data for this location")
        print("  2. The include parameter isn't working as expected")
        print("  3. This feature isn't available yet")
        print("\nThe test data has been saved but without roofConditionConfidenceStats.")


def test_roof_condition_confidence_stats_with_real_data(roof_with_rccs):
    """
    Test that roofConditionConfidenceStats is correctly extracted using real API data.

    Uses real API data from the payload fixture. The confidenceStats are nested within
    the Roof Condition attribute's components, not at the root level.
    """
    # Use real API data from fixture - confidenceStats are in attributes.components
    roof = {
        'feature_id': roof_with_rccs['id'],
        'class_id': roof_with_rccs['classId'],
        'attributes': roof_with_rccs.get('attributes', [])
    }

    # Test the flatten_roof_attributes function with the processed data
    result = flatten_roof_attributes([roof], country="us")

    # Verify confidenceStats histogram bins were extracted correctly
    # Each roof condition component should have histogram bins (default: 18 bins, extreme: 3 bins)
    # Look for histogram bin columns
    default_bins = [key for key in result.keys() if "confidence_stats_default_bin_" in key]
    extreme_bins = [key for key in result.keys() if "confidence_stats_extreme_bin_" in key]

    assert len(default_bins) > 0, "Should have default histogram bins"
    assert len(extreme_bins) > 0, "Should have extreme histogram bins"

    # Check that we have the expected number of bins for at least one component
    # Default histogram has 18 bins (0-17), extreme has 3 bins (0-2)
    component_default_bins = [key for key in default_bins if key.startswith("structural_damage_confidence_stats_default_bin_")]
    if len(component_default_bins) > 0:
        assert len(component_default_bins) == 18, f"Expected 18 default bins, got {len(component_default_bins)}"

    component_extreme_bins = [key for key in extreme_bins if key.startswith("structural_damage_confidence_stats_extreme_bin_")]
    if len(component_extreme_bins) > 0:
        assert len(component_extreme_bins) == 3, f"Expected 3 extreme bins, got {len(component_extreme_bins)}"

    # Verify the bin values are ratios between 0-1
    for key in default_bins[:5] + extreme_bins[:5]:  # Check first few bins
        assert 0 <= result[key] <= 1, f"Bin ratio {key} should be between 0 and 1, got {result[key]}"


def test_rollup_with_roof_condition_confidence_stats(roof_condition_confidence_stats_payload):
    """Test parcel rollup with roofConditionConfidenceStats to generate CSV output."""
    from nmaipy.parcels import parcel_rollup
    from shapely.geometry import shape, box
    import pandas as pd

    # Use payload from fixture
    payload = roof_condition_confidence_stats_payload

    # Convert features to GeoDataFrame
    features_list = []
    for feature in payload['features']:
        feature_dict = {
            'feature_id': feature['id'],
            'class_id': feature['classId'],
            'description': feature.get('description'),
            'confidence': feature.get('confidence'),
            'area_sqm': feature.get('areaSqm', 0),
            'area_sqft': feature.get('areaSqft', 0),
            'clipped_area_sqm': feature.get('clippedAreaSqm', 0),
            'clipped_area_sqft': feature.get('clippedAreaSqft', 0),
            'unclipped_area_sqm': feature.get('unclippedAreaSqm', 0),
            'unclipped_area_sqft': feature.get('unclippedAreaSqft', 0),
            'fidelity': feature.get('fidelity', None),
            'attributes': feature.get('attributes', []),
            'survey_date': payload.get('surveyDate'),
            'mesh_date': payload.get('3dDate'),
            'geometry': shape(feature['geometry'])
        }
        features_list.append(feature_dict)

    features_gdf = gpd.GeoDataFrame(features_list, crs="EPSG:4326")
    features_gdf['aoi_id'] = 'test_parcel_rccs'
    features_gdf = features_gdf.set_index(AOI_ID_COLUMN_NAME)

    # Create parcels GeoDataFrame using extent from payload
    parcel_geom = box(payload['extentMinX'], payload['extentMinY'],
                       payload['extentMaxX'], payload['extentMaxY'])
    parcels_gdf = gpd.GeoDataFrame(
        {'aoi_id': ['test_parcel_rccs'], 'geometry': [parcel_geom]},
        crs='EPSG:4326'
    ).set_index(AOI_ID_COLUMN_NAME)

    # Create classes dataframe
    classes = {}
    for feature in payload['features']:
        classes[feature['classId']] = feature['description']
    classes_df = pd.DataFrame({'description': classes}).rename_axis('class_id')

    # Calculate rollup
    rollup_df = parcel_rollup(
        parcels_gdf=parcels_gdf,
        features_gdf=features_gdf,
        classes_df=classes_df,
        country='us',
        primary_decision='largest_intersection'
    )

    # Save to CSV
    outfile = data_directory / "test_parcels_rollup_roof_condition_confidence_stats.csv"
    rollup_df.to_csv(outfile)
    print(f"\nSaved rollup with {len(rollup_df.columns)} columns to {outfile}")

    # Verify histogram bin columns are present
    columns = list(rollup_df.columns)
    default_bin_cols = [col for col in columns if "confidence_stats_default_bin_" in col]
    extreme_bin_cols = [col for col in columns if "confidence_stats_extreme_bin_" in col]

    print(f"Found {len(default_bin_cols)} default histogram bin columns")
    print(f"Found {len(extreme_bin_cols)} extreme histogram bin columns")
    print(f"Total confidence stats columns: {len(default_bin_cols) + len(extreme_bin_cols)}")

    print(f"\nSample default bins:")
    for col in default_bin_cols[:5]:  # Show first 5
        print(f"  - {col}")
    if len(default_bin_cols) > 5:
        print(f"  ... and {len(default_bin_cols) - 5} more")

    print(f"\nSample extreme bins:")
    for col in extreme_bin_cols[:5]:  # Show first 5
        print(f"  - {col}")
    if len(extreme_bin_cols) > 5:
        print(f"  ... and {len(extreme_bin_cols) - 5} more")

    assert len(default_bin_cols) > 0, "Should have default histogram bin columns in rollup"
    assert len(extreme_bin_cols) > 0, "Should have extreme histogram bin columns in rollup"


def test_handles_missing_rccs_gracefully():
    """Test that the function handles missing roofConditionConfidenceStats gracefully."""

    test_data_file = data_directory / "test_features.csv"  # Original test data without RCCS

    if not test_data_file.exists():
        pytest.skip(f"Test data file {test_data_file} does not exist.")

    # Load the test data
    df = pd.read_csv(test_data_file)

    # Get a roof feature (which won't have RCCS)
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
                    attrs = ast.literal_eval(attrs)
                except:
                    attrs = []
        roof['attributes'] = attrs if isinstance(attrs, list) else []
    else:
        roof['attributes'] = []

    # Run flatten_roof_attributes - should not crash even without RCCS
    result = flatten_roof_attributes([roof], country="au")

    # Verify no RCCS fields in result
    assert "roof_condition_confidence_stats" not in result, \
        "roof_condition_confidence_stats should not be present when not in source data"

    print("Successfully handled missing roofConditionConfidenceStats")
