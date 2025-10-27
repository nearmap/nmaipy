#!/usr/bin/env python
"""Test that defensibleSpace is correctly handled in flatten_roof_attributes."""

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
def defensible_space_payload():
    """Load the raw defensibleSpace API payload."""
    raw_payload_file = data_directory / "test_defensible_space_raw_payload.json"
    if not raw_payload_file.exists():
        pytest.skip(f"Raw payload file {raw_payload_file} does not exist. Run test_gen_defensible_space_data first.")

    with open(raw_payload_file, 'r') as f:
        return json.load(f)


@pytest.fixture
def roof_with_defensible_space(defensible_space_payload):
    """Get a roof feature with defensibleSpace from the API payload."""
    for feature in defensible_space_payload['features']:
        if feature.get('classId') == ROOF_ID and 'defensibleSpace' in feature:
            return feature
    pytest.skip("No roof with defensibleSpace found in payload")


@pytest.fixture
def all_roofs_with_defensible_space(defensible_space_payload):
    """Get all roof features with defensibleSpace from the API payload."""
    roofs = [f for f in defensible_space_payload['features']
             if f.get('classId') == ROOF_ID and 'defensibleSpace' in f]
    if not roofs:
        pytest.skip("No roofs with defensibleSpace found in payload")
    return roofs


@pytest.mark.skip("Comment out this line if you wish to generate defensibleSpace test data")
def test_gen_defensible_space_data(cache_directory: Path):
    """
    Generate test data with defensibleSpace from the API.
    This should be run once to create the test data file.
    """
    # US location - Phoenix, AZ area
    test_polygon = Polygon([
        [-111.9260, 33.4152],
        [-111.9250, 33.4152],
        [-111.9250, 33.4142],
        [-111.9260, 33.4142],
        [-111.9260, 33.4152]
    ])

    test_gdf = gpd.GeoDataFrame(
        [{"aoi_id": "defensible_space_test_area", "geometry": test_polygon}],
        crs="EPSG:4326"
    ).set_index(AOI_ID_COLUMN_NAME)

    api_key = os.getenv("API_KEY")
    if not api_key:
        pytest.skip("API_KEY not found in environment")

    # Fetch features with defensibleSpace included
    api = FeatureApi(api_key=api_key, cache_dir=cache_directory)

    # Get the raw payload to save
    print("Fetching raw payload...")
    raw_payload = api.get_features(
        geometry=test_polygon,
        region="us",
        packs=["building"],
        include=["defensibleSpace"]
        # No date constraints - get the latest available
    )

    # Save raw payload
    raw_payload_file = data_directory / "test_defensible_space_raw_payload.json"
    with open(raw_payload_file, 'w') as f:
        json.dump(raw_payload, f, indent=2)
    print(f"Saved raw payload to {raw_payload_file}")

    # Now get the processed GeoDataFrame (without date constraints)
    features_gdf, metadata_df, errors_df = api.get_features_gdf_bulk(
        test_gdf,
        region="us",
        packs=["building"],
        include=["defensibleSpace"]
        # No date constraints - get the latest available
    )

    if len(errors_df) > 0:
        print(f"Errors fetching data: {errors_df}")

    # Save the features
    outfile = data_directory / "test_features_defensible_space.csv"
    features_gdf.to_csv(outfile)
    print(f"Saved {len(features_gdf)} features to {outfile}")

    # Check if we got defensibleSpace data
    roof_features = features_gdf[features_gdf['class_id'] == ROOF_ID]
    ds_count = 0

    for idx, row in roof_features.iterrows():
        # Check for defensibleSpace as a root-level property
        has_defensible_space = False

        # Check both possible field names
        if 'defensible_space' in row.index and pd.notna(row.get('defensible_space')):
            has_defensible_space = True
        elif 'defensibleSpace' in row.index and pd.notna(row.get('defensibleSpace')):
            has_defensible_space = True

        if has_defensible_space:
            ds_count += 1

    print(f"Found {ds_count} roofs with defensibleSpace out of {len(roof_features)} total roof features")
    if ds_count == 0:
        print("WARNING: No defensibleSpace data found in API response.")
        print("This might be because:")
        print("  1. The API doesn't have defensibleSpace data for this location")
        print("  2. The include parameter isn't working as expected")
        print("\nThe test data has been saved but without defensibleSpace.")


def test_defensible_space_with_real_data(roof_with_defensible_space):
    """
    Test that defensibleSpace is correctly extracted using real API data.

    defensibleSpace is available with gen6 on "Roof" features as a root-level property.
    Uses real API data from the payload fixture.
    """
    # Use real API data from fixture
    roof = {
        'feature_id': roof_with_defensible_space['id'],
        'class_id': roof_with_defensible_space['classId'],
        'defensible_space': roof_with_defensible_space['defensibleSpace'],
        'attributes': roof_with_defensible_space.get('attributes', [])
    }

    result = flatten_roof_attributes([roof], country="us")

    # Verify defensibleSpace zones were extracted correctly
    zones = roof_with_defensible_space['defensibleSpace']['zones']
    for zone in zones:
        zone_id = zone['zoneId']
        prefix = f"defensible_space_zone_{zone_id}"

        # Check that zone metrics are present (using imperial units for US)
        assert f"{prefix}_zone_area_sqft" in result
        assert f"{prefix}_defensible_space_area_sqft" in result
        assert f"{prefix}_risk_object_area_sqft" in result
        assert f"{prefix}_coverage_ratio" in result

        # Verify the values match
        assert result[f"{prefix}_zone_area_sqft"] == zone['zoneAreaSqft']
        assert result[f"{prefix}_defensible_space_area_sqft"] == zone['defensibleSpaceAreaSqft']
        assert result[f"{prefix}_risk_object_area_sqft"] == zone['totalRiskObjectAreaSqft']
        assert result[f"{prefix}_coverage_ratio"] == zone['defensibleSpaceCoverageRatio']


def test_defensible_space_multiple_zones(roof_with_defensible_space):
    """Test that multiple zones are correctly flattened."""

    roof = {
        'feature_id': roof_with_defensible_space['id'],
        'class_id': roof_with_defensible_space['classId'],
        'defensible_space': roof_with_defensible_space['defensibleSpace'],
        'attributes': []
    }

    result = flatten_roof_attributes([roof], country="us")

    # Verify we have data for all zones
    num_zones = len(roof_with_defensible_space['defensibleSpace']['zones'])
    for zone_id in range(1, num_zones + 1):
        prefix = f"defensible_space_zone_{zone_id}"
        assert f"{prefix}_zone_area_sqft" in result, f"Zone {zone_id} should be present"


def test_handles_missing_defensible_space_gracefully():
    """Test that the function handles missing defensibleSpace gracefully."""

    # Create a roof without defensibleSpace
    roof = {
        'feature_id': 'test_feature_no_ds',
        'class_id': ROOF_ID,
        'attributes': []
    }

    # Run flatten_roof_attributes - should not crash even without defensibleSpace
    result = flatten_roof_attributes([roof], country="us")

    # Verify no defensibleSpace fields in result
    for key in result.keys():
        assert not key.startswith("defensible_space_"), \
            "defensibleSpace fields should not be present when not in source data"


def test_defensible_space_with_camel_case(roof_with_defensible_space):
    """Test that camelCase field names are handled (as returned directly from API)."""

    # Test with camelCase as it appears in raw API response
    roof = {
        'feature_id': roof_with_defensible_space['id'],
        'class_id': roof_with_defensible_space['classId'],
        'defensibleSpace': roof_with_defensible_space['defensibleSpace'],  # camelCase
        'attributes': roof_with_defensible_space.get('attributes', [])
    }

    result = flatten_roof_attributes([roof], country="us")

    # Should still extract correctly from real data
    zones = roof_with_defensible_space['defensibleSpace']['zones']
    first_zone = zones[0]
    zone_id = first_zone['zoneId']
    prefix = f"defensible_space_zone_{zone_id}"

    assert f"{prefix}_zone_area_sqft" in result
    assert result[f"{prefix}_zone_area_sqft"] == first_zone['zoneAreaSqft']


def test_defensible_space_metric_units(roof_with_defensible_space):
    """Test that metric units are used for non-imperial countries."""

    roof = {
        'feature_id': roof_with_defensible_space['id'],
        'class_id': roof_with_defensible_space['classId'],
        'defensible_space': roof_with_defensible_space['defensibleSpace'],
        'attributes': []
    }

    # Test with Australia (metric)
    result = flatten_roof_attributes([roof], country="au")

    # Verify metric units are used
    zones = roof_with_defensible_space['defensibleSpace']['zones']
    first_zone = zones[0]
    zone_id = first_zone['zoneId']
    prefix = f"defensible_space_zone_{zone_id}"

    assert f"{prefix}_zone_area_sqm" in result
    assert f"{prefix}_defensible_space_area_sqm" in result
    assert f"{prefix}_risk_object_area_sqm" in result
    assert result[f"{prefix}_zone_area_sqm"] == first_zone['zoneAreaSqm']


def test_defensible_space_invalid_data():
    """Test that invalid defensibleSpace data is handled gracefully."""

    # Test with scalar defensibleSpace (should be a dict)
    roof_scalar = {
        'feature_id': 'test_feature_1',
        'class_id': ROOF_ID,
        'defensible_space': 123,  # Invalid - should be a dict
        'attributes': []
    }

    result_scalar = flatten_roof_attributes([roof_scalar], country="us")
    for key in result_scalar.keys():
        assert not key.startswith("defensible_space_")

    # Test with None defensibleSpace
    roof_none = {
        'feature_id': 'test_feature_2',
        'class_id': ROOF_ID,
        'defensible_space': None,
        'attributes': []
    }

    result_none = flatten_roof_attributes([roof_none], country="us")
    for key in result_none.keys():
        assert not key.startswith("defensible_space_")


def test_defensible_space_with_other_includes(roof_with_defensible_space):
    """Test that defensibleSpace can coexist with other include parameters."""

    # Add mock RSI and hurricaneScore alongside real defensibleSpace
    roof = {
        'feature_id': roof_with_defensible_space['id'],
        'class_id': roof_with_defensible_space['classId'],
        'defensible_space': roof_with_defensible_space['defensibleSpace'],
        'roof_spotlight_index': {
            'value': 85,
            'confidence': 0.92
        },
        'hurricane_score': {
            'vulnerabilityScore': 5,
            'vulnerabilityProbability': 0.808,
            'vulnerabilityRateFactor': 0.34
        },
        'attributes': []
    }

    result = flatten_roof_attributes([roof], country="us")

    # All three should be present
    assert "defensible_space_zone_1_zone_area_sqft" in result
    assert "roof_spotlight_index" in result
    assert "hurricane_vulnerability_score" in result


def test_rollup_with_defensible_space(defensible_space_payload):
    """
    Test parcel rollup with defensibleSpace to generate CSV output.

    This test creates a full rollup CSV file showing how defensibleSpace
    appears in the flattened output.
    Uses real API data from the payload fixture.
    """
    from nmaipy.parcels import parcel_rollup
    from shapely.geometry import shape, box

    # Use payload from fixture
    payload = defensible_space_payload

    # Convert features to GeoDataFrame
    features = []
    for feature in payload['features']:
        feat_dict = {
            'feature_id': feature['id'],
            'class_id': feature['classId'],
            'description': feature['description'],
            'confidence': feature['confidence'],
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

        # Add defensibleSpace if present (from real API response)
        if 'defensibleSpace' in feature:
            feat_dict['defensible_space'] = feature['defensibleSpace']

        features.append(feat_dict)

    # Create features GeoDataFrame
    features_gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')
    features_gdf['aoi_id'] = 'test_parcel_defensible_space'
    features_gdf = features_gdf.set_index(AOI_ID_COLUMN_NAME)

    # Create parcels GeoDataFrame
    parcel_geom = box(payload['extentMinX'], payload['extentMinY'],
                       payload['extentMaxX'], payload['extentMaxY'])
    parcels_gdf = gpd.GeoDataFrame(
        {'aoi_id': ['test_parcel_defensible_space'], 'geometry': [parcel_geom]},
        crs='EPSG:4326'
    ).set_index(AOI_ID_COLUMN_NAME)

    # Create classes dataframe
    classes = {}
    for feature in payload['features']:
        classes[feature['classId']] = feature['description']
    classes_df = pd.DataFrame({'description': classes}).rename_axis('class_id')

    # Run parcel_rollup
    rollup_df = parcel_rollup(
        parcels_gdf=parcels_gdf,
        features_gdf=features_gdf,
        classes_df=classes_df,
        country='us',
        primary_decision='largest_intersection'
    )

    # Save rollup CSV
    output_file = data_directory / "test_parcels_rollup_defensible_space.csv"
    rollup_df.to_csv(output_file)

    # Verify defensibleSpace columns are present in primary roof
    # Should have data for all 3 zones
    for zone_id in [1, 2, 3]:
        assert f'primary_roof_defensible_space_zone_{zone_id}_zone_area_sqft' in rollup_df.columns
        assert f'primary_roof_defensible_space_zone_{zone_id}_defensible_space_area_sqft' in rollup_df.columns
        assert f'primary_roof_defensible_space_zone_{zone_id}_risk_object_area_sqft' in rollup_df.columns
        assert f'primary_roof_defensible_space_zone_{zone_id}_coverage_ratio' in rollup_df.columns

    # Verify the values are present and reasonable
    zone_1_area = rollup_df['primary_roof_defensible_space_zone_1_zone_area_sqft'].iloc[0]
    assert zone_1_area > 0, f"Expected positive zone area, got {zone_1_area}"

    coverage_ratio = rollup_df['primary_roof_defensible_space_zone_1_coverage_ratio'].iloc[0]
    assert 0 <= coverage_ratio <= 1, f"Expected coverage ratio between 0 and 1, got {coverage_ratio}"
