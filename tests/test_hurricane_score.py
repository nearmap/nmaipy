#!/usr/bin/env python
"""Test that hurricaneScore is correctly handled in flatten_roof_attributes."""

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
def hurricane_score_payload():
    """Load the raw hurricaneScore API payload."""
    raw_payload_file = data_directory / "test_hurricane_score_raw_payload.json"
    if not raw_payload_file.exists():
        pytest.skip(f"Raw payload file {raw_payload_file} does not exist. Run test_gen_hurricane_score_data first.")

    with open(raw_payload_file, 'r') as f:
        return json.load(f)


@pytest.fixture
def roof_with_hurricane_score(hurricane_score_payload):
    """Get a roof feature with hurricaneScore from the API payload."""
    for feature in hurricane_score_payload['features']:
        if feature.get('classId') == ROOF_ID and 'hurricaneScore' in feature:
            return feature
    pytest.skip("No roof with hurricaneScore found in payload")


@pytest.fixture
def roof_with_both_scores(hurricane_score_payload):
    """Get a roof feature with both hurricaneScore and roofSpotlightIndex from the API payload."""
    for feature in hurricane_score_payload['features']:
        if (feature.get('classId') == ROOF_ID and
            'hurricaneScore' in feature and
            'roofSpotlightIndex' in feature):
            return feature
    pytest.skip("No roof with both scores found in payload")


@pytest.fixture
def all_roofs_with_hurricane_score(hurricane_score_payload):
    """Get all roof features with hurricaneScore from the API payload."""
    roofs = [f for f in hurricane_score_payload['features']
             if f.get('classId') == ROOF_ID and 'hurricaneScore' in f]
    if not roofs:
        pytest.skip("No roofs with hurricaneScore found in payload")
    return roofs


@pytest.mark.skip("Comment out this line if you wish to generate hurricaneScore test data")
def test_gen_hurricane_score_data(cache_directory: Path):
    """
    Generate test data with hurricaneScore from the API.
    This should be run once to create the test data file.
    """
    # US location - Phoenix area (same as RSI test for consistency)
    test_polygon = Polygon([
        [-111.9260, 33.4152],  # Phoenix, AZ area
        [-111.9250, 33.4152],
        [-111.9250, 33.4142],
        [-111.9260, 33.4142],
        [-111.9260, 33.4152]
    ])

    test_gdf = gpd.GeoDataFrame(
        [{"aoi_id": "hurricane_score_test_area", "geometry": test_polygon}],
        crs="EPSG:4326"
    ).set_index(AOI_ID_COLUMN_NAME)

    api_key = os.getenv("API_KEY")
    if not api_key:
        pytest.skip("API_KEY not found in environment")

    # Fetch features with hurricaneScore included
    api = FeatureApi(api_key=api_key, cache_dir=cache_directory)

    # Also get the raw payload to save
    print("Fetching raw payload...")
    raw_payload = api.get_features(
        geometry=test_polygon,
        region="us",
        packs=["building"],
        include=["hurricaneScore", "roofSpotlightIndex"]  # Get BOTH includes
        # No date constraints - get the latest available
    )

    # Save raw payload
    raw_payload_file = data_directory / "test_hurricane_score_raw_payload.json"
    with open(raw_payload_file, 'w') as f:
        json.dump(raw_payload, f, indent=2)
    print(f"Saved raw payload to {raw_payload_file}")

    # Now get the processed GeoDataFrame (without date constraints)
    features_gdf, metadata_df, errors_df = api.get_features_gdf_bulk(
        test_gdf,
        region="us",
        packs=["building"],
        include=["hurricaneScore", "roofSpotlightIndex"]  # Get BOTH includes
        # No date constraints - get the latest available
    )

    if len(errors_df) > 0:
        print(f"Errors fetching data: {errors_df}")

    # Save the features
    outfile = data_directory / "test_features_hurricane_score.csv"
    features_gdf.to_csv(outfile)
    print(f"Saved {len(features_gdf)} features to {outfile}")

    # Check if we got hurricaneScore data
    roof_features = features_gdf[features_gdf['class_id'] == ROOF_ID]
    hurricane_score_count = 0
    roof_samples = []

    for idx, row in roof_features.iterrows():
        # Check for hurricaneScore as a root-level property (similar to roofSpotlightIndex)
        # After feature_api processing, camelCase gets converted to snake_case
        has_hurricane_score = False

        # Check both possible field names
        if 'hurricane_score' in row.index and pd.notna(row.get('hurricane_score')):
            has_hurricane_score = True
        elif 'hurricaneScore' in row.index and pd.notna(row.get('hurricaneScore')):
            has_hurricane_score = True

        if has_hurricane_score:
            hurricane_score_count += 1
            # Save sample of roof with hurricaneScore
            roof_sample = {
                'feature_id': row.get('feature_id'),
                'hurricane_score': row.get('hurricane_score') or row.get('hurricaneScore'),
                'attributes': row.get('attributes')
            }
            roof_samples.append(roof_sample)
            print(f"Found hurricaneScore: {roof_sample['hurricane_score']}")

            # Save a sample with hurricane score
            hurricane_score_sample_file = data_directory / "test_hurricane_score_sample.json"
            with open(hurricane_score_sample_file, 'w') as f:
                json.dump(roof_sample, f, indent=2)
            print(f"Saved hurricaneScore sample to {hurricane_score_sample_file}")

    # Save samples of roofs with hurricaneScore
    if roof_samples:
        samples_file = data_directory / "test_roof_hurricane_score_samples.json"
        with open(samples_file, 'w') as f:
            json.dump(roof_samples, f, indent=2)
        print(f"Saved {len(roof_samples)} roof hurricaneScore samples to {samples_file}")

    print(f"Found {hurricane_score_count} roofs with hurricaneScore out of {len(roof_features)} total roof features")
    if hurricane_score_count == 0:
        print("WARNING: No hurricaneScore data found in API response.")
        print("This might be because:")
        print("  1. The API doesn't have hurricaneScore data for this location")
        print("  2. The date doesn't have hurricaneScore data available")
        print("  3. The include parameter isn't working as expected")
        print("\nThe test data has been saved but without hurricaneScore.")


def test_hurricane_score_with_real_data(roof_with_hurricane_score):
    """
    Test that hurricaneScore is correctly extracted using real API data.

    Uses actual roof feature from Phoenix, AZ with gen6 hurricaneScore data.
    """
    # Use real API data from fixture
    roof = {
        'feature_id': roof_with_hurricane_score['id'],
        'class_id': roof_with_hurricane_score['classId'],
        'hurricane_score': roof_with_hurricane_score['hurricaneScore'],
        'attributes': roof_with_hurricane_score.get('attributes', [])
    }

    result = flatten_roof_attributes([roof], country="us")

    # Verify hurricaneScore was extracted correctly using real values from API
    assert "hurricane_vulnerability_score" in result
    assert result["hurricane_vulnerability_score"] == roof_with_hurricane_score['hurricaneScore']['vulnerabilityScore']

    assert "hurricane_vulnerability_probability" in result
    assert result["hurricane_vulnerability_probability"] == roof_with_hurricane_score['hurricaneScore']['vulnerabilityProbability']

    assert "hurricane_vulnerability_rate_factor" in result
    assert result["hurricane_vulnerability_rate_factor"] == roof_with_hurricane_score['hurricaneScore']['vulnerabilityRateFactor']

    # Verify modelInputFeatures are NOT flattened
    assert "hurricane_model_flat_present" not in result
    assert "hurricane_model_shingle_present" not in result


def test_hurricane_score_different_values(all_roofs_with_hurricane_score):
    """Test with a different real roof that has different hurricaneScore values."""

    # Get the second roof (if available, otherwise use first)
    roof_feature = all_roofs_with_hurricane_score[1] if len(all_roofs_with_hurricane_score) > 1 else all_roofs_with_hurricane_score[0]

    roof = {
        'feature_id': roof_feature['id'],
        'class_id': roof_feature['classId'],
        'hurricane_score': roof_feature['hurricaneScore'],
        'attributes': roof_feature.get('attributes', [])
    }

    result = flatten_roof_attributes([roof], country="us")

    # Verify values are extracted correctly from real data
    assert result["hurricane_vulnerability_score"] == roof_feature['hurricaneScore']['vulnerabilityScore']
    assert result["hurricane_vulnerability_probability"] == roof_feature['hurricaneScore']['vulnerabilityProbability']
    assert result["hurricane_vulnerability_rate_factor"] == roof_feature['hurricaneScore']['vulnerabilityRateFactor']


def test_handles_missing_hurricane_score_gracefully():
    """Test that the function handles missing hurricaneScore gracefully."""

    # Create a roof without hurricaneScore (simulating building-only API response)
    roof = {
        'feature_id': 'test_feature_no_score',
        'class_id': ROOF_ID,
        'attributes': []
    }

    # Run flatten_roof_attributes - should not crash even without hurricaneScore
    result = flatten_roof_attributes([roof], country="us")

    # Verify no hurricaneScore fields in result
    assert "hurricane_vulnerability_score" not in result
    assert "hurricane_vulnerability_probability" not in result
    assert "hurricane_vulnerability_rate_factor" not in result


def test_hurricane_score_with_camel_case(roof_with_hurricane_score):
    """Test that camelCase field names are handled (as returned directly from API)."""

    # Test with camelCase as it appears in raw API response
    roof = {
        'feature_id': roof_with_hurricane_score['id'],
        'class_id': roof_with_hurricane_score['classId'],
        'hurricaneScore': roof_with_hurricane_score['hurricaneScore'],  # camelCase
        'attributes': roof_with_hurricane_score.get('attributes', [])
    }

    result = flatten_roof_attributes([roof], country="us")

    # Should still extract correctly from real data
    assert result["hurricane_vulnerability_score"] == roof_with_hurricane_score['hurricaneScore']['vulnerabilityScore']
    assert result["hurricane_vulnerability_probability"] == roof_with_hurricane_score['hurricaneScore']['vulnerabilityProbability']
    assert result["hurricane_vulnerability_rate_factor"] == roof_with_hurricane_score['hurricaneScore']['vulnerabilityRateFactor']


def test_hurricane_score_with_partial_data(roof_with_hurricane_score):
    """Test hurricaneScore extraction when only some fields are present."""

    # Extract real data but remove some fields to simulate partial response
    partial_score = {
        'vulnerabilityScore': roof_with_hurricane_score['hurricaneScore']['vulnerabilityScore']
        # Missing vulnerabilityProbability and vulnerabilityRateFactor
    }

    roof = {
        'feature_id': roof_with_hurricane_score['id'],
        'class_id': roof_with_hurricane_score['classId'],
        'hurricane_score': partial_score,
        'attributes': []
    }

    result = flatten_roof_attributes([roof], country="us")

    # vulnerabilityScore should be present (from real data)
    assert result["hurricane_vulnerability_score"] == roof_with_hurricane_score['hurricaneScore']['vulnerabilityScore']

    # But other fields should not be present
    assert "hurricane_vulnerability_probability" not in result
    assert "hurricane_vulnerability_rate_factor" not in result


def test_hurricane_score_invalid_data():
    """Test that invalid hurricaneScore data is handled gracefully."""

    # Test with scalar hurricaneScore (should be a dict)
    roof_scalar = {
        'feature_id': 'test_feature_1',
        'class_id': ROOF_ID,
        'hurricane_score': 5,  # Invalid - should be a dict
        'attributes': []
    }

    result_scalar = flatten_roof_attributes([roof_scalar], country="us")
    assert "hurricane_vulnerability_score" not in result_scalar

    # Test with None hurricaneScore
    roof_none = {
        'feature_id': 'test_feature_2',
        'class_id': ROOF_ID,
        'hurricane_score': None,
        'attributes': []
    }

    result_none = flatten_roof_attributes([roof_none], country="us")
    assert "hurricane_vulnerability_score" not in result_none


def test_hurricane_score_and_roof_spotlight_index_together(roof_with_both_scores):
    """
    Test that hurricaneScore and roofSpotlightIndex can coexist on the same roof.

    Uses real API data from a call with include=["hurricaneScore", "roofSpotlightIndex"].
    """

    # Use real roof with BOTH scores from the same API call
    roof = {
        'feature_id': roof_with_both_scores['id'],
        'class_id': roof_with_both_scores['classId'],
        'hurricane_score': roof_with_both_scores['hurricaneScore'],
        'roof_spotlight_index': roof_with_both_scores['roofSpotlightIndex'],
        'attributes': roof_with_both_scores.get('attributes', [])
    }

    result = flatten_roof_attributes([roof], country="us")

    # Both should be present with real values from the API
    assert result["hurricane_vulnerability_score"] == roof_with_both_scores['hurricaneScore']['vulnerabilityScore']
    assert result["hurricane_vulnerability_probability"] == roof_with_both_scores['hurricaneScore']['vulnerabilityProbability']
    assert result["hurricane_vulnerability_rate_factor"] == roof_with_both_scores['hurricaneScore']['vulnerabilityRateFactor']
    assert result["roof_spotlight_index"] == roof_with_both_scores['roofSpotlightIndex']['value']
    assert result["roof_spotlight_index_confidence"] == roof_with_both_scores['roofSpotlightIndex']['confidence']


def test_rollup_with_hurricane_score(hurricane_score_payload):
    """
    Test parcel rollup with hurricaneScore to generate CSV output.

    This test creates a full rollup CSV file showing how hurricaneScore
    and roofSpotlightIndex appear in the flattened output.
    Uses real API data from the payload fixture.
    """
    from nmaipy.parcels import parcel_rollup
    from shapely.geometry import shape, box

    # Use payload from fixture
    payload = hurricane_score_payload

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

        # Add hurricaneScore if present (from real API response)
        if 'hurricaneScore' in feature:
            feat_dict['hurricane_score'] = feature['hurricaneScore']

        # Add roofSpotlightIndex if present (from real API response)
        if 'roofSpotlightIndex' in feature:
            feat_dict['roof_spotlight_index'] = feature['roofSpotlightIndex']

        features.append(feat_dict)

    # Create features GeoDataFrame
    features_gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')
    features_gdf['aoi_id'] = 'test_parcel_hurricane'
    features_gdf = features_gdf.set_index(AOI_ID_COLUMN_NAME)

    # Create parcels GeoDataFrame
    parcel_geom = box(payload['extentMinX'], payload['extentMinY'],
                       payload['extentMaxX'], payload['extentMaxY'])
    parcels_gdf = gpd.GeoDataFrame(
        {'aoi_id': ['test_parcel_hurricane'], 'geometry': [parcel_geom]},
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
    output_file = data_directory / "test_parcels_rollup_hurricane_score.csv"
    rollup_df.to_csv(output_file)

    # Verify hurricaneScore columns are present in primary roof
    assert 'primary_roof_hurricane_vulnerability_score' in rollup_df.columns
    assert 'primary_roof_hurricane_vulnerability_probability' in rollup_df.columns
    assert 'primary_roof_hurricane_vulnerability_rate_factor' in rollup_df.columns

    # Verify the hurricaneScore values (use whatever roof was selected as primary)
    primary_score = rollup_df['primary_roof_hurricane_vulnerability_score'].iloc[0]
    assert primary_score in [4, 5], f"Expected hurricaneScore 4 or 5, got {primary_score}"

    # Probability should be present and reasonable
    primary_prob = rollup_df['primary_roof_hurricane_vulnerability_probability'].iloc[0]
    assert 0 < primary_prob < 1, f"Expected probability between 0 and 1, got {primary_prob}"

    # Rate factor should be present
    assert 'primary_roof_hurricane_vulnerability_rate_factor' in rollup_df.columns

    # RSI should also be present (from real API response)
    assert 'primary_roof_roof_spotlight_index' in rollup_df.columns
    # Real RSI value from the API
    primary_rsi = rollup_df['primary_roof_roof_spotlight_index'].iloc[0]
    assert primary_rsi > 0, f"Expected positive RSI value, got {primary_rsi}"

    assert 'primary_roof_roof_spotlight_index_confidence' in rollup_df.columns
    # Confidence should be present and reasonable
    rsi_conf = rollup_df['primary_roof_roof_spotlight_index_confidence'].iloc[0]
    assert 0 < rsi_conf <= 1, f"Expected confidence between 0 and 1, got {rsi_conf}"
