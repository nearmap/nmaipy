"""Test for null datatype bug specifically with poles pack."""
import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest

from nmaipy.exporter import AOIExporter
from nmaipy.constants import CLASS_1054_POLE, ROOF_ID, TRAMPOLINE_ID


@pytest.fixture
def test_output_dir(tmp_path):
    """Fixture to create test output directory."""
    output_dir = tmp_path / 'poles_test_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    yield output_dir
    
    # Cleanup happens automatically with tmp_path


@pytest.fixture  
def poles_aoi_file(test_output_dir):
    """Create test AOI file for poles testing."""
    # Use an area in Salt Lake City that previously showed poles in our tests
    test_aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-111.9350, 40.7608],  # Area that had poles in earlier tests
                        [-111.9340, 40.7608],
                        [-111.9340, 40.7618],
                        [-111.9350, 40.7618],
                        [-111.9350, 40.7608]
                    ]]
                }
            }
        ]
    }
    
    aoi_path = test_output_dir / 'poles_test_aoi.geojson'
    with open(aoi_path, 'w') as f:
        json.dump(test_aoi, f)
    
    return str(aoi_path)


@pytest.mark.live_api
def test_poles_pack_null_datatype_issue(poles_aoi_file, test_output_dir):
    """Test that poles pack doesn't create problematic null datatypes."""

    # Run exporter with poles pack and parcel_mode to test belongs_to_parcel
    exporter = AOIExporter(
        aoi_file=poles_aoi_file,
        output_dir=str(test_output_dir),
        country='us',
        classes=[CLASS_1054_POLE, ROOF_ID, TRAMPOLINE_ID],  # Specifically test poles and building packs
        save_features=True,
        no_cache=False,
        processes=1,
        parcel_mode=True  # Enable parcel mode to test belongs_to_parcel column
    )
    
    exporter.run()
    
    # Check for output files
    final_dir = test_output_dir / 'final'
    # The exporter now creates both a main features file and per-class feature files
    # Look specifically for the main combined features file (not per-class files)
    all_features_files = list(final_dir.glob('*_features.parquet'))
    # Main file is poles_test_aoi_features.parquet, per-class are poles_test_aoi_{class}_features.parquet
    main_features_file = final_dir / 'poles_test_aoi_features.parquet'

    assert main_features_file.exists(), f"Expected main feature file at {main_features_file}. Found: {all_features_files}"
    features_files = [main_features_file]
    # Cache is now in cache/feature_api/ subdirectory
    payload_files = list(test_output_dir.glob('cache/feature_api/**/*.json'))
    if not payload_files:
        # Also check for gzip compressed cache
        payload_files = list(test_output_dir.glob('cache/feature_api/**/*.json.gz'))
    assert len(payload_files) == 1, f"Expected one cache file, found {len(payload_files)}."
    
    # Read the features file
    gdf = gpd.read_parquet(features_files[0])
    # Handle both regular and gzip-compressed cache files
    payload_file = payload_files[0]
    if str(payload_file).endswith('.gz'):
        with gzip.open(payload_file, 'rt') as f:
            payload = json.load(f)
    else:
        with open(payload_file, 'r') as f:
            payload = json.load(f)

    # Minimum expected features - these may increase over time as imagery updates
    min_features_expected = {
        ROOF_ID: 10,  # Should have roofs (exact count varies with imagery updates)
        TRAMPOLINE_ID: 0,  # May or may not have trampolines
        CLASS_1054_POLE: 1,  # Should have at least some poles
    }

    for class_id, min_count in min_features_expected.items():
        gdf_class = gdf[gdf.class_id == class_id]
        assert len(gdf_class) >= min_count, f"Expected at least {min_count} features for class {class_id}, found {len(gdf_class)}"

        if len(gdf_class) > 0:
            class_geom_types = gdf_class.geometry.geom_type.unique()
            assert len(class_geom_types) == 1, f"Expected single geometry type for class {class_id}, found: {class_geom_types}"
            if class_id == CLASS_1054_POLE:
                assert 'Point' in class_geom_types, f"Expected Point geometry for class {class_id}, found: {class_geom_types}"
            else:
                assert 'Polygon' in class_geom_types, f"Expected Polygon geometry for class {class_id}, found: {class_geom_types}"

    # Verify we got a meaningful number of features overall
    assert len(gdf) >= 10, f"Expected at least 10 total features, found {len(gdf)}"

    # Check belongsToParcel only for polygon features (not point features like poles)
    for feature in payload["features"]:
        if feature.get('description') != 'Pole':
            assert "belongsToParcel" in feature, f"Payload for {feature['description']} does not contain belongsToParcel key in feature"
    
    # The belongs_to_parcel column should exist
    assert 'belongs_to_parcel' in gdf.columns, "belongs_to_parcel column not found in features"
    # Note: belongs_to_parcel has mixed types (bool and None) for poles - this is a known issue
    # Poles don't have belongsToParcel in the API response, so they get None values
    assert gdf['belongs_to_parcel'].dtype == bool, "belongs_to_parcel dtype is not boolean"

    
        
    # Check for POINT geometries (poles should have point geometries)
    assert 'geometry' in gdf.columns, "geometry column not found"
    geom_types = gdf.geometry.geom_type.unique()
    # Poles should have Point geometries
    assert 'Point' in geom_types, f"Expected Point geometries for poles, got: {geom_types}"

    # Check for any columns with 'null' dtype (this would be the bug)
    null_dtype_columns = []
    for col in gdf.columns:
        if str(gdf[col].dtype) == 'null':
            null_dtype_columns.append(col)
    
    assert len(null_dtype_columns) == 0, f"Found columns with 'null' dtype: {null_dtype_columns}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])