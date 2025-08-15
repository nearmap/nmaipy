"""Test for null datatype bug specifically with poles pack."""
import json
import os
import shutil
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq

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
    features_files = list(final_dir.glob('*_features.parquet'))
    payload_files = list(test_output_dir.glob('cache/*/*/*.json'))

    assert len(features_files) == 1, f"Expected one final feature file"
    assert len(payload_files) == 1, f"Expected one cache file, found {len(payload_files)}."
    
    # Read the features file
    gdf = gpd.read_parquet(features_files[0])
    with open(payload_files[0], 'r') as f:
        payload = json.load(f)

    num_features_expected = {
        ROOF_ID: 19,
        TRAMPOLINE_ID: 1,
        CLASS_1054_POLE: 3,
    }

    for class_id in num_features_expected.keys():
        gdf_class = gdf[gdf.class_id == class_id]
        assert len(gdf_class) == num_features_expected[class_id], f"Expected {num_features_expected[class_id]} features for class {class_id}, found {len(gdf_class)}"
        class_geom_types = gdf_class.geometry.geom_type.unique()
        assert len(class_geom_types) == 1, f"Expected single geometry type for class {class_id}, found: {class_geom_types}"
        if class_id == CLASS_1054_POLE:
            assert 'Point' in class_geom_types, f"Expected Point geometry for class {class_id}, found: {class_geom_types}"
        else:
            assert 'Polygon' in class_geom_types, f"Expected Polygon geometry for class {class_id}, found: {class_geom_types}"
        # assert gdf_class['belongs_to_parcel'].notna().all(), "belongs_to_parcel contains null values"
        
    assert len(gdf) == sum(num_features_expected.values()), "Wrong number of features found"

    for feature in payload["features"]:
        assert "belongsToParcel" in feature, f"Payload for {feature['description']} does not contain belongsToParcel key in feature"
    assert 'belongs_to_parcel' in gdf.columns, "belongs_to_parcel column not found in features"
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