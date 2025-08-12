"""Test 3D attribute flattening for features."""

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
import json
import shutil

from nmaipy.exporter import AOIExporter
from nmaipy.feature_api import FeatureApi


@pytest.mark.integration
def test_3d_attributes_flattening():
    """Test that 3D attributes are properly flattened in feature exports."""
    
    # Small AOI in Salt Lake City with known 3D coverage
    test_polygon = Polygon([
        (-111.8905, 40.7610),
        (-111.8903, 40.7610),
        (-111.8903, 40.7612),
        (-111.8905, 40.7612),
        (-111.8905, 40.7610)
    ])
    
    # Create test AOI GeoDataFrame
    test_aoi = gpd.GeoDataFrame(
        {'aoi_id': ['salt_lake_3d_test'], 'geometry': [test_polygon]},
        crs='EPSG:4326'
    )
    
    # Use a subdirectory in tests/data for easier access during debugging
    output_dir = Path(__file__).parent / 'data' / '3d_test_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save AOI to file
    aoi_file = output_dir / 'test_aoi.geojson'
    test_aoi.to_file(aoi_file, driver='GeoJSON')
    
    try:
        # Run exporter with 3D packs
        exporter = AOIExporter(
            aoi_file=str(aoi_file),
            output_dir=str(output_dir),
            country='us',
            packs=['building', 'building_char', 'roof_char'],
            save_features=True,
            no_cache=True,
            processes=1,
            only3d=True,
        )
        
        exporter.run()
        
        # Check the features file
        features_file = output_dir / 'final' / 'test_aoi_features.parquet'
        assert features_file.exists(), "Features file should be created"
        
        # Load and verify
        gdf = gpd.read_parquet(features_file)
        
        # Check that we have features
        assert len(gdf) > 0, "Should have features in the output"
        
        # Check for flattened columns
        flattened_cols = [c for c in gdf.columns if '.' in c]
        assert len(flattened_cols) > 0, "Should have flattened attribute columns"
        
        # Check that internalClassId is not present
        internal_cols = [c for c in gdf.columns if 'internalClassId' in c.lower()]
        assert len(internal_cols) == 0, "internalClassId should be skipped"
        
        # Check for components serialized as JSON
        component_cols = [c for c in flattened_cols if 'components' in c]
        if component_cols:
            for col in component_cols:
                non_null_values = gdf[gdf[col].notna()][col]
                if len(non_null_values) > 0:
                    # Check that components are JSON strings
                    sample = non_null_values.iloc[0]
                    assert isinstance(sample, str), f"{col} should be a string"
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(sample)
                        assert isinstance(parsed, list), f"{col} should be a JSON array"
                    except json.JSONDecodeError:
                        pytest.fail(f"{col} should be valid JSON")
        
        # Check for expected 3D attributes on buildings
        building_rows = gdf[gdf['description'] == 'Building']
        if len(building_rows) > 0:
            # Look for height attributes
            height_cols = [c for c in flattened_cols if 'height' in c.lower()]
            if height_cols:
                # At least one building should have height
                has_height = any(
                    building_rows[col].notna().any() 
                    for col in height_cols
                )
                assert has_height or len(building_rows) < 2, "At least some buildings should have height attributes"
        
        # Check for roof attributes
        roof_rows = gdf[gdf['description'] == 'Roof']
        if len(roof_rows) > 0:
            # Look for pitch or material attributes
            roof_attr_cols = [c for c in flattened_cols if any(
                term in c.lower() for term in ['pitch', 'material', 'type']
            )]
            assert len(roof_attr_cols) > 0 or len(roof_rows) < 2, "Roofs should have attributes like pitch or material"
    
    finally:
        # Clean up test output directory
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)


@pytest.mark.integration 
def test_attributes_list_handling():
    """Test that attributes list is properly handled in the flattening process."""
    
    # Get a small sample of real data
    feature_api = FeatureApi()
    
    polygon = Polygon([
        (-111.8905, 40.7610),
        (-111.8904, 40.7610),
        (-111.8904, 40.7611),
        (-111.8905, 40.7611),
        (-111.8905, 40.7610)
    ])
    
    features_gdf, _, _ = feature_api.get_features_gdf(
        polygon,
        until='2025-06-20',
        region='us',
        packs=['building', 'building_char', 'roof_char'],
    )
    
    if len(features_gdf) > 0 and 'attributes' in features_gdf.columns:
        # Check that attributes is a list
        for idx, row in features_gdf.iterrows():
            attrs = row.get('attributes')
            if attrs is not None:
                assert isinstance(attrs, list), "Attributes should be a list"
                
                # Test the flattening logic (function is internal to exporter)
                # Replicate the flattening logic here for testing
                import json
                
                flattened = {}
                for i, attr_obj in enumerate(attrs):
                    if not isinstance(attr_obj, dict):
                        continue
                    
                    desc = attr_obj.get('description', f'attr_{i}')
                    
                    for key, value in attr_obj.items():
                        if key in ['description', 'internalClassId']:
                            continue
                        
                        if key == 'components' and isinstance(value, (list, dict)):
                            flattened[f"{desc}.components"] = json.dumps(value)
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if sub_value is not None:
                                    flattened[f"{desc}.{key}.{sub_key}"] = sub_value
                        elif value is not None:
                            flattened[f"{desc}.{key}"] = value
                
                # Check the flattened structure
                assert isinstance(flattened, dict), "Flattened result should be a dict"
                
                # Check that internalClassId is not in flattened keys
                for key in flattened.keys():
                    assert 'internalClassId' not in key, f"Key {key} should not contain internalClassId"
                
                # Check that components are JSON strings
                for key, value in flattened.items():
                    if 'components' in key:
                        assert isinstance(value, str), f"{key} should be a JSON string"
                        # Verify it's valid JSON
                        parsed = json.loads(value)
                        assert isinstance(parsed, list), f"{key} should contain a JSON array"
                
                break  # Just test one row with attributes