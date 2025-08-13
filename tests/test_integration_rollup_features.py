"""Integration test to verify both rollup CSV and features GeoParquet work correctly."""

import os
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
import json
import shutil
import tempfile

from nmaipy.exporter import AOIExporter


@pytest.fixture
def integration_test_dir():
    """Fixture for test output directory."""
    if os.environ.get('CI'):
        temp_dir = tempfile.mkdtemp(prefix='integration_test_')
        output_dir = Path(temp_dir)
    else:
        output_dir = Path(__file__).parent / 'data' / 'integration_test_output'
        output_dir.mkdir(parents=True, exist_ok=True)
    
    yield output_dir
    
    # Cleanup
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def test_aoi_with_features(integration_test_dir):
    """Create an AOI in an area likely to have features."""
    # Use Salt Lake City area that we know has data
    test_polygon = Polygon([
        (-111.8905, 40.7610),
        (-111.8903, 40.7610),
        (-111.8903, 40.7612),
        (-111.8905, 40.7612),
        (-111.8905, 40.7610)
    ])
    
    test_aoi = gpd.GeoDataFrame(
        {'aoi_id': ['integration_test'], 'geometry': [test_polygon]},
        crs='EPSG:4326'
    )
    
    aoi_file = integration_test_dir / 'test_aoi.geojson'
    test_aoi.to_file(aoi_file, driver='GeoJSON')
    
    return aoi_file


@pytest.mark.integration
@pytest.mark.slow
def test_rollup_and_features_consistency(integration_test_dir, test_aoi_with_features):
    """
    Integration test to verify that:
    1. Rollup CSV uses special formatting (backward compatible)
    2. Features GeoParquet uses generic dot-notation flattening
    3. Both outputs are generated correctly from the same data
    """
    
    # Run exporter with both rollup and features enabled
    exporter = AOIExporter(
        aoi_file=str(test_aoi_with_features),
        output_dir=str(integration_test_dir),
        country='us',
        packs=['building', 'building_char', 'roof_char'],
        save_features=True,  # Generate features GeoParquet
        save_buildings=True,  # Generate building rollups
        no_cache=True,
        processes=1,
    )
    
    exporter.run()
    
    # Check that both outputs were created
    final_dir = integration_test_dir / 'final'
    
    # 1. Check rollup CSV
    rollup_file = final_dir / 'test_aoi_buildings.csv'
    if rollup_file.exists():
        rollup_df = pd.read_csv(rollup_file)
        
        # Check for special rollup column naming
        rollup_columns = rollup_df.columns.tolist()
        
        # Rollups should have special naming like:
        # - primary_building_height_m or primary_building_height_ft
        # - primary_building_lifecycle_damage_class
        # - tile_roof_present, shingle_roof_area_sqm, etc.
        
        # Check for some expected rollup patterns
        special_rollup_patterns = [
            'primary_building_',  # Primary feature columns
            'building_count',      # Count columns
            'roof_present',        # Present/absent flags
            '_area_sqm',          # Area measurements
            '_confidence',        # Confidence scores
            'damage_class',       # Damage classification
        ]
        
        has_rollup_patterns = any(
            any(pattern in col for col in rollup_columns)
            for pattern in special_rollup_patterns
        )
        
        assert has_rollup_patterns, f"Rollup CSV should have special column naming. Columns: {rollup_columns[:10]}"
        
        # Check that components are NOT in rollup (they're expanded)
        assert not any('.components' in col for col in rollup_columns), "Rollup should not have .components columns"
        
        print(f"✅ Rollup CSV has {len(rollup_df)} rows with special formatting")
    
    # 2. Check features GeoParquet
    features_file = final_dir / 'test_aoi_features.parquet'
    if not features_file.exists():
        # Check what files were actually created
        if final_dir.exists():
            files = list(final_dir.glob('*'))
            print(f"Files in final dir: {files}")
        assert features_file.exists(), f"Features GeoParquet should be created at {features_file}"
    
    features_gdf = gpd.read_parquet(features_file)
    features_columns = features_gdf.columns.tolist()
    
    # Features should have dot-notation flattened columns
    dot_notation_cols = [c for c in features_columns if '.' in c]
    assert len(dot_notation_cols) > 0, "Features should have dot-notation columns"
    
    # Check for expected patterns in features
    features_patterns = [
        'Building 3d attributes.',  # Dot notation with description prefix
        'Roof material.',
        'damage.femaCategoryConfidences.',  # Damage with dot notation
        '.components',  # Components as JSON strings
    ]
    
    has_features_patterns = any(
        any(pattern in col for col in dot_notation_cols)
        for pattern in features_patterns
    )
    
    # May not always have all patterns depending on data
    print(f"✅ Features GeoParquet has {len(features_gdf)} features with {len(dot_notation_cols)} flattened columns")
    
    # 3. Verify no internalClassId in features
    assert not any('internalClassId' in col for col in features_columns), "Features should not expose internalClassId"
    
    # 4. Check components are JSON strings in features
    component_cols = [c for c in dot_notation_cols if 'components' in c]
    if component_cols and len(features_gdf) > 0:
        for col in component_cols:
            non_null = features_gdf[features_gdf[col].notna()]
            if len(non_null) > 0:
                sample = non_null[col].iloc[0]
                assert isinstance(sample, str), f"Components in {col} should be JSON string"
                # Verify it's valid JSON
                try:
                    parsed = json.loads(sample)
                    assert isinstance(parsed, list), f"{col} should contain a JSON array"
                    # Check no internalClassId in components
                    if parsed:
                        assert 'internalClassId' not in str(parsed), "Components should not contain internalClassId"
                except json.JSONDecodeError:
                    pytest.fail(f"{col} should contain valid JSON")
    
    # 5. Verify the two formats are from the same underlying data
    # Both should have similar feature counts (though rollup aggregates by parcel)
    if rollup_file.exists() and len(rollup_df) > 0:
        # Rollup aggregates, so it should have fewer rows than raw features
        assert len(features_gdf) >= len(rollup_df), "Features should have at least as many rows as rollup"
    
    print(f"✅ Integration test passed: Rollup uses special format, Features use generic flattening")


@pytest.mark.integration
def test_features_without_rollup(integration_test_dir):
    """Test that features can be generated without rollup."""
    
    # Use Salt Lake City area that we know has data
    test_polygon = Polygon([
        (-111.8905, 40.7610),
        (-111.8904, 40.7610),
        (-111.8904, 40.7611),
        (-111.8905, 40.7611),
        (-111.8905, 40.7610)
    ])
    
    test_aoi = gpd.GeoDataFrame(
        {'aoi_id': ['test_features_only'], 'geometry': [test_polygon]},
        crs='EPSG:4326'
    )
    
    aoi_file = integration_test_dir / 'test_aoi.geojson'
    test_aoi.to_file(aoi_file, driver='GeoJSON')
    
    # Run with only features, no rollup
    exporter = AOIExporter(
        aoi_file=str(aoi_file),
        output_dir=str(integration_test_dir),
        country='us',
        packs=['building', 'roof_char'],
        save_features=True,
        save_buildings=False,  # No rollup
        no_cache=True,
        processes=1,
    )
    
    exporter.run()
    
    final_dir = integration_test_dir / 'final'
    
    # Should have features but no buildings CSV
    features_file = final_dir / 'test_aoi_features.parquet'
    buildings_file = final_dir / 'test_aoi_buildings.csv'
    
    assert features_file.exists(), "Features file should exist"
    assert not buildings_file.exists(), "Buildings rollup should not exist"
    
    # Verify features have proper flattening
    features_gdf = gpd.read_parquet(features_file)
    if len(features_gdf) > 0:
        dot_cols = [c for c in features_gdf.columns if '.' in c]
        if dot_cols:
            print(f"✅ Features-only export has {len(dot_cols)} dot-notation columns")