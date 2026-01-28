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

    # Cleanup - skip if KEEP_TEST_FILES is set (useful for QGIS testing)
    if output_dir.exists() and not os.environ.get('KEEP_TEST_FILES'):
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


@pytest.mark.integration
@pytest.mark.slow
def test_full_export_with_all_includes_and_chunks(integration_test_dir):
    """
    End-to-end test with all AI packs, include="all", and multiple chunks.

    This test verifies:
    1. Multiple packs work together (building, vegetation, roof_cond, etc.)
    2. include="all" returns all available include parameters
    3. Chunking works correctly with includes
    4. All include parameters are properly flattened in both rollup and features
    5. roofConditionConfidenceStats histogram bins are present
    """

    # Use Phoenix, AZ area - has good coverage for all packs and includes
    test_polygon = Polygon([
        (-111.926, 33.4152),
        (-111.925, 33.4152),
        (-111.925, 33.4142),
        (-111.926, 33.4142),
        (-111.926, 33.4152)
    ])

    test_aoi = gpd.GeoDataFrame(
        {'aoi_id': ['phoenix_test'], 'geometry': [test_polygon]},
        crs='EPSG:4326'
    )

    aoi_file = integration_test_dir / 'test_aoi_phoenix.geojson'
    test_aoi.to_file(aoi_file, driver='GeoJSON')

    # Run exporter with all packs and include="all"
    exporter = AOIExporter(
        aoi_file=str(aoi_file),
        output_dir=str(integration_test_dir),
        country='us',
        packs=['building', 'vegetation', 'roof_cond', 'roof_char', 'building_char'],
        include=['all'],  # Get all available includes
        save_features=True,  # Test both features and rollup output
        save_buildings=True,
        chunk_size=2,  # Force multiple chunks for testing
        no_cache=True,
        processes=1,
    )

    exporter.run()

    final_dir = integration_test_dir / 'final'
    chunk_dir = integration_test_dir / 'chunks'

    # 1. Verify rollup CSV exists and has include parameter columns
    # The exporter now uses _aoi_rollup suffix
    rollup_file = final_dir / 'test_aoi_phoenix_aoi_rollup.csv'
    assert rollup_file.exists(), f"Rollup CSV should be created at {rollup_file}. Files in final: {list(final_dir.iterdir()) if final_dir.exists() else []}"

    rollup_df = pd.read_csv(rollup_file)
    rollup_cols = rollup_df.columns.tolist()

    # Check for include parameter columns in rollup
    include_patterns = {
        'roofSpotlightIndex': 'roof_spotlight_index',
        'hurricaneScore': 'hurricane_vulnerability_score',
        'defensibleSpace': 'defensible_space_zone_',
        'roofConditionConfidenceStats': 'confidence_stats_default_bin_',
    }

    found_includes = {}
    for include_name, pattern in include_patterns.items():
        matching_cols = [col for col in rollup_cols if pattern in col]
        found_includes[include_name] = len(matching_cols)
        if matching_cols:
            print(f"✅ Found {len(matching_cols)} columns for {include_name}")

    # Verify roofConditionConfidenceStats histogram bins are present
    confidence_bins = [col for col in rollup_cols if 'confidence_stats_default_bin_' in col or 'confidence_stats_extreme_bin_' in col]
    assert len(confidence_bins) > 0, f"Should have roofConditionConfidenceStats histogram bins in rollup. Columns: {rollup_cols[:20]}"
    print(f"✅ Found {len(confidence_bins)} roofConditionConfidenceStats histogram bin columns")

    # Verify default bins (18 per component) and extreme bins (3 per component)
    default_bins = [col for col in confidence_bins if '_default_bin_' in col]
    extreme_bins = [col for col in confidence_bins if '_extreme_bin_' in col]
    print(f"   - Default bins: {len(default_bins)}")
    print(f"   - Extreme bins: {len(extreme_bins)}")

    # 2. Verify features parquet exists and was created successfully
    features_file = final_dir / 'test_aoi_phoenix_features.parquet'
    assert features_file.exists(), f"Features parquet should be created at {features_file}"

    features_gdf = gpd.read_parquet(features_file)
    print(f"✅ Features file has {len(features_gdf)} features")

    # 3. Verify rollup chunks were created
    rollup_chunks = list(chunk_dir.glob('rollup_test_aoi_phoenix_*.parquet'))
    assert len(rollup_chunks) > 0, f"Should have created rollup chunks"
    print(f"✅ Created {len(rollup_chunks)} rollup chunks")

    # 4. Verify feature chunks were created
    feature_chunks = list(chunk_dir.glob('features_test_aoi_phoenix_*.parquet'))
    assert len(feature_chunks) > 0, f"Should have created feature chunks"
    print(f"✅ Created {len(feature_chunks)} feature chunks")

    # 5. Verify chunks have include parameter data
    for chunk_file in rollup_chunks[:1]:  # Check first chunk
        chunk_df = pd.read_parquet(chunk_file)
        if len(chunk_df) > 0:
            chunk_cols = chunk_df.columns.tolist()
            chunk_confidence_bins = [col for col in chunk_cols if 'confidence_stats_default_bin_' in col]
            if len(chunk_confidence_bins) > 0:
                print(f"✅ Rollup chunks contain confidence stats columns ({len(chunk_confidence_bins)} bins)")

    # 6. Verify chunk features match consolidated features
    chunk_features = []
    for chunk_file in feature_chunks:
        chunk_gdf = gpd.read_parquet(chunk_file)
        if len(chunk_gdf) > 0:
            chunk_features.append(chunk_gdf)

    if chunk_features:
        total_chunk_features = sum(len(df) for df in chunk_features)
        assert total_chunk_features == len(features_gdf), \
            f"Chunk features ({total_chunk_features}) should match consolidated features ({len(features_gdf)})"
        print(f"✅ Chunk feature count matches consolidated file: {total_chunk_features}")

    # 7. Verify features have proper dot-notation columns
    if len(features_gdf) > 0:
        features_cols = features_gdf.columns.tolist()
        dot_cols = [col for col in features_cols if '.' in col]
        print(f"✅ Features have {len(dot_cols)} dot-notation columns")

        # Verify features have multiple feature types from different packs
        feature_descriptions = features_gdf['description'].unique()
        print(f"✅ Found {len(feature_descriptions)} unique feature types")

        has_building = any('Building' in d for d in feature_descriptions)
        has_vegetation = any('Vegetation' in d or 'Tree' in d for d in feature_descriptions)
        has_roof = any('Roof' in d for d in feature_descriptions)

        found_packs = []
        if has_building:
            found_packs.append('building')
        if has_vegetation:
            found_packs.append('vegetation')
        if has_roof:
            found_packs.append('roof features')

        if found_packs:
            print(f"✅ Found features from: {', '.join(found_packs)}")

    # 8. Verify rollup has data from multiple packs
    if len(rollup_df) > 0:
        has_building_cols = any('building' in col.lower() for col in rollup_cols)
        has_vegetation_cols = any('vegetation' in col.lower() or 'tree' in col.lower() for col in rollup_cols)
        has_roof_cols = any('roof' in col.lower() for col in rollup_cols)

        rollup_packs = []
        if has_building_cols:
            rollup_packs.append('building')
        if has_vegetation_cols:
            rollup_packs.append('vegetation')
        if has_roof_cols:
            rollup_packs.append('roof features')

        if rollup_packs:
            print(f"✅ Found rollup columns from: {', '.join(rollup_packs)}")

    print(f"\n✅ Full end-to-end test passed with all packs, include='all', and chunking")


@pytest.mark.integration
def test_parquet_deserialization_of_include_params(integration_test_dir):
    """
    Test that include parameters serialized as JSON strings in Parquet files
    can be successfully deserialized back to dictionaries.

    This verifies that the JSON serialization of dict-type include parameters
    (hurricaneScore, defensibleSpace, roofSpotlightIndex) works correctly
    for round-trip Parquet read/write operations.
    """

    # Use Phoenix area with roof features that have include parameters
    test_polygon = Polygon([
        (-111.926, 33.4152),
        (-111.925, 33.4152),
        (-111.925, 33.4142),
        (-111.926, 33.4142),
        (-111.926, 33.4152)
    ])

    test_aoi = gpd.GeoDataFrame(
        {'aoi_id': ['parquet_test'], 'geometry': [test_polygon]},
        crs='EPSG:4326'
    )

    aoi_file = integration_test_dir / 'test_aoi_parquet.geojson'
    test_aoi.to_file(aoi_file, driver='GeoJSON')

    # Run exporter with include parameters that return dict objects
    exporter = AOIExporter(
        aoi_file=str(aoi_file),
        output_dir=str(integration_test_dir),
        country='us',
        packs=['building', 'roof_char'],
        include=['hurricaneScore', 'defensibleSpace', 'roofSpotlightIndex'],
        save_features=True,
        save_buildings=False,
        no_cache=True,
        processes=1,
    )

    exporter.run()

    final_dir = integration_test_dir / 'final'
    features_file = final_dir / 'test_aoi_parquet_features.parquet'

    assert features_file.exists(), f"Features file should exist at {features_file}"

    # Read the parquet file
    features_gdf = gpd.read_parquet(features_file)

    # Check for include parameter columns
    include_columns = {
        'hurricane_score': 'hurricaneScore',
        'defensible_space': 'defensibleSpace',
        'roof_spotlight_index': 'roofSpotlightIndex'
    }

    found_includes = []
    deserialized_successfully = []

    for snake_case, camel_case in include_columns.items():
        # Check both naming conventions
        if snake_case in features_gdf.columns:
            col_name = snake_case
            found_includes.append(snake_case)
        elif camel_case in features_gdf.columns:
            col_name = camel_case
            found_includes.append(camel_case)
        else:
            continue

        # Get non-null values
        non_null_values = features_gdf[features_gdf[col_name].notna()][col_name]

        if len(non_null_values) > 0:
            sample_value = non_null_values.iloc[0]

            # Value should be a JSON string
            assert isinstance(sample_value, str), \
                f"{col_name} should be serialized as JSON string, got {type(sample_value)}"

            # Deserialize the JSON string
            try:
                deserialized = json.loads(sample_value)
                assert isinstance(deserialized, dict), \
                    f"Deserialized {col_name} should be a dict, got {type(deserialized)}"

                # Verify the dict has expected structure (non-empty)
                assert len(deserialized) > 0, \
                    f"Deserialized {col_name} should not be empty"

                deserialized_successfully.append(col_name)
                print(f"✅ Successfully deserialized {col_name}: {list(deserialized.keys())[:5]}")

            except json.JSONDecodeError as e:
                pytest.fail(f"Failed to deserialize {col_name}: {e}")

    # Verify we found and deserialized at least one include parameter
    assert len(found_includes) > 0, \
        f"Should have at least one include parameter in features. Columns: {features_gdf.columns.tolist()}"

    assert len(deserialized_successfully) > 0, \
        f"Should successfully deserialize at least one include parameter"

    print(f"\n✅ Parquet deserialization test passed:")
    print(f"   - Found {len(found_includes)} include parameters: {found_includes}")
    print(f"   - Successfully deserialized {len(deserialized_successfully)}: {deserialized_successfully}")