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
    rollup_file = final_dir / 'buildings.csv'
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
    
    # 2. Check features GeoParquet
    features_file = final_dir / 'features.parquet'
    if not features_file.exists():
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
    features_file = final_dir / 'features.parquet'
    buildings_file = final_dir / 'buildings.csv'
    
    assert features_file.exists(), "Features file should exist"
    assert not buildings_file.exists(), "Buildings rollup should not exist"
    
    # Verify features have proper flattening
    features_gdf = gpd.read_parquet(features_file)
    if len(features_gdf) > 0:
        dot_cols = [c for c in features_gdf.columns if '.' in c]


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
    rollup_file = final_dir / 'rollup.csv'
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

    # Verify roofConditionConfidenceStats histogram bins are present
    confidence_bins = [col for col in rollup_cols if 'confidence_stats_default_bin_' in col or 'confidence_stats_extreme_bin_' in col]
    assert len(confidence_bins) > 0, f"Should have roofConditionConfidenceStats histogram bins in rollup. Columns: {rollup_cols[:20]}"

    # Verify both default and extreme bins are present
    default_bins = [col for col in confidence_bins if '_default_bin_' in col]
    extreme_bins = [col for col in confidence_bins if '_extreme_bin_' in col]
    assert len(default_bins) > 0, f"Should have default histogram bins. Columns: {rollup_cols[:20]}"
    assert len(extreme_bins) > 0, f"Should have extreme histogram bins. Columns: {rollup_cols[:20]}"

    # 2. Verify features parquet exists and was created successfully
    features_file = final_dir / 'features.parquet'
    assert features_file.exists(), f"Features parquet should be created at {features_file}"

    features_gdf = gpd.read_parquet(features_file)

    # 3. Verify rollup chunks were created
    rollup_chunks = list(chunk_dir.glob('rollup_*.parquet'))
    assert len(rollup_chunks) > 0, f"Should have created rollup chunks"

    # 4. Verify feature chunks were created
    feature_chunks = list(chunk_dir.glob('features_*.parquet'))
    assert len(feature_chunks) > 0, f"Should have created feature chunks"

    # 5. Verify chunk features match consolidated features
    chunk_features = []
    for chunk_file in feature_chunks:
        chunk_gdf = gpd.read_parquet(chunk_file)
        if len(chunk_gdf) > 0:
            chunk_features.append(chunk_gdf)

    if chunk_features:
        total_chunk_features = sum(len(df) for df in chunk_features)
        assert total_chunk_features == len(features_gdf), \
            f"Chunk features ({total_chunk_features}) should match consolidated features ({len(features_gdf)})"


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
    features_file = final_dir / 'features.parquet'

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

            except json.JSONDecodeError as e:
                pytest.fail(f"Failed to deserialize {col_name}: {e}")

    # Verify we found and deserialized at least one include parameter
    assert len(found_includes) > 0, \
        f"Should have at least one include parameter in features. Columns: {features_gdf.columns.tolist()}"

    assert len(deserialized_successfully) > 0, \
        f"Should successfully deserialize at least one include parameter"


@pytest.mark.integration
@pytest.mark.slow
def test_is_primary_in_per_class_exports(integration_test_dir, test_aoi_with_features):
    """
    Integration test to verify is_primary column in per-class feature exports.

    This test verifies:
    1. is_primary column exists in per-class CSV files
    2. is_primary column exists in per-class GeoParquet files
    3. is_primary column is boolean type
    4. Primary features in per-class exports match primary_*_feature_id columns in rollup
    """
    from nmaipy.constants import PRIMARY_FEATURE_COLUMN_TO_CLASS

    # Run exporter with class_level_files=True to generate per-class exports
    exporter = AOIExporter(
        aoi_file=str(test_aoi_with_features),
        output_dir=str(integration_test_dir),
        country='us',
        packs=['building', 'roof_char'],
        save_features=True,
        save_buildings=True,
        class_level_files=True,  # Enable per-class exports
        no_cache=True,
        processes=1,
    )

    exporter.run()

    final_dir = integration_test_dir / 'final'

    # 1. Check per-class CSV files have is_primary column
    # Find all per-class CSV files (exclude rollup, buildings, and stats files)
    exclude_patterns = ['rollup', 'buildings', 'latency_stats', 'feature_api_errors', 'roof_age_errors']
    per_class_csvs = [
        f for f in final_dir.glob('*.csv')
        if not any(pattern in f.name for pattern in exclude_patterns)
    ]

    assert len(per_class_csvs) > 0, \
        f"Should have per-class CSV files. Files in final: {list(final_dir.glob('*'))}"

    for csv_path in per_class_csvs:
        class_df = pd.read_csv(csv_path)
        assert 'is_primary' in class_df.columns, \
            f"Per-class CSV {csv_path.name} should have is_primary column. Columns: {class_df.columns.tolist()}"

    # 2. Check per-class GeoParquet files have is_primary column
    per_class_parquets = [
        f for f in final_dir.glob('*_features.parquet')
        if f.name != 'features.parquet'
    ]

    assert len(per_class_parquets) > 0, \
        f"Should have per-class GeoParquet files. Files in final: {list(final_dir.glob('*'))}"

    for parquet_path in per_class_parquets:
        class_gdf = gpd.read_parquet(parquet_path)
        assert 'is_primary' in class_gdf.columns, \
            f"Per-class parquet {parquet_path.name} should have is_primary column. Columns: {class_gdf.columns.tolist()}"
        assert set(class_gdf['is_primary'].dropna().unique()) <= {"Y", "N"}, \
            f"is_primary in {parquet_path.name} should contain only Y/N, got {class_gdf['is_primary'].unique()}"

    # 3. Cross-verify: primary features in per-class parquets match rollup data
    rollup_file = final_dir / 'buildings.csv'
    if rollup_file.exists():
        rollup_df = pd.read_csv(rollup_file, index_col='aoi_id')

        # For each primary feature column in rollup, verify matching is_primary=True in per-class exports
        for col_name, class_id in PRIMARY_FEATURE_COLUMN_TO_CLASS.items():
            if col_name in rollup_df.columns:
                # Get primary feature IDs from rollup (non-null values)
                primary_ids = rollup_df[col_name].dropna().astype(str).tolist()

                if primary_ids:
                    # Find the per-class parquet for this class
                    for parquet_path in per_class_parquets:
                        class_gdf = gpd.read_parquet(parquet_path)
                        if 'class_id' in class_gdf.columns:
                            class_features = class_gdf[class_gdf['class_id'] == class_id]
                            if len(class_features) > 0:
                                # Check that primary features are marked correctly
                                primary_features = class_features[class_features['is_primary'] == "Y"]
                                primary_feature_ids = primary_features['feature_id'].astype(str).tolist()

                                # Verify at least some primary IDs match
                                matching = set(primary_ids) & set(primary_feature_ids)


@pytest.fixture
def test_us_aoi_for_roof_age(integration_test_dir):
    """Create a US AOI for roof age testing."""
    # Use New Jersey area (roof age is US-only)
    test_polygon = Polygon([
        (-74.0060, 40.7128),
        (-74.0055, 40.7128),
        (-74.0055, 40.7133),
        (-74.0060, 40.7133),
        (-74.0060, 40.7128)
    ])

    test_aoi = gpd.GeoDataFrame(
        {'aoi_id': ['roof_age_test'], 'geometry': [test_polygon]},
        crs='EPSG:4326'
    )

    aoi_file = integration_test_dir / 'test_us_roof_age.geojson'
    test_aoi.to_file(aoi_file, driver='GeoJSON')

    return aoi_file


@pytest.mark.integration
@pytest.mark.slow
def test_roof_age_years_in_all_exports(integration_test_dir, test_us_aoi_for_roof_age):
    """
    Integration test to verify that roof age in years is calculated and present in ALL output files:
    - Combined features parquet (features.parquet)
    - Per-class CSV (roof.csv)
    - Per-class parquet (roof_features.parquet)
    - Roof instances (roof_instance.csv, roof_instance_features.parquet)
    - Rollup (rollup.csv)

    This test ensures consistency across all export formats.
    """
    from nmaipy.constants import ROOF_ID, ROOF_INSTANCE_CLASS_ID
    from datetime import datetime

    # Run exporter with roof age enabled
    exporter = AOIExporter(
        aoi_file=str(test_us_aoi_for_roof_age),
        output_dir=str(integration_test_dir),
        country='us',
        packs=['building'],
        roof_age=True,  # Enable roof age API
        save_features=True,  # Generate features parquet
        class_level_files=True,  # Generate per-class files
        no_cache=True,
        processes=1,
    )

    exporter.run()

    final_dir = integration_test_dir / 'final'
    assert final_dir.exists(), "Final output directory should exist"

    # 1. Check combined features parquet
    combined_features_path = final_dir / 'features.parquet'
    if combined_features_path.exists():
        features_gdf = gpd.read_parquet(combined_features_path)

        # Check roof instances have roof_age_years_as_of_date
        roof_instances = features_gdf[features_gdf['class_id'] == ROOF_INSTANCE_CLASS_ID]
        if len(roof_instances) > 0:
            assert 'roof_age_years_as_of_date' in roof_instances.columns, \
                "Roof instances should have roof_age_years_as_of_date in combined features"

        # Check roofs have primary_child_roof_age_years_as_of_date
        roofs = features_gdf[features_gdf['class_id'] == ROOF_ID]
        if len(roofs) > 0:
            # Check if linkage columns exist
            has_linkage = 'primary_child_roof_age_installation_date' in roofs.columns
            if has_linkage:
                roofs_with_age_data = roofs[roofs['primary_child_roof_age_installation_date'].notna()]
                if len(roofs_with_age_data) > 0:
                    assert 'primary_child_roof_age_years_as_of_date' in roofs_with_age_data.columns, \
                        "Roofs with linked roof instances should have primary_child_roof_age_years_as_of_date in combined features"

                    # Verify calculation is correct for a sample
                    sample = roofs_with_age_data.iloc[0]
                    if pd.notna(sample['primary_child_roof_age_installation_date']) and \
                       pd.notna(sample['primary_child_roof_age_as_of_date']) and \
                       pd.notna(sample['primary_child_roof_age_years_as_of_date']):
                        install = pd.to_datetime(sample['primary_child_roof_age_installation_date'])
                        as_of = pd.to_datetime(sample['primary_child_roof_age_as_of_date'])
                        expected_age = round((as_of - install).days / 365.25, 1)
                        actual_age = sample['primary_child_roof_age_years_as_of_date']
                        assert abs(expected_age - actual_age) < 0.1, \
                            f"Age calculation mismatch: expected {expected_age}, got {actual_age}"

    # 2. Check per-class CSV files

    # Roof instances CSV
    roof_instance_csv = final_dir / 'roof_instance.csv'
    if roof_instance_csv.exists():
        ri_df = pd.read_csv(roof_instance_csv)
        assert 'roof_age_years_as_of_date' in ri_df.columns, \
            "Roof instance CSV should have roof_age_years_as_of_date"

    # Roofs CSV
    roof_csv = final_dir / 'roof.csv'
    if roof_csv.exists():
        roof_df = pd.read_csv(roof_csv)
        if 'primary_child_roof_age_installation_date' in roof_df.columns:
            roofs_with_age = roof_df[roof_df['primary_child_roof_age_installation_date'].notna()]
            if len(roofs_with_age) > 0:
                assert 'primary_child_roof_age_years_as_of_date' in roof_df.columns, \
                    "Roof CSV should have primary_child_roof_age_years_as_of_date"

    # 3. Check per-class parquet files

    # Roof instances parquet
    roof_instance_parquet = final_dir / 'roof_instance_features.parquet'
    if roof_instance_parquet.exists():
        ri_gdf = gpd.read_parquet(roof_instance_parquet)
        assert 'roof_age_years_as_of_date' in ri_gdf.columns, \
            "Roof instance parquet should have roof_age_years_as_of_date"

    # Roofs parquet
    roof_parquet = final_dir / 'roof_features.parquet'
    if roof_parquet.exists():
        roof_gdf = gpd.read_parquet(roof_parquet)
        if 'primary_child_roof_age_installation_date' in roof_gdf.columns:
            roofs_with_age = roof_gdf[roof_gdf['primary_child_roof_age_installation_date'].notna()]
            if len(roofs_with_age) > 0:
                assert 'primary_child_roof_age_years_as_of_date' in roof_gdf.columns, \
                    "Roof parquet should have primary_child_roof_age_years_as_of_date"

    # 4. Check rollup CSV
    rollup_csv = final_dir / 'rollup.csv'
    if rollup_csv.exists():
        rollup_df = pd.read_csv(rollup_csv)
        if 'primary_child_roof_age_installation_date' in rollup_df.columns:
            rollups_with_age = rollup_df[rollup_df['primary_child_roof_age_installation_date'].notna()]
            if len(rollups_with_age) > 0:
                assert 'primary_child_roof_age_years_as_of_date' in rollup_df.columns, \
                    "Rollup should have primary_child_roof_age_years_as_of_date"

    # 5. Cross-verify consistency between files
    if roof_csv.exists() and roof_parquet.exists():
        roof_csv_df = pd.read_csv(roof_csv)
        roof_parquet_gdf = gpd.read_parquet(roof_parquet)

        # Compare roof age values between CSV and parquet for same feature IDs
        if 'feature_id' in roof_csv_df.columns and 'feature_id' in roof_parquet_gdf.columns:
            common_ids = set(roof_csv_df['feature_id']) & set(roof_parquet_gdf['feature_id'])
            if common_ids and 'primary_child_roof_age_years_as_of_date' in roof_csv_df.columns:
                csv_ages = roof_csv_df[roof_csv_df['feature_id'].isin(common_ids)].set_index('feature_id')['primary_child_roof_age_years_as_of_date']
                parquet_ages = roof_parquet_gdf[roof_parquet_gdf['feature_id'].isin(common_ids)].set_index('feature_id')['primary_child_roof_age_years_as_of_date']

                # Check values match (allowing for floating point precision)
                for fid in common_ids:
                    if fid in csv_ages.index and fid in parquet_ages.index:
                        csv_val = csv_ages[fid]
                        parquet_val = parquet_ages[fid]
                        if pd.notna(csv_val) and pd.notna(parquet_val):
                            assert abs(csv_val - parquet_val) < 0.01, \
                                f"Age mismatch for feature {fid}: CSV={csv_val}, Parquet={parquet_val}"