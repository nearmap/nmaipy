"""Test damage classification functionality."""

import pytest
import json
from pathlib import Path
import geopandas as gpd
from shapely.wkt import loads
from nmaipy.feature_api import FeatureApi
from nmaipy.constants import API_CRS

# Test data - 900x900m area from Houston affected by Hurricane Beryl
TEST_AREA_WKT = "POLYGON ((-95.78126757509912 29.55035811820608, -95.78148993878693 29.55847120996382, -95.79076953781752 29.558276436626358, -95.79054643462453 29.55016340881871, -95.78126757509912 29.55035811820608))"

# Output directory for test results - this will be checked into git for comparison
TEST_OUTPUT_DIR = Path(__file__).parent / "data" / "damage_classification_results"


class TestDamageClassification:
    """Test damage classification API functionality."""
    
    @pytest.fixture
    def test_area_geometry(self):
        """Get test area geometry from WKT."""
        return loads(TEST_AREA_WKT)
    
    @pytest.fixture
    def cache_directory(self, tmp_path):
        """Create a temporary cache directory for tests."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir
    
    def test_damage_classification_basic(self, test_area_geometry, cache_directory):
        """Test basic damage classification API call."""
        # Parameters for damage classification
        country = "us"
        packs = ["damage"]
        include = ["damage"]
        since = "2024-07-08"
        until = "2024-07-11"
        aoi_id = "beryl_test_area"
        
        # Create API client
        feature_api = FeatureApi(cache_dir=cache_directory)
        
        # Make API request
        features_gdf, metadata, error = feature_api.get_features_gdf(
            test_area_geometry,
            country,
            packs,
            classes=None,
            include=include,
            aoi_id=aoi_id,
            since=since,
            until=until
        )
        
        # Assertions
        assert error is None, f"API request failed with error: {error}"
        assert features_gdf is not None, "No features returned"
        
        # Debug: print what we got
        print(f"\nTotal features returned: {len(features_gdf)}")
        if len(features_gdf) > 0:
            print(f"Columns: {list(features_gdf.columns)}")
            print(f"Unique descriptions: {features_gdf['description'].unique()}")
            print(f"Unique class_ids: {features_gdf['class_id'].unique()}")
            
            # Check if we have a damage column
            if 'damage' in features_gdf.columns:
                print(f"Damage column present with {features_gdf['damage'].notna().sum()} non-null values")
                # Sample a damage value to see its structure
                sample_damage = features_gdf[features_gdf['damage'].notna()]['damage'].iloc[0] if features_gdf['damage'].notna().any() else None
                if sample_damage:
                    print(f"Sample damage data structure: {type(sample_damage)}")
        
        assert len(features_gdf) > 0, "No features found in test area"
        
        # Check that we got damage-related features - could be in description or damage column
        has_damage_data = False
        if 'damage' in features_gdf.columns:
            # Check if damage column has non-null values
            damage_features = features_gdf[features_gdf['damage'].notna()]
            has_damage_data = len(damage_features) > 0
        
        if not has_damage_data and 'description' in features_gdf.columns:
            # Check description column
            damage_features = features_gdf[features_gdf['description'].str.contains('damage', case=False, na=False)]
            has_damage_data = len(damage_features) > 0
        
        assert has_damage_data, "No damage classification data found in features"
        
        # Check metadata
        assert metadata is not None, "No metadata returned"
        assert 'survey_date' in metadata or 'date' in metadata, "No survey date in metadata"
        
        print(f"\nFound {len(features_gdf)} total features")
        print(f"Found {len(damage_features)} damage-related features")
        if 'description' in features_gdf.columns:
            print(f"Feature types: {features_gdf['description'].value_counts().to_dict()}")
        
        # Save results to GeoPackage for visual inspection and comparison
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = TEST_OUTPUT_DIR / "beryl_damage_classification_test.gpkg"
        
        # Prepare the GeoDataFrame for output
        output_gdf = features_gdf.copy()
        
        # Extract damage classification from the damage column if it exists
        if 'damage' in output_gdf.columns:
            # Extract key damage attributes for easier viewing in QGIS
            def extract_damage_info(damage_dict):
                if damage_dict and isinstance(damage_dict, dict):
                    # Extract the main damage classification
                    confidences = damage_dict.get('confidences', {}).get('raw', {})
                    # Find the highest confidence class
                    if confidences:
                        max_class = max(confidences.items(), key=lambda x: x[1])
                        return {
                            'damage_class': max_class[0],
                            'damage_confidence': max_class[1],
                            'undamaged_conf': confidences.get('Undamaged', 0),
                            'affected_conf': confidences.get('Affected', 0),
                            'minor_conf': confidences.get('Minor', 0),
                            'major_conf': confidences.get('Major', 0),
                            'destroyed_conf': confidences.get('Destroyed', 0)
                        }
                return {
                    'damage_class': None,
                    'damage_confidence': None,
                    'undamaged_conf': None,
                    'affected_conf': None,
                    'minor_conf': None,
                    'major_conf': None,
                    'destroyed_conf': None
                }
            
            damage_info = output_gdf['damage'].apply(extract_damage_info)
            for key in ['damage_class', 'damage_confidence', 'undamaged_conf', 
                       'affected_conf', 'minor_conf', 'major_conf', 'destroyed_conf']:
                output_gdf[key] = damage_info.apply(lambda x: x[key])
        
        # Add metadata to the output
        output_gdf['survey_date'] = metadata.get('date', metadata.get('survey_date'))
        output_gdf['postcat'] = metadata.get('postcat', False)
        
        # Save to GeoPackage
        output_gdf.to_file(output_file, driver='GPKG', layer='damage_features')
        print(f"\nSaved damage classification results to: {output_file}")
        print(f"This file can be opened in QGIS for visual inspection and comparison")
    
    def test_damage_classification_rapid_mode(self, test_area_geometry, cache_directory):
        """Test damage classification with rapid mode enabled."""
        # Parameters for rapid damage classification
        country = "us"
        packs = ["damage"]
        include = ["damage"]
        since = "2024-07-08"
        until = "2024-07-11"
        aoi_id = "beryl_test_rapid"
        
        # Create API client with rapid mode enabled
        feature_api = FeatureApi(cache_dir=cache_directory, rapid=True)
        
        # Make API request
        features_gdf, metadata, error = feature_api.get_features_gdf(
            test_area_geometry,
            country,
            packs,
            classes=None,
            include=include,
            aoi_id=aoi_id,
            since=since,
            until=until
        )
        
        # Assertions
        assert error is None, f"API request with rapid mode failed with error: {error}"
        assert features_gdf is not None, "No features returned in rapid mode"
        
        # In rapid mode, we might get features faster but potentially with different processing
        print(f"\nRapid mode: Found {len(features_gdf) if features_gdf is not None else 0} features")
        
        if features_gdf is not None and len(features_gdf) > 0:
            damage_features = features_gdf[features_gdf['description'].str.contains('damage', case=False, na=False)]
            print(f"Rapid mode: Found {len(damage_features)} damage-related features")
    
    def test_damage_classification_with_order(self, test_area_geometry, cache_directory):
        """Test damage classification with order parameter for date selection."""
        # Test with 'latest' order
        country = "us"
        packs = ["damage"]
        include = ["damage"]
        since = "2024-07-08"
        until = "2024-07-11"
        aoi_id = "beryl_test_latest"
        
        # Create API client with order='latest'
        feature_api_latest = FeatureApi(cache_dir=cache_directory, order="latest")
        
        # Make API request
        features_gdf_latest, metadata_latest, error_latest = feature_api_latest.get_features_gdf(
            test_area_geometry,
            country,
            packs,
            classes=None,
            include=include,
            aoi_id=aoi_id,
            since=since,
            until=until
        )
        
        # Test with 'earliest' order
        aoi_id = "beryl_test_earliest"
        
        # Create API client with order='earliest'
        feature_api_earliest = FeatureApi(cache_dir=cache_directory, order="earliest")
        
        features_gdf_earliest, metadata_earliest, error_earliest = feature_api_earliest.get_features_gdf(
            test_area_geometry,
            country,
            packs,
            classes=None,
            include=include,
            aoi_id=aoi_id,
            since=since,
            until=until
        )
        
        # Assertions
        assert error_latest is None, f"API request with order='latest' failed: {error_latest}"
        assert error_earliest is None, f"API request with order='earliest' failed: {error_earliest}"
        
        # Check that we got results for both
        if features_gdf_latest is not None and metadata_latest:
            print(f"\nOrder='latest': Survey date: {metadata_latest.get('survey_date', metadata_latest.get('date'))}")
            print(f"Order='latest': Found {len(features_gdf_latest)} features")
        
        if features_gdf_earliest is not None and metadata_earliest:
            print(f"Order='earliest': Survey date: {metadata_earliest.get('survey_date', metadata_earliest.get('date'))}")
            print(f"Order='earliest': Found {len(features_gdf_earliest)} features")
    
    def test_damage_classification_exclude_occlusion(self, test_area_geometry, cache_directory):
        """Test damage classification with exclude_tiles_with_occlusion parameter."""
        # Parameters with occlusion exclusion
        country = "us"
        packs = ["damage"]
        include = ["damage"]
        since = "2024-07-08"
        until = "2024-07-11"
        aoi_id = "beryl_test_no_occlusion"
        
        # Create API client with occlusion exclusion
        feature_api = FeatureApi(cache_dir=cache_directory, exclude_tiles_with_occlusion=True)
        
        # Make API request
        features_gdf, metadata, error = feature_api.get_features_gdf(
            test_area_geometry,
            country,
            packs,
            classes=None,
            include=include,
            aoi_id=aoi_id,
            since=since,
            until=until
        )
        
        # Assertions
        assert error is None, f"API request with occlusion exclusion failed: {error}"
        
        print(f"\nWith occlusion exclusion: Found {len(features_gdf) if features_gdf is not None else 0} features")
        
        if features_gdf is not None and len(features_gdf) > 0:
            # Check metadata for occlusion information if available
            if metadata:
                print(f"Metadata: {metadata}")
    
    def test_damage_classification_all_parameters(self, test_area_geometry, cache_directory):
        """Test damage classification with all parameters combined."""
        # All parameters together
        country = "us"
        packs = ["damage"]
        include = ["damage"]
        since = "2024-07-08"
        until = "2024-07-11"
        aoi_id = "beryl_test_all_params"
        
        # Create API client with all parameters
        feature_api = FeatureApi(
            cache_dir=cache_directory,
            rapid=True,
            order="latest",
            exclude_tiles_with_occlusion=True
        )
        
        # Make API request
        features_gdf, metadata, error = feature_api.get_features_gdf(
            test_area_geometry,
            country,
            packs,
            classes=None,
            include=include,
            aoi_id=aoi_id,
            since=since,
            until=until
        )
        
        # Assertions
        assert error is None, f"API request with all parameters failed: {error}"
        
        print(f"\nAll parameters enabled:")
        print(f"  - rapid mode: True")
        print(f"  - order: latest")
        print(f"  - exclude_tiles_with_occlusion: True")
        print(f"  - Found {len(features_gdf) if features_gdf is not None else 0} features")
        
        if features_gdf is not None and len(features_gdf) > 0:
            # Verify we have damage features - check damage column
            if 'damage' in features_gdf.columns:
                damage_features = features_gdf[features_gdf['damage'].notna()]
                assert len(damage_features) > 0, "No damage features found with all parameters"
                print(f"  - Damage features with data: {len(damage_features)}")
            
            # Check for expected damage classifications
            if 'description' in features_gdf.columns:
                descriptions = features_gdf['description'].unique()
                print(f"  - Feature types found: {list(descriptions)}")