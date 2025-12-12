"""Test damage classification functionality."""

import os
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from nmaipy.parcels import (
    flatten_building_lifecycle_damage_attributes,
    feature_attributes
)
from nmaipy.constants import BUILDING_LIFECYCLE_ID, AOI_ID_COLUMN_NAME


class TestDamageClassification:
    """Test damage classification processing with the new API structure."""

    def test_flatten_damage_with_valid_data(self):
        """Test normal damage data processing (damage.confidences.raw)."""
        building_lifecycles = [{
            "damage": {
                "confidences": {
                    "raw": {
                        "Undamaged": 0.807,
                        "Affected": 0.176,
                        "Minor": 0.015,
                        "Major": 0.002,
                        "Destroyed": 0
                    },
                    "3tier": {
                        "UndamagedOrAffected": 0.983,
                        "Minor": 0.015,
                        "MajorOrDestroyed": 0.002
                    },
                    "2tier": {
                        "UndamagedOrAffectedOrMinor": 0.998,
                        "MajorOrDestroyed": 0.002
                    }
                },
                "ratios": []
            }
        }]

        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)

        assert result["damage_class"] == "Undamaged"
        assert result["damage_class_confidence"] == 0.807
        assert result["damage_class_Undamaged_confidence"] == 0.807
        assert result["damage_class_Affected_confidence"] == 0.176
        assert result["damage_class_Minor_confidence"] == 0.015

    def test_flatten_damage_with_ratios_and_2tier(self):
        """Test damage data processing including ratios and 2tier confidences."""
        building_lifecycles = [{
            "damage": {
                "confidences": {
                    "raw": {
                        "Undamaged": 0.967,
                        "Affected": 0.028,
                        "Minor": 0.004,
                        "Major": 0.001,
                        "Destroyed": 0
                    },
                    "3tier": {
                        "UndamagedOrAffected": 0.995,
                        "Minor": 0.004,
                        "MajorOrDestroyed": 0.001
                    },
                    "2tier": {
                        "UndamagedOrAffectedOrMinor": 0.999,
                        "MajorOrDestroyed": 0.001
                    }
                },
                "ratios": [
                    {
                        "classId": "2322ca41-5d3d-5782-b2b7-1a2ffd0c4b78",
                        "description": "Exposed Underlayment",
                        "ratioAbove50PctConf": 0
                    },
                    {
                        "classId": "dec855e2-ae6f-56b5-9cbb-f9967ff8ca12",
                        "description": "Missing Roof Tile or Shingle",
                        "ratioAbove50PctConf": 0.15
                    },
                    {
                        "classId": "f907e625-26b3-59db-a806-d41f62ce1f1b",
                        "description": "Structural Damage",
                        "ratioAbove50PctConf": 0
                    }
                ]
            }
        }]

        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)

        # Check raw confidences
        assert result["damage_class"] == "Undamaged"
        assert result["damage_class_confidence"] == 0.967
        assert result["damage_class_Undamaged_confidence"] == 0.967
        assert result["damage_class_Affected_confidence"] == 0.028
        assert result["damage_class_Minor_confidence"] == 0.004
        assert result["damage_class_Major_confidence"] == 0.001
        assert result["damage_class_Destroyed_confidence"] == 0

        # Check 2tier confidences
        assert result["damage_2tier_UndamagedOrAffectedOrMinor_confidence"] == 0.999
        assert result["damage_2tier_MajorOrDestroyed_confidence"] == 0.001

        # Check ratios
        assert result["damage_ratio_exposed_underlayment"] == 0
        assert result["damage_ratio_missing_roof_tile_or_shingle"] == 0.15
        assert result["damage_ratio_structural_damage"] == 0
    
    def test_flatten_damage_with_none_value(self):
        """Test damage processing when damage is None."""
        building_lifecycles = [{
            "damage": None
        }]

        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}

    def test_flatten_damage_with_scalar_value(self):
        """Test damage processing when damage is scalar."""
        building_lifecycles = [{
            "damage": 0  # Scalar value instead of dict
        }]

        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}

    def test_flatten_damage_with_missing_confidences(self):
        """Test damage processing when confidences is missing."""
        building_lifecycles = [{
            "damage": {
                "someOtherField": "value"
            }
        }]

        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}

    def test_flatten_damage_with_empty_confidences(self):
        """Test damage processing when confidences is empty."""
        building_lifecycles = [{
            "damage": {
                "confidences": {}
            }
        }]

        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}

    def test_flatten_damage_missing_field(self):
        """Test damage processing when damage key is missing."""
        building_lifecycles = [{}]

        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}
    
    def test_flatten_damage_empty_list(self):
        """Test damage processing with empty list."""
        building_lifecycles = []
        
        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}
    
    def test_feature_attributes_with_damage_primary(self):
        """Test that feature_attributes correctly processes damage for primary building lifecycle."""
        # Create mock features with BUILDING_LIFECYCLE_ID
        features_data = pd.DataFrame({
            'class_id': [BUILDING_LIFECYCLE_ID],
            'description': ['building_lifecycle'],
            'area_sqm': [100.0],
            'area_sqft': [1076.4],
            'clipped_area_sqm': [90.0],
            'clipped_area_sqft': [968.7],
            'unclipped_area_sqm': [100.0],
            'unclipped_area_sqft': [1076.4],
            'confidence': [0.95],
            'fidelity': [0.9],
            'feature_id': ['feat_1'],
            'attributes': [[]],
            'damage': [{
                "confidences": {
                    "raw": {
                        "Undamaged": 0.05,
                        "Affected": 0.05,
                        "Minor": 0.10,
                        "Major": 0.80,
                        "Destroyed": 0.00
                    },
                    "2tier": {
                        "UndamagedOrAffectedOrMinor": 0.20,
                        "MajorOrDestroyed": 0.80
                    }
                },
                "ratios": []
            }]
        })
        features_data.index = ['parcel_1']
        features_data.index.name = AOI_ID_COLUMN_NAME

        features_gdf = gpd.GeoDataFrame(features_data)

        # Mock classes dataframe
        classes_df = pd.DataFrame({
            'description': ['building_lifecycle']
        })
        classes_df.index = [BUILDING_LIFECYCLE_ID]

        # Create polygon for parcel
        parcel_geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        # Process features
        result = feature_attributes(
            features_gdf=features_gdf,
            classes_df=classes_df,
            country='au',
            parcel_geom=parcel_geom,
            primary_decision='largest_intersection',
            primary_lat=None,
            primary_lon=None
        )

        # Check that damage attributes were processed for primary feature
        assert 'primary_building_lifecycle_damage_class' in result
        assert result['primary_building_lifecycle_damage_class'] == 'Major'
        assert result['primary_building_lifecycle_damage_class_confidence'] == 0.80
        assert result['primary_building_lifecycle_damage_class_Major_confidence'] == 0.80
        assert result['primary_building_lifecycle_damage_2tier_MajorOrDestroyed_confidence'] == 0.80
    
    def test_feature_attributes_with_invalid_damage_primary(self):
        """Test that feature_attributes handles invalid damage data gracefully."""
        # Create mock features with invalid damage data
        features_data = pd.DataFrame({
            'class_id': [BUILDING_LIFECYCLE_ID],
            'description': ['building_lifecycle'],
            'area_sqm': [100.0],
            'area_sqft': [1076.4],
            'clipped_area_sqm': [90.0],
            'clipped_area_sqft': [968.7],
            'unclipped_area_sqm': [100.0],
            'unclipped_area_sqft': [1076.4],
            'confidence': [0.95],
            'fidelity': [0.9],
            'feature_id': ['feat_1'],
            'attributes': [[]],
            'damage': [0]  # Invalid scalar damage value
        })
        features_data.index = ['parcel_1']
        features_data.index.name = AOI_ID_COLUMN_NAME
        
        features_gdf = gpd.GeoDataFrame(features_data)
        
        # Mock classes dataframe
        classes_df = pd.DataFrame({
            'description': ['building_lifecycle']
        })
        classes_df.index = [BUILDING_LIFECYCLE_ID]
        
        # Create polygon for parcel
        parcel_geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        
        # Process features - should not raise an error
        result = feature_attributes(
            features_gdf=features_gdf,
            classes_df=classes_df,
            country='au',
            parcel_geom=parcel_geom,
            primary_decision='largest_intersection',
            primary_lat=None,
            primary_lon=None
        )
        
        # Check that basic attributes are present but damage attributes are not
        assert 'building_lifecycle_present' in result
        assert result['building_lifecycle_present'] == 'Y'
        assert 'building_lifecycle_count' in result
        assert result['building_lifecycle_count'] == 1
        # Damage attributes should not be present for invalid damage data
        assert 'primary_building_lifecycle_damage_class' not in result
    
    @pytest.mark.live_api
    @pytest.mark.skipif(not os.environ.get('API_KEY'), reason="API_KEY not set")
    def test_damage_classification_api_basic(self, cache_directory):
        """Test basic damage classification API call with Hurricane Beryl data."""
        from shapely.wkt import loads
        from nmaipy.feature_api import FeatureApi
        from pathlib import Path
        import os
        
        # Test data - 900x900m area from Houston affected by Hurricane Beryl
        TEST_AREA_WKT = "POLYGON ((-95.78126757509912 29.55035811820608, -95.78148993878693 29.55847120996382, -95.79076953781752 29.558276436626358, -95.79054643462453 29.55016340881871, -95.78126757509912 29.55035811820608))"
        test_area_geometry = loads(TEST_AREA_WKT)
        
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
        features_gdf, metadata, error, _ = feature_api.get_features_gdf(
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
            
            # Check if we have damage-related data
            has_damage_data = False
            if 'damage' in features_gdf.columns:
                damage_features = features_gdf[features_gdf['damage'].notna()]
                has_damage_data = len(damage_features) > 0
                print(f"Damage column present with {len(damage_features)} non-null values")
            
            if 'attributes' in features_gdf.columns:
                # Check for damage in attributes
                attrs_with_damage = features_gdf[features_gdf['attributes'].notna()]
                if len(attrs_with_damage) > 0:
                    # Sample to see structure
                    sample_attr = attrs_with_damage['attributes'].iloc[0]
                    if isinstance(sample_attr, dict) and 'damage' in sample_attr:
                        has_damage_data = True
                        print(f"Found damage data in attributes column")
            
            assert has_damage_data, "No damage classification data found in features"
        
        assert len(features_gdf) > 0, "No features found in test area"
        
        # Check metadata
        assert metadata is not None, "No metadata returned"
        assert 'survey_date' in metadata or 'date' in metadata, "No survey date in metadata"
        
        print(f"\nFound {len(features_gdf)} total features")
        if 'description' in features_gdf.columns:
            print(f"Feature types: {features_gdf['description'].value_counts().to_dict()}")
        
        # Optional: Save results for inspection
        TEST_OUTPUT_DIR = Path(__file__).parent / "data" / "damage_classification_results"
        if os.environ.get('SAVE_TEST_OUTPUT'):
            TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = TEST_OUTPUT_DIR / "beryl_damage_test.gpkg"
            features_gdf.to_file(output_file, driver='GPKG', layer='damage_features')
            print(f"\nSaved damage classification results to: {output_file}")