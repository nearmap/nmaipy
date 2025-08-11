"""Regression tests for damage classification bug fix."""

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from nmaipy.parcels import (
    flatten_building_lifecycle_damage_attributes,
    feature_attributes
)
from nmaipy.constants import BUILDING_LIFECYCLE_ID, AOI_ID_COLUMN_NAME


class TestDamageBugRegression:
    """Test suite to prevent regression of the damage scalar bug (issue #TRO-3810)."""
    
    def test_flatten_damage_with_scalar_value(self):
        """Test the specific bug: damage processing when damage is scalar."""
        # This was the bug: damage was a scalar value (0 or 0.0) instead of dict
        building_lifecycles = [{
            "attributes": {
                "damage": 0  # Scalar value that caused "invalid index to scalar variable" error
            }
        }]
        
        # Should not raise an error, should return empty dict
        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}
    
    def test_flatten_damage_with_float_scalar(self):
        """Test damage processing with float scalar."""
        building_lifecycles = [{
            "attributes": {
                "damage": 0.0
            }
        }]
        
        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}
    
    def test_flatten_damage_with_none_value(self):
        """Test damage processing when damage is None."""
        building_lifecycles = [{
            "attributes": {
                "damage": None
            }
        }]
        
        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}
    
    def test_flatten_damage_with_empty_dict(self):
        """Test damage processing with empty damage dict."""
        building_lifecycles = [{
            "attributes": {
                "damage": {}
            }
        }]
        
        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}
    
    def test_flatten_damage_with_missing_fema_categories(self):
        """Test damage processing when femaCategoryConfidences is missing."""
        building_lifecycles = [{
            "attributes": {
                "damage": {
                    "someOtherField": "value"
                }
            }
        }]
        
        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}
    
    def test_flatten_damage_with_none_fema_categories(self):
        """Test damage processing when femaCategoryConfidences is None."""
        building_lifecycles = [{
            "attributes": {
                "damage": {
                    "femaCategoryConfidences": None
                }
            }
        }]
        
        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}
    
    def test_flatten_damage_with_empty_fema_categories(self):
        """Test damage processing when femaCategoryConfidences is empty dict."""
        building_lifecycles = [{
            "attributes": {
                "damage": {
                    "femaCategoryConfidences": {}
                }
            }
        }]
        
        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        assert result == {}
    
    def test_flatten_damage_with_valid_data(self):
        """Test normal damage data processing still works."""
        building_lifecycles = [{
            "attributes": {
                "damage": {
                    "femaCategoryConfidences": {
                        "Affected": 0.1,
                        "Minor": 0.2,
                        "Major": 0.6,
                        "Destroyed": 0.1
                    }
                }
            }
        }]
        
        result = flatten_building_lifecycle_damage_attributes(building_lifecycles)
        
        assert result["damage_class"] == "Major"
        assert result["damage_class_confidence"] == 0.6
        assert result["damage_class_Major_confidence"] == 0.6
        assert result["damage_class_Minor_confidence"] == 0.2
        assert result["damage_class_Affected_confidence"] == 0.1
        assert result["damage_class_Destroyed_confidence"] == 0.1
    
    def test_feature_attributes_handles_scalar_damage(self):
        """Test that feature_attributes correctly handles scalar damage in primary features."""
        # Create mock features with scalar damage (the bug scenario)
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
            'attributes': [{
                "damage": 0  # Scalar damage value that caused the bug
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
        
        # This should not raise an error (previously would crash with "invalid index to scalar variable")
        result = feature_attributes(
            features_gdf=features_gdf,
            classes_df=classes_df,
            country='au',
            parcel_geom=parcel_geom,
            primary_decision='largest_intersection',
            primary_lat=None,
            primary_lon=None
        )
        
        # Check that basic attributes are present
        assert 'building_lifecycle_present' in result
        assert result['building_lifecycle_present'] == 'Y'
        assert 'building_lifecycle_count' in result
        assert result['building_lifecycle_count'] == 1
        
        # Damage attributes should not be present for scalar damage data
        assert 'primary_building_lifecycle_damage_class' not in result
        assert 'primary_building_lifecycle_damage_class_confidence' not in result
    
    def test_feature_attributes_with_valid_damage(self):
        """Test that feature_attributes still works correctly with valid damage data."""
        # Create mock features with valid damage data
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
            'attributes': [{
                "damage": {
                    "femaCategoryConfidences": {
                        "Affected": 0.05,
                        "Minor": 0.10,
                        "Major": 0.80,
                        "Destroyed": 0.05
                    }
                }
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
        
        # Check that damage attributes were correctly processed
        assert 'primary_building_lifecycle_damage_class' in result
        assert result['primary_building_lifecycle_damage_class'] == 'Major'
        assert result['primary_building_lifecycle_damage_class_confidence'] == 0.80
        assert result['primary_building_lifecycle_damage_class_Major_confidence'] == 0.80
        assert result['primary_building_lifecycle_damage_class_Minor_confidence'] == 0.10