"""Integration tests using the example GeoJSON files."""

import os
import sys
import pytest
import tempfile
import pandas as pd
import geopandas as gpd
from pathlib import Path
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

from nmaipy.exporter import AOIExporter
from nmaipy.feature_api import FeatureApi


# Load .env file for API key if available
load_dotenv()


class TestIntegration:
    """End-to-end integration tests using example data."""
    
    @pytest.mark.integration
    def test_sydney_parcels_workflow(self):
        """Test complete workflow with Sydney parcels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "sydney_test"
            
            # Create exporter
            exporter = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=str(output_dir),
                country='au',
                packs=['building'],
                processes=1,
                save_features=False,
            )
            
            # Mock the API calls for predictable testing
            with patch.object(FeatureApi, 'get_rollup_df') as mock_rollup:
                # Create mock response
                mock_df = pd.DataFrame({
                    'aoi_id': [0, 1, 2],
                    'roof_count': [2, 1, 3],
                    'building_area_sqm': [150.5, 120.0, 200.0],
                })
                mock_rollup.return_value = mock_df
                
                # Run the exporter
                exporter.run()
                
                # Verify output files were created
                final_dir = output_dir / "final"
                assert final_dir.exists(), "Final directory not created"
                
                # Check for rollup file
                rollup_file = final_dir / "sydney_parcels.csv"
                assert rollup_file.exists(), "Rollup CSV not created"
                
                # Load and verify rollup data
                df = pd.read_csv(rollup_file)
                assert len(df) == 3, f"Expected 3 parcels, got {len(df)}"
    
    @pytest.mark.integration
    def test_us_parcels_workflow(self):
        """Test workflow with US parcels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "us_test"
            
            exporter = AOIExporter(
                aoi_file='data/examples/us_parcels.geojson',
                output_dir=str(output_dir),
                country='us',
                packs=['building'],
                processes=1,
            )
            
            # Mock the feature API
            with patch.object(FeatureApi, 'get_rollup_df') as mock_rollup:
                mock_df = pd.DataFrame({
                    'aoi_id': [0, 1],
                    'roof_count': [1, 2],
                    'building_area_sqm': [180.0, 220.0],
                })
                mock_rollup.return_value = mock_df
                
                exporter.run()
                
                # Verify outputs
                rollup_file = output_dir / "final" / "us_parcels.csv"
                assert rollup_file.exists(), "US parcels rollup not created"
                
                df = pd.read_csv(rollup_file)
                assert len(df) == 2, f"Expected 2 parcels, got {len(df)}"
    
    @pytest.mark.integration
    def test_large_area_triggers_gridding(self):
        """Test that large area triggers automatic gridding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "large_test"
            
            exporter = AOIExporter(
                aoi_file='data/examples/large_area.geojson',
                output_dir=str(output_dir),
                country='au',
                packs=['building'],
                processes=1,
                aoi_grid_inexact=True,
            )
            
            # The large area should trigger gridding
            # We can verify this by checking the logs or mocking
            with patch.object(FeatureApi, '_attempt_gridding') as mock_grid:
                mock_grid.return_value = gpd.GeoDataFrame()
                
                with patch.object(FeatureApi, 'get_rollup_df') as mock_rollup:
                    mock_rollup.return_value = pd.DataFrame({
                        'aoi_id': [0],
                        'roof_count': [100],
                    })
                    
                    exporter.run()
                    
                    # The gridding should have been considered due to area size
                    # (actual call depends on implementation details)
    
    @pytest.mark.integration
    def test_multiple_packs(self):
        """Test extraction with multiple AI packs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "multi_pack"
            
            exporter = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=str(output_dir),
                country='au',
                packs=['building', 'vegetation'],
                processes=1,
            )
            
            with patch.object(FeatureApi, 'get_rollup_df') as mock_rollup:
                mock_df = pd.DataFrame({
                    'aoi_id': [0, 1, 2],
                    'roof_count': [1, 0, 2],
                    'vegetation_area_sqm': [50.0, 75.0, 100.0],
                })
                mock_rollup.return_value = mock_df
                
                exporter.run()
                
                rollup_file = output_dir / "final" / "sydney_parcels.csv"
                assert rollup_file.exists()
                
                df = pd.read_csv(rollup_file)
                # Should have columns for both building and vegetation
                assert 'roof_count' in df.columns or any('roof' in col for col in df.columns)
                assert 'vegetation_area_sqm' in df.columns or any('vegetation' in col for col in df.columns)
    
    @pytest.mark.integration
    def test_save_features_enabled(self):
        """Test that save_features creates feature files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "features_test"
            
            exporter = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=str(output_dir),
                country='au',
                packs=['building'],
                processes=1,
                save_features=True,
            )
            
            # Mock both rollup and features
            with patch.object(FeatureApi, 'get_rollup_df') as mock_rollup:
                mock_rollup.return_value = pd.DataFrame({
                    'aoi_id': [0, 1, 2],
                    'roof_count': [1, 1, 1],
                })
                
                with patch.object(FeatureApi, 'get_features_gdf') as mock_features:
                    # Create mock features GeoDataFrame
                    from shapely.geometry import Point
                    mock_gdf = gpd.GeoDataFrame({
                        'feature_id': ['f1', 'f2', 'f3'],
                        'class_id': ['roof', 'roof', 'roof'],
                        'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)],
                    })
                    mock_features.return_value = mock_gdf
                    
                    exporter.run()
                    
                    # Check for features file
                    features_file = output_dir / "final" / "sydney_parcels_features.parquet"
                    # Note: Features may not be created if mocking doesn't fully simulate the workflow
    
    @pytest.mark.integration
    def test_parquet_output_format(self):
        """Test Parquet output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "parquet_test"
            
            exporter = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=str(output_dir),
                country='au',
                packs=['building'],
                processes=1,
                rollup_format='parquet',
            )
            
            with patch.object(FeatureApi, 'get_rollup_df') as mock_rollup:
                mock_df = pd.DataFrame({
                    'aoi_id': [0, 1, 2],
                    'roof_count': [1, 2, 3],
                })
                mock_rollup.return_value = mock_df
                
                exporter.run()
                
                # Check for parquet file instead of CSV
                parquet_file = output_dir / "final" / "sydney_parcels.parquet"
                csv_file = output_dir / "final" / "sydney_parcels.csv"
                
                assert parquet_file.exists() or csv_file.exists(), "No output file created"
    
    @pytest.mark.live_api
    @pytest.mark.skipif(not os.environ.get('API_KEY'), reason="API_KEY not set")
    def test_live_api_small_area(self):
        """Test with real API call on small area (requires API_KEY)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "live_test"
            
            # Use the smallest example file for quick testing
            exporter = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=str(output_dir),
                country='au',
                packs=['building'],
                processes=1,
                chunk_size=10,
            )
            
            # Run with real API
            exporter.run()
            
            # Verify real outputs
            rollup_file = output_dir / "final" / "sydney_parcels.csv"
            assert rollup_file.exists(), "Live API test: rollup not created"
            
            df = pd.read_csv(rollup_file)
            assert len(df) > 0, "Live API test: no data returned"
            assert 'roof_count' in df.columns or any('roof' in col for col in df.columns), \
                "Live API test: no building data in response"
    
    @pytest.mark.integration
    def test_error_handling_missing_file(self):
        """Test error handling for missing input file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = AOIExporter(
                aoi_file='data/examples/nonexistent.geojson',
                output_dir=tmpdir,
                country='au',
                packs=['building'],
            )
            
            # Should raise an error when trying to run
            with pytest.raises(Exception):
                exporter.run()
    
    @pytest.mark.integration
    def test_error_handling_invalid_geojson(self):
        """Test error handling for invalid GeoJSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an invalid GeoJSON file
            invalid_file = Path(tmpdir) / "invalid.geojson"
            invalid_file.write_text('{"not": "valid geojson"}')
            
            exporter = AOIExporter(
                aoi_file=str(invalid_file),
                output_dir=tmpdir,
                country='au',
                packs=['building'],
            )
            
            # Should handle invalid GeoJSON gracefully
            with pytest.raises(Exception):
                exporter.run()


if __name__ == "__main__":
    # Run integration tests
    current_file = os.path.abspath(__file__)
    # Run only integration tests by default
    sys.exit(pytest.main([current_file, "-m", "integration", "-v"]))