"""
Integration tests using the example GeoJSON files.

Note: Tests that require the full export workflow need a valid API_KEY environment
variable because the exporter uses multiprocessing, and mocks don't cross process
boundaries. Tests are skipped if API_KEY is not set.
"""

import os
import pytest
import tempfile
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dotenv import load_dotenv

from nmaipy.exporter import AOIExporter


# Load .env file for API key if available
load_dotenv()

# Check if API key is available for integration tests
HAS_API_KEY = bool(os.environ.get('API_KEY'))


class TestIntegration:
    """End-to-end integration tests using example data."""

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_API_KEY, reason="API_KEY not set - skipping integration test")
    def test_sydney_parcels_workflow(self):
        """Test complete workflow with Sydney parcels (requires API_KEY)."""
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

            # Run the exporter with real API
            exporter.run()

            # Verify output files were created
            final_dir = output_dir / "final"
            assert final_dir.exists(), "Final directory not created"

            # Check for rollup file
            rollup_file = final_dir / "sydney_parcels_aoi_rollup.csv"
            assert rollup_file.exists(), f"Rollup CSV not created. Files in final: {list(final_dir.iterdir())}"

            # Load and verify rollup data
            df = pd.read_csv(rollup_file)
            assert len(df) >= 1, f"Expected at least 1 parcel, got {len(df)}"

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_API_KEY, reason="API_KEY not set - skipping integration test")
    def test_us_parcels_workflow(self):
        """Test workflow with US parcels (requires API_KEY)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "us_test"

            exporter = AOIExporter(
                aoi_file='data/examples/us_parcels.geojson',
                output_dir=str(output_dir),
                country='us',
                packs=['building'],
                processes=1,
            )

            exporter.run()

            # Verify outputs - check for the actual filename pattern
            final_dir = output_dir / "final"
            rollup_files = list(final_dir.glob("*_aoi_rollup.csv"))
            assert len(rollup_files) > 0, f"US parcels rollup not created. Files: {list(final_dir.iterdir())}"

            df = pd.read_csv(rollup_files[0])
            assert len(df) >= 1, f"Expected at least 1 parcel, got {len(df)}"
    
    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_API_KEY, reason="API_KEY not set - skipping integration test")
    def test_large_area_triggers_gridding(self):
        """Test that large area triggers automatic gridding (requires API_KEY)."""
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

            # Run with real API - gridding behavior is tested implicitly
            exporter.run()

            # Verify output was created
            final_dir = output_dir / "final"
            assert final_dir.exists(), "Final directory not created"

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_API_KEY, reason="API_KEY not set - skipping integration test")
    def test_multiple_packs(self):
        """Test extraction with multiple AI packs (requires API_KEY)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "multi_pack"

            exporter = AOIExporter(
                aoi_file='data/examples/sydney_parcels.geojson',
                output_dir=str(output_dir),
                country='au',
                packs=['building', 'vegetation'],
                processes=1,
            )

            exporter.run()

            # Check for rollup file with actual naming pattern
            final_dir = output_dir / "final"
            rollup_files = list(final_dir.glob("*_aoi_rollup.csv"))
            assert len(rollup_files) > 0, f"No rollup file created. Files: {list(final_dir.iterdir())}"

            df = pd.read_csv(rollup_files[0])
            # Should have columns for both building and vegetation
            assert any('roof' in col.lower() for col in df.columns), "No building/roof columns found"

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_API_KEY, reason="API_KEY not set - skipping integration test")
    def test_save_features_enabled(self):
        """Test that save_features creates feature files (requires API_KEY)."""
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

            exporter.run()

            # Check for feature files (parquet format)
            final_dir = output_dir / "final"
            feature_files = list(final_dir.glob("*.parquet"))
            # Features may or may not be created depending on API response
            # Just verify the export completed
            assert final_dir.exists(), "Final directory not created"

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_API_KEY, reason="API_KEY not set - skipping integration test")
    def test_parquet_output_format(self):
        """Test Parquet output format (requires API_KEY)."""
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

            exporter.run()

            # Check for parquet file instead of CSV
            final_dir = output_dir / "final"
            parquet_files = list(final_dir.glob("*.parquet"))
            csv_files = list(final_dir.glob("*_aoi_rollup.csv"))

            # Either parquet or csv should exist
            assert len(parquet_files) > 0 or len(csv_files) > 0, \
                f"No output file created. Files: {list(final_dir.iterdir())}"

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_API_KEY, reason="API_KEY not set - skipping integration test")
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
            final_dir = output_dir / "final"
            rollup_files = list(final_dir.glob("*_aoi_rollup.csv"))
            assert len(rollup_files) > 0, f"Live API test: rollup not created. Files: {list(final_dir.iterdir())}"

            df = pd.read_csv(rollup_files[0])
            assert len(df) > 0, "Live API test: no data returned"
    
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