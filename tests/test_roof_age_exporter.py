"""
Tests for the Roof Age Exporter CLI.
"""
import json
from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from nmaipy.roof_age_exporter import RoofAgeExporter
from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS


@pytest.fixture
def test_aoi_file(tmp_path, parcels_2_gdf):
    """Create a temporary AOI file for testing (uses US parcels)"""
    # Take first 3 parcels from parcels_2_gdf (New Jersey)
    small_gdf = parcels_2_gdf.head(3)
    aoi_file = tmp_path / "test_aois.geojson"
    small_gdf.to_file(aoi_file, driver="GeoJSON")
    return aoi_file


@pytest.fixture
def test_output_dir(tmp_path):
    """Create a temporary output directory"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


def test_exporter_initialization(test_aoi_file, test_output_dir):
    """Test that RoofAgeExporter initializes correctly"""
    exporter = RoofAgeExporter(
        aoi_file=str(test_aoi_file),
        output_dir=str(test_output_dir),
        country="us",
        api_key="test_key",
    )

    assert exporter.aoi_file == str(test_aoi_file)
    assert exporter.output_dir == Path(test_output_dir)
    assert exporter.country == "us"
    assert exporter.threads == 10  # default


def test_exporter_validates_country(test_aoi_file, test_output_dir):
    """Test that exporter rejects non-US countries"""
    with pytest.raises(ValueError, match="only available for US"):
        RoofAgeExporter(
            aoi_file=str(test_aoi_file),
            output_dir=str(test_output_dir),
            country="au",  # Not supported
            api_key="test_key",
        )


def test_exporter_run_mocked(test_aoi_file, test_output_dir, data_directory):
    """Test the full export workflow with mocked API"""
    # Load test response
    fixture_path = data_directory / "test_roof_age_nj_response.json"
    with open(fixture_path, 'r') as f:
        test_response = json.load(f)

    exporter = RoofAgeExporter(
        aoi_file=str(test_aoi_file),
        output_dir=str(test_output_dir),
        output_format="both",  # Test both formats
        country="us",
        api_key="test_key",
        threads=2,
    )

    # Mock the RoofAgeApi.get_roof_age_bulk method
    with patch('nmaipy.roof_age_exporter.RoofAgeApi') as mock_api_class:
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        # Create mock return data
        roofs_gdf = gpd.GeoDataFrame(
            [
                {
                    AOI_ID_COLUMN_NAME: 0,
                    "installationDate": "2001-07-09",
                    "trustScore": 51.5,
                    "area": 107.66,
                    "geometry": Polygon([[-74.275, 40.642], [-74.274, 40.642], [-74.274, 40.641], [-74.275, 40.641], [-74.275, 40.642]]),
                }
            ],
            crs=API_CRS
        )
        metadata_df = pd.DataFrame([{AOI_ID_COLUMN_NAME: 0, "resourceId": "test-resource"}])
        metadata_df = metadata_df.set_index(AOI_ID_COLUMN_NAME)
        errors_df = pd.DataFrame()

        mock_api.get_roof_age_bulk.return_value = (roofs_gdf, metadata_df, errors_df)

        # Run export
        exporter.run()

        # Verify API was called
        assert mock_api.get_roof_age_bulk.called

        # Verify output files were created
        assert (test_output_dir / "test_aois_roofs.parquet").exists()
        assert (test_output_dir / "test_aois_roofs.csv").exists()
        assert (test_output_dir / "test_aois_metadata.csv").exists()


def test_exporter_handles_errors(test_aoi_file, test_output_dir):
    """Test that exporter handles API errors gracefully"""
    exporter = RoofAgeExporter(
        aoi_file=str(test_aoi_file),
        output_dir=str(test_output_dir),
        country="us",
        api_key="test_key",
    )

    # Mock the RoofAgeApi.get_roof_age_bulk to return some errors
    with patch('nmaipy.roof_age_exporter.RoofAgeApi') as mock_api_class:
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        # Return empty results with errors
        roofs_gdf = gpd.GeoDataFrame(columns=[AOI_ID_COLUMN_NAME, "geometry"], crs=API_CRS)
        metadata_df = pd.DataFrame()
        errors_df = pd.DataFrame([
            {AOI_ID_COLUMN_NAME: 0, "status_code": 404, "message": "Not found"},
            {AOI_ID_COLUMN_NAME: 1, "status_code": 404, "message": "Not found"},
        ])
        errors_df = errors_df.set_index(AOI_ID_COLUMN_NAME)

        mock_api.get_roof_age_bulk.return_value = (roofs_gdf, metadata_df, errors_df)

        # Should complete without raising exception
        exporter.run()

        # Verify error file was created
        assert (test_output_dir / "test_aois_errors.csv").exists()


@pytest.mark.integration
def test_exporter_integration(test_aoi_file, test_output_dir):
    """
    Integration test with real API.

    Requires valid API_KEY environment variable.
    """
    exporter = RoofAgeExporter(
        aoi_file=str(test_aoi_file),
        output_dir=str(test_output_dir),
        country="us",
        threads=2,
    )

    exporter.run()

    # Should create at least the metadata file
    assert (test_output_dir / "test_aois_metadata.csv").exists()

    # Check if roofs were found
    roofs_file = test_output_dir / "test_aois_roofs.parquet"
    if roofs_file.exists():
        roofs_gdf = gpd.read_parquet(roofs_file)
        assert len(roofs_gdf) > 0
        assert "installationDate" in roofs_gdf.columns
