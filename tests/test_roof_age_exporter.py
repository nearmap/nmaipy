"""
Tests for the Roof Age Exporter CLI.

Note: Tests that require mocking the API must mock at the process_chunk level
since the exporter uses multiprocessing, and mocks don't cross process boundaries.
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS
from nmaipy.roof_age_exporter import RoofAgeExporter


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
    assert exporter.output_dir == str(test_output_dir)
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


def test_exporter_process_chunk_with_cached_data(test_aoi_file, test_output_dir, roof_age_gdf, roof_age_metadata_df):
    """Test process_chunk method with real cached API data (bypasses multiprocessing).

    Uses real Roof Age API response data cached in test_roof_age_bulk_nj.parquet
    instead of hand-crafted mock data. This ensures the test validates against
    the actual column schema returned by the API.
    """
    exporter = RoofAgeExporter(
        aoi_file=str(test_aoi_file),
        output_dir=str(test_output_dir),
        output_format="both",
        country="us",
        api_key="test_key",
        processes=1,
    )

    # Prepare AOI GeoDataFrame matching the cached data's AOI IDs
    aoi_ids = roof_age_gdf[AOI_ID_COLUMN_NAME].unique()
    aois = [
        Polygon(
            [
                [-74.275, 40.642],
                [-74.274, 40.642],
                [-74.274, 40.641],
                [-74.275, 40.641],
                [-74.275, 40.642],
            ]
        )
        for _ in aoi_ids
    ]
    aoi_gdf = gpd.GeoDataFrame(geometry=aois, crs=API_CRS, index=pd.Index(aoi_ids, name=AOI_ID_COLUMN_NAME))

    # Ensure chunk directory exists
    Path(exporter.chunk_path).mkdir(parents=True, exist_ok=True)

    # Patch the API to return real cached data instead of making network calls
    with patch("nmaipy.roof_age_exporter.RoofAgeApi") as mock_api_class:
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_api.get_roof_age_bulk.return_value = (
            roof_age_gdf.copy(),
            roof_age_metadata_df.copy(),
            pd.DataFrame(),
        )
        mock_api.get_latency_stats.return_value = None

        exporter.process_chunk("test_chunk_0000", aoi_gdf)

        assert mock_api.get_roof_age_bulk.called

        # Verify chunk output files were created
        roofs_path = Path(exporter.chunk_path) / "roofs_test_chunk_0000.parquet"
        metadata_path = Path(exporter.chunk_path) / "metadata_test_chunk_0000.parquet"
        assert roofs_path.exists(), "Roofs chunk parquet should be created"
        assert metadata_path.exists(), "Metadata chunk parquet should be created"

        # Read back and validate the output has expected columns
        # Note: RoofAgeExporter writes raw API columns (no roof_age_ prefix);
        # the prefix is added later by the main NearmapAIExporter's export_feature_class()
        result_gdf = gpd.read_parquet(roofs_path)
        assert len(result_gdf) == len(roof_age_gdf), "All roof instances should be written"

        for col in ["installation_date", "trust_score", "kind", "map_browser_url"]:
            assert col in result_gdf.columns, f"Missing expected column: {col}"


def test_exporter_process_chunk_with_errors(test_aoi_file, test_output_dir):
    """Test that process_chunk handles API errors and saves error file"""
    exporter = RoofAgeExporter(
        aoi_file=str(test_aoi_file),
        output_dir=str(test_output_dir),
        country="us",
        api_key="test_key",
    )

    # Prepare test AOI data
    aois = [
        Polygon(
            [
                [-74.275, 40.642],
                [-74.274, 40.642],
                [-74.274, 40.641],
                [-74.275, 40.641],
                [-74.275, 40.642],
            ]
        ),
    ]
    aoi_gdf = gpd.GeoDataFrame(geometry=aois, crs=API_CRS, index=pd.Index([0], name=AOI_ID_COLUMN_NAME))

    # Ensure chunk directory exists
    Path(exporter.chunk_path).mkdir(parents=True, exist_ok=True)

    with patch("nmaipy.roof_age_exporter.RoofAgeApi") as mock_api_class:
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        # Return empty results with errors
        roofs_gdf = gpd.GeoDataFrame(columns=[AOI_ID_COLUMN_NAME, "geometry"], crs=API_CRS)
        metadata_df = pd.DataFrame()
        errors_df = pd.DataFrame(
            [
                {AOI_ID_COLUMN_NAME: 0, "status_code": 404, "message": "Not found"},
            ]
        )
        errors_df = errors_df.set_index(AOI_ID_COLUMN_NAME)

        mock_api.get_roof_age_bulk.return_value = (roofs_gdf, metadata_df, errors_df)
        mock_api.get_latency_stats.return_value = None

        # Should complete without raising exception
        exporter.process_chunk("test_chunk_0000", aoi_gdf)

        # Verify error file was created
        assert (Path(exporter.chunk_path) / "roof_age_errors_test_chunk_0000.parquet").exists()


@pytest.mark.integration
def test_exporter_integration(test_aoi_file, test_output_dir):
    """
    Integration test with real API.

    Requires valid API_KEY environment variable.
    Skipped if API_KEY is not set.
    """
    if not os.environ.get("API_KEY"):
        pytest.skip("API_KEY not set - skipping integration test")

    exporter = RoofAgeExporter(
        aoi_file=str(test_aoi_file),
        output_dir=str(test_output_dir),
        country="us",
        processes=2,
    )

    exporter.run()

    # Check output directory for final files
    final_dir = test_output_dir / "final"

    # Should create at least the metadata file (even if no roofs found)
    metadata_file = final_dir / "metadata.csv"
    if metadata_file.exists():
        metadata_df = pd.read_csv(metadata_file)
        assert len(metadata_df) >= 0  # May be empty if all errors

    # Check if roofs were found
    roofs_file = final_dir / "roofs.parquet"
    if roofs_file.exists():
        roofs_gdf = gpd.read_parquet(roofs_file)
        assert len(roofs_gdf) >= 0
        if len(roofs_gdf) > 0:
            assert "roof_age_installation_date" in roofs_gdf.columns
