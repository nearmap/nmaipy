"""Tests for robust CSV and Parquet file reading with mixed types and explicit dtypes."""
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from nmaipy import parcels
from nmaipy.constants import AOI_ID_COLUMN_NAME


class TestRobustFileReading:
    """Test robust reading of CSV and Parquet files with various edge cases."""

    def test_csv_mixed_types_in_column(self):
        """Test reading CSV with mixed types in a column (mostly numeric with some text)."""
        # Create a temporary CSV file with mixed types in street_address column
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(f"{AOI_ID_COLUMN_NAME},street_address,city,geometry\n")
            f.write('1,"123","Melbourne","POLYGON((144.9 -37.8, 145.0 -37.8, 145.0 -37.9, 144.9 -37.9, 144.9 -37.8))"\n')
            f.write('2,"456","Melbourne","POLYGON((144.9 -37.7, 145.0 -37.7, 145.0 -37.8, 144.9 -37.8, 144.9 -37.7))"\n')
            f.write('3,"789","Melbourne","POLYGON((144.9 -37.6, 145.0 -37.6, 145.0 -37.7, 144.9 -37.7, 144.9 -37.6))"\n')
            f.write('4,"12A Main St","Melbourne","POLYGON((144.9 -37.5, 145.0 -37.5, 145.0 -37.6, 144.9 -37.6, 144.9 -37.5))"\n')
            f.write('5,"","Melbourne","POLYGON((144.9 -37.4, 145.0 -37.4, 145.0 -37.5, 144.9 -37.5, 144.9 -37.4))"\n')
            csv_path = f.name

        try:
            # Read the file
            result = parcels.read_from_file(Path(csv_path))

            # Verify all rows were read
            assert len(result) == 5

            # Verify street_address column is string type (not partially numeric)
            assert result.loc[1, "street_address"] == "123"
            assert result.loc[4, "street_address"] == "12A Main St"

            # Verify missing value is handled correctly
            assert pd.isna(result.loc[5, "street_address"]) or result.loc[5, "street_address"] == ""

            # Verify geometries were parsed correctly
            assert all(result.geometry.is_valid)
            assert len(result[result.geometry.is_empty]) == 0

        finally:
            Path(csv_path).unlink()

    def test_csv_all_separators(self):
        """Test reading CSV, PSV, and TSV files."""
        test_cases = [
            (".csv", ","),
            (".psv", "|"),
            (".tsv", "\t"),
        ]

        for suffix, sep in test_cases:
            with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
                f.write(f"{AOI_ID_COLUMN_NAME}{sep}name{sep}geometry\n")
                f.write(f'1{sep}"Test 1"{sep}"POLYGON((144.9 -37.8, 145.0 -37.8, 145.0 -37.9, 144.9 -37.9, 144.9 -37.8))"\n')
                f.write(f'2{sep}"Test 2"{sep}"POLYGON((144.9 -37.7, 145.0 -37.7, 145.0 -37.8, 144.9 -37.8, 144.9 -37.7))"\n')
                file_path = f.name

            try:
                result = parcels.read_from_file(Path(file_path))
                assert len(result) == 2
                assert result.index.name == AOI_ID_COLUMN_NAME
                assert list(result.index) == [1, 2]
            finally:
                Path(file_path).unlink()

    def test_parquet_without_geometry(self):
        """Test reading parquet file without geometry column (non-geoparquet)."""
        # Create a non-geo parquet file with explicit dtypes
        df = pd.DataFrame({
            AOI_ID_COLUMN_NAME: [1, 2, 3, 4, 5],
            "street_address": pd.array(["123", "456", "789", "12A Main St", None], dtype="string"),
            "postal_code": pd.array([3000, 3001, 3002, 3003, None], dtype="Int64"),
            "property_value": pd.array([500000.5, 600000.0, 700000.25, None, 550000.0], dtype="Float64"),
        })

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            # Write parquet with explicit dtypes
            df.to_parquet(parquet_path, index=False)

            # Read it back
            result = parcels.read_from_file(Path(parquet_path))

            # Verify data integrity
            assert len(result) == 5

            # Verify dtypes are preserved (should be string or object, not numeric)
            assert result.loc[1, "street_address"] == "123"
            assert result.loc[4, "street_address"] == "12A Main St"

            # Verify nullable integer type is preserved
            assert pd.isna(result.loc[5, "postal_code"])
            assert result.loc[1, "postal_code"] == 3000

            # Verify nullable float is preserved
            assert pd.isna(result.loc[4, "property_value"])
            assert result.loc[1, "property_value"] == 500000.5

        finally:
            Path(parquet_path).unlink()

    def test_parquet_with_geometry(self):
        """Test reading geoparquet file with geometry column."""
        # Create a geoparquet file
        gdf = gpd.GeoDataFrame({
            AOI_ID_COLUMN_NAME: [1, 2, 3],
            "name": ["Parcel 1", "Parcel 2", "Parcel 3"],
            "geometry": [
                Polygon([(144.9, -37.8), (145.0, -37.8), (145.0, -37.9), (144.9, -37.9)]),
                Polygon([(144.9, -37.7), (145.0, -37.7), (145.0, -37.8), (144.9, -37.8)]),
                Polygon([(144.9, -37.6), (145.0, -37.6), (145.0, -37.7), (144.9, -37.7)]),
            ],
        }, crs="EPSG:4326")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            # Write geoparquet
            gdf.to_parquet(parquet_path)

            # Read it back
            result = parcels.read_from_file(Path(parquet_path))

            # Verify it's a GeoDataFrame
            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 3
            assert all(result.geometry.is_valid)
            assert result.crs is not None

        finally:
            Path(parquet_path).unlink()

    def test_csv_empty_and_null_handling(self):
        """Test CSV reading with various empty and null representations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(f"{AOI_ID_COLUMN_NAME},value1,value2,geometry\n")
            f.write('1,"","NA","POLYGON((144.9 -37.8, 145.0 -37.8, 145.0 -37.9, 144.9 -37.9, 144.9 -37.8))"\n')
            f.write('2,"test",,"POLYGON((144.9 -37.7, 145.0 -37.7, 145.0 -37.8, 144.9 -37.8, 144.9 -37.7))"\n')
            csv_path = f.name

        try:
            result = parcels.read_from_file(Path(csv_path))

            # Verify null handling
            assert len(result) == 2
            # Empty string should be preserved or converted to null
            assert pd.isna(result.loc[1, "value1"]) or result.loc[1, "value1"] == ""
            # "NA" should be treated as string, not null (unless explicitly configured)
            # Missing value (,,) should be null
            assert pd.isna(result.loc[2, "value2"])

        finally:
            Path(csv_path).unlink()

    def test_csv_no_id_column(self):
        """Test CSV reading when id_column doesn't exist - should generate unique IDs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,geometry\n")
            f.write('"Test 1","POLYGON((144.9 -37.8, 145.0 -37.8, 145.0 -37.9, 144.9 -37.9, 144.9 -37.8))"\n')
            f.write('"Test 2","POLYGON((144.9 -37.7, 145.0 -37.7, 145.0 -37.8, 144.9 -37.8, 144.9 -37.7))"\n')
            csv_path = f.name

        try:
            result = parcels.read_from_file(Path(csv_path))

            # Should have generated unique IDs
            assert len(result) == 2
            assert result.index.name == AOI_ID_COLUMN_NAME
            # Index should be unique
            assert not result.index.duplicated().any()

        finally:
            Path(csv_path).unlink()

    def test_large_csv_type_consistency(self):
        """Test that type inference is consistent across a larger file (simulating chunking scenario)."""
        # Create a CSV where early rows would suggest numeric type, but later rows have text
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(f"{AOI_ID_COLUMN_NAME},street_number,geometry\n")
            # First 100 rows are all numeric
            for i in range(1, 101):
                f.write(f'{i},"{i * 10}","POLYGON((144.9 -37.8, 145.0 -37.8, 145.0 -37.9, 144.9 -37.9, 144.9 -37.8))"\n')
            # Row 101 has text
            f.write(f'101,"10A","POLYGON((144.9 -37.8, 145.0 -37.8, 145.0 -37.9, 144.9 -37.9, 144.9 -37.8))"\n')
            csv_path = f.name

        try:
            result = parcels.read_from_file(Path(csv_path))

            # Verify all rows read
            assert len(result) == 101

            # Verify type is consistently string/object (not numeric then failing)
            assert result.loc[1, "street_number"] == "10"
            assert result.loc[101, "street_number"] == "10A"

            # Both should be same type
            assert type(result.loc[1, "street_number"]) == type(result.loc[101, "street_number"])

        finally:
            Path(csv_path).unlink()

    def test_parquet_with_aoi_id_column_no_spurious_index(self):
        """Regression: reading a parquet where aoi_id is a column (not the index)
        must not create a spurious 'index' column from the original RangeIndex.

        Previously, reset_index() created an 'index' column that was only dropped
        when aoi_id was missing. When aoi_id existed as a column, the 'index'
        column leaked through, propagating row positions into all output files.
        """
        gdf = gpd.GeoDataFrame({
            AOI_ID_COLUMN_NAME: [10, 20, 30],
            "name": ["A", "B", "C"],
            "geometry": [
                Polygon([(144.9, -37.8), (145.0, -37.8), (145.0, -37.9), (144.9, -37.9)]),
                Polygon([(144.9, -37.7), (145.0, -37.7), (145.0, -37.8), (144.9, -37.8)]),
                Polygon([(144.9, -37.6), (145.0, -37.6), (145.0, -37.7), (144.9, -37.7)]),
            ],
        }, crs="EPSG:4326")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            # Write with aoi_id as a column, not the index (default RangeIndex)
            gdf.to_parquet(parquet_path, index=False)

            result = parcels.read_from_file(Path(parquet_path))

            assert result.index.name == AOI_ID_COLUMN_NAME, "aoi_id should be the index"
            assert "index" not in result.columns, (
                "Spurious 'index' column from RangeIndex leaked into output"
            )
            assert list(result.index) == [10, 20, 30], "aoi_id values should be preserved"
            assert "name" in result.columns, "Data columns should be preserved"
        finally:
            Path(parquet_path).unlink()

    def test_parquet_with_named_index_preserved(self):
        """A meaningful named index (not aoi_id) should be preserved as a column
        when read_from_file resets it to use aoi_id instead."""
        gdf = gpd.GeoDataFrame({
            "property_id": ["P1", "P2", "P3"],
            AOI_ID_COLUMN_NAME: [100, 200, 300],
            "geometry": [
                Polygon([(144.9, -37.8), (145.0, -37.8), (145.0, -37.9), (144.9, -37.9)]),
                Polygon([(144.9, -37.7), (145.0, -37.7), (145.0, -37.8), (144.9, -37.8)]),
                Polygon([(144.9, -37.6), (145.0, -37.6), (145.0, -37.7), (144.9, -37.7)]),
            ],
        }, crs="EPSG:4326").set_index("property_id")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = f.name

        try:
            gdf.to_parquet(parquet_path)

            result = parcels.read_from_file(Path(parquet_path))

            assert result.index.name == AOI_ID_COLUMN_NAME
            assert "property_id" in result.columns, (
                "Named index should be preserved as a column after reset_index"
            )
            assert "index" not in result.columns, (
                "No spurious 'index' column should exist"
            )
        finally:
            Path(parquet_path).unlink()
