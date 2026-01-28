"""
AOI (Area of Interest) File I/O Utilities

This module provides functions for reading and writing AOI data from various file formats.
Supported formats include:
- CSV/PSV/TSV with WKT geometry
- GeoJSON
- GeoPackage (GPKG)
- Parquet/GeoParquet

These functions handle CRS transformations, geometry validation, and unique identifier management.
"""
import warnings
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd

from nmaipy import log
from nmaipy.constants import ADDRESS_FIELDS, AOI_ID_COLUMN_NAME, LAT_LONG_CRS

logger = log.get_logger()


def read_from_file(
    path: Union[str, Path],
    drop_empty: Optional[bool] = True,
    id_column: Optional[str] = AOI_ID_COLUMN_NAME,
    source_crs: Optional[str] = LAT_LONG_CRS,
    target_crs: Optional[str] = LAT_LONG_CRS,
) -> gpd.GeoDataFrame:
    """
    Read AOI/parcel data from a file. Supported formats are:
     - CSV/PSV/TSV with geometries as WKTs
     - GPKG (GeoPackage)
     - GeoJSON
     - Parquet with geometries as WKBs (GeoParquet)

    Args:
        path: Path to file
        drop_empty: If true, rows with empty geometries will be dropped.
        id_column: Unique identifier column name. This column will be used as the index.
                   Defaults to the standard AOI ID column name.
        source_crs: CRS of the source data - defaults to lat/long (WGS84). If the source
                    data has a CRS set, this field is ignored.
        target_crs: CRS of data being returned.

    Returns:
        GeoDataFrame with AOI geometries and the specified id_column as the index.

    Raises:
        NotImplementedError: If the file format is not supported
        RuntimeError: If no valid parcels found in file
        ValueError: If duplicate IDs are found

    Example:
        >>> from nmaipy.aoi_io import read_from_file
        >>> aoi_gdf = read_from_file("parcels.geojson", id_column="parcel_id")
        >>> print(f"Loaded {len(aoi_gdf)} AOIs")
    """
    if isinstance(path, str):
        suffix = path.split(".")[-1]
    elif isinstance(path, Path):
        suffix = path.suffix[1:]

    if suffix in ("csv", "psv", "tsv"):
        # Determine separator based on file extension
        if suffix == "csv":
            sep = ","
        elif suffix == "psv":
            sep = "|"
        elif suffix == "tsv":
            sep = "\t"

        # Read CSV with robust type handling:
        # - Use low_memory=False to scan entire file for type inference
        # - This prevents issues with mixed types (e.g., numeric street addresses with some text)
        # - Ensures consistent typing even when chunking would see different types in different chunks
        # - Keep geometry as regular dtype (object) since we'll convert it to actual geometries
        dtype_overrides = None
        if "geometry" in pd.read_csv(path, sep=sep, nrows=0).columns:
            # If there's a geometry column, keep it as object dtype (not StringArray)
            # so we can convert it to actual geometry objects later
            dtype_overrides = {"geometry": "object"}

        parcels_gdf = pd.read_csv(
            path,
            sep=sep,
            low_memory=False,  # Scan whole file for proper type inference
            dtype=dtype_overrides,  # Keep geometry as object if present
        )

        # Set the index only if the column exists
        if id_column in parcels_gdf.columns:
            parcels_gdf = parcels_gdf.set_index(id_column)

    elif suffix == "parquet":
        # Try geopandas first for geoparquet files with geometry columns
        # Fall back to pandas for non-geo parquet (e.g., address-only files with explicit dtypes)
        # This preserves explicit dtypes that may have been carefully specified
        try:
            parcels_gdf = gpd.read_parquet(path)
            logger.info(f"Read geoparquet file with geometry using geopandas")
        except (ValueError, Exception) as e:
            # Handle both "Missing geo metadata" and other parquet reading issues
            if "Missing geo metadata" in str(e) or "geo" in str(e).lower():
                # Non-geo parquet file - read with pandas to preserve explicit dtypes
                logger.info(f"Reading non-geo parquet file with pandas (preserves explicit dtypes)")
                parcels_gdf = pd.read_parquet(path)
            else:
                # Unknown error - try pandas as fallback
                logger.warning(f"geopandas read failed with: {e}. Trying pandas fallback.")
                try:
                    parcels_gdf = pd.read_parquet(path)
                    logger.info(f"Successfully read parquet with pandas fallback")
                except Exception as e2:
                    logger.error(f"Both geopandas and pandas failed to read parquet file")
                    raise e2

    elif suffix in ("geojson", "gpkg"):
        parcels_gdf = gpd.read_file(path)

    else:
        raise NotImplementedError(f"Source format not supported: {suffix=}")

    if "geometry" not in parcels_gdf:
        logger.warning(f"Input file has no AOI geometries - some operations will not work.")
    else:
        if not isinstance(parcels_gdf, gpd.GeoDataFrame):
            # If from a tabular data source, try to convert to a GeoDataFrame (requires a geometry column)
            geometry = gpd.GeoSeries.from_wkt(parcels_gdf.geometry.fillna("POLYGON(EMPTY)"))
            parcels_gdf = gpd.GeoDataFrame(
                parcels_gdf,
                geometry=geometry,
                crs=source_crs,
            )

    # Check if both geometry and address fields are present - geometry will take priority
    has_geometry = "geometry" in parcels_gdf.columns
    has_address_fields = set(parcels_gdf.columns).issuperset(set(ADDRESS_FIELDS))
    if has_geometry and has_address_fields:
        logger.info(
            f"Input has both geometry and address fields - geometry mode will be used where available. "
            f"Address fields {ADDRESS_FIELDS} will be ignored for API queries if geometry is present (falling back to address mode if not)."
        )

    if "geometry" in parcels_gdf:
        # Set CRS and project if data CRS is not equal to target CRS
        if parcels_gdf.crs is None:
            parcels_gdf.set_crs(source_crs)
        if parcels_gdf.crs != target_crs:
            parcels_gdf = parcels_gdf.to_crs(target_crs)

        # Drop any empty geometries
        if drop_empty:
            num_dropped = len(parcels_gdf)
            parcels_gdf = parcels_gdf.dropna(subset=["geometry"])
            parcels_gdf = parcels_gdf[~parcels_gdf.is_empty]
            parcels_gdf = parcels_gdf[parcels_gdf.is_valid]
            # For this we only check if the shape has a non-zero area, the value doesn't matter,
            # so the warning can be ignored.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.")
                parcels_gdf = parcels_gdf[parcels_gdf.area > 0]
            num_dropped -= len(parcels_gdf)
            if num_dropped > 0:
                logger.warning(
                    f"Dropping {num_dropped} rows with empty or invalid geometries, or ones with zero area"
                )

    if len(parcels_gdf) == 0:
        raise RuntimeError(f"No valid parcels in {path=}")

    # Check that identifier is unique
    if parcels_gdf.index.name != id_column:
        # Bump the index to a column in case it's important
        parcels_gdf = parcels_gdf.reset_index()
        if id_column not in parcels_gdf:
            logger.info(f"Missing {AOI_ID_COLUMN_NAME} column in parcel data - generating unique IDs")
            # Drop the "index" column if reset_index() created it from an unnamed integer index
            if "index" in parcels_gdf.columns:
                parcels_gdf = parcels_gdf.drop(columns=["index"])
            parcels_gdf.index.name = id_column  # Set a new unique ordered index for reference
        else:  # The index must already be there as a column
            logger.warning(f"Moving {AOI_ID_COLUMN_NAME} to be the index - generating unique IDs")
            parcels_gdf = parcels_gdf.set_index(id_column)

    if parcels_gdf.index.duplicated().any():
        raise ValueError(f"Duplicate IDs found for {id_column=}")

    return parcels_gdf
