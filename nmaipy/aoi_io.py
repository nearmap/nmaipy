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

import datetime
import re
import warnings
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from nmaipy import log
from nmaipy.constants import ADDRESS_FIELDS, AOI_ID_COLUMN_NAME, LAT_LONG_CRS, SINCE_COL_NAME, UNTIL_COL_NAME

logger = log.get_logger()

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _is_iso_date(value) -> bool:
    """True for a ``YYYY-MM-DD`` string naming a real calendar date."""
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return False
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def normalise_date_column(values: pd.Series) -> pd.Series:
    """Return a per-AOI ``since``/``until`` column as ``YYYY-MM-DD`` strings, with null for no override.

    GeoJSON and GeoPackage readers parse date-like values as ``datetime64``, and a typed parquet column holds
    ``datetime.date`` objects; both become strings, discarding any time of day. An empty or blank string means no
    per-AOI override, as a null does, so it becomes null (CSV readers already give NaN there). Anything else is left
    for ``validate_date_cutoffs`` to accept or reject.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.strftime("%Y-%m-%d")

    def normalise(value):
        if isinstance(value, datetime.date):  # also datetime.datetime, pd.Timestamp and pd.NaT
            return None if pd.isna(value) else value.strftime("%Y-%m-%d")
        if isinstance(value, str) and not value.strip():
            return None
        return value

    return values.map(normalise)


def blank_date_to_none(value: Optional[str]) -> Optional[str]:
    """A bulk date cutoff that is empty or blank text means no cutoff: return ``None`` for it, else ``value``."""
    return None if isinstance(value, str) and not value.strip() else value


def validate_date_cutoffs(aoi_df: pd.DataFrame, since: Optional[str] = None, until: Optional[str] = None) -> None:
    """Raise ``ValueError`` unless every date cutoff is a ``YYYY-MM-DD`` date and no AOI's since is after its until.

    Args:
        aoi_df: AOIs indexed by id, optionally with per-AOI ``since``/``until`` columns already passed through
            ``normalise_date_column``. A null cell means the AOI uses the bulk value.
        since: Bulk earliest date, used where an AOI has no ``since`` of its own. Empty or blank means none.
        until: Bulk latest date, used where an AOI has no ``until`` of its own. Empty or blank means none.

    Raises:
        ValueError: If a bulk or per-AOI value is not a ``YYYY-MM-DD`` date string, or if an AOI's effective since
            is later than its effective until (the API rejects every such request).
    """
    # A blank bulk date sends no date filter, as a blank per-AOI cell does (FeatureApi and RoofAgeApi drop it).
    since, until = blank_date_to_none(since), blank_date_to_none(until)
    for flag, value in (("--since", since), ("--until", until)):
        if value is not None and not _is_iso_date(value):
            raise ValueError(f"{flag} must be a YYYY-MM-DD date, got {value!r}")

    effective = {}
    for col, bulk in ((SINCE_COL_NAME, since), (UNTIL_COL_NAME, until)):
        if col not in aoi_df.columns:
            effective[col] = pd.Series(bulk, index=aoi_df.index, dtype=object)
            continue
        values = aoi_df[col]
        # Check each distinct value once: an AOI file holds millions of rows but few distinct dates.
        malformed = [value for value in values.dropna().unique() if not _is_iso_date(value)]
        invalid = values[values.isin(malformed)]
        if len(invalid):
            sample = ", ".join(f"{aoi_id!r}: {value!r}" for aoi_id, value in invalid.head(5).items())
            raise ValueError(
                f"AOI file column '{col}' must hold YYYY-MM-DD date strings (or be empty for the bulk value); "
                f"{len(invalid)} row(s) don't, e.g. {sample}"
            )
        effective[col] = values.astype(object).where(values.notna(), bulk)

    both = effective[SINCE_COL_NAME].notna() & effective[UNTIL_COL_NAME].notna()
    inverted = both & (effective[SINCE_COL_NAME].where(both, "") > effective[UNTIL_COL_NAME].where(both, ""))
    if inverted.any():
        sample = ", ".join(
            f"{aoi_id!r}: since {effective[SINCE_COL_NAME][aoi_id]} > until {effective[UNTIL_COL_NAME][aoi_id]}"
            for aoi_id in inverted[inverted].index[:5]
        )
        raise ValueError(f"{int(inverted.sum())} AOI(s) have a since date after their until date, e.g. {sample}")


def prepare_date_cutoffs(
    aoi_df: pd.DataFrame, since: Optional[str] = None, until: Optional[str] = None
) -> pd.DataFrame:
    """Return ``aoi_df`` with its per-AOI ``since``/``until`` normalised, after validating every cutoff.

    For frames handed straight to a bulk query rather than read with ``read_from_file``. The caller's frame is not
    modified.

    Args:
        aoi_df: AOIs indexed by id, optionally with per-AOI ``since``/``until`` columns.
        since: Bulk earliest date, used where an AOI has no ``since`` of its own.
        until: Bulk latest date, used where an AOI has no ``until`` of its own.

    Returns:
        ``aoi_df``, with its date columns as ``YYYY-MM-DD`` strings or null (see ``normalise_date_column``).

    Raises:
        ValueError: As ``validate_date_cutoffs``.
    """
    columns = {
        col: normalise_date_column(aoi_df[col]) for col in (SINCE_COL_NAME, UNTIL_COL_NAME) if col in aoi_df.columns
    }
    if columns:
        aoi_df = aoi_df.assign(**columns)
    validate_date_cutoffs(aoi_df, since=since, until=until)
    return aoi_df


def _path_suffix(path: Union[str, Path]) -> str:
    """Return the lowercased extension of ``path``.

    Strips off any query string (``?key=val``) and fragment (``#x``) so that
    S3-style URIs like ``s3://bucket/file.csv?x=y`` resolve to ``"csv"``.
    Returns an empty string if the path has no extension.
    """
    s = str(path)
    s = s.split("?", 1)[0].split("#", 1)[0]
    return s.rsplit(".", 1)[-1].lower() if "." in s else ""


def read_header_columns(path: Union[str, Path]) -> set[str]:
    """Return the set of column names from an AOI file by reading only its header.

    Mirrors the format support of :func:`read_from_file` (CSV/PSV/TSV with WKT,
    GeoJSON, GeoPackage, Parquet/GeoParquet). Avoids materialising the full file.

    Raises:
        NotImplementedError: If the file format is not supported.
    """
    suffix = _path_suffix(path)
    sep_for = {"csv": ",", "psv": "|", "tsv": "\t"}
    if suffix in sep_for:
        return set(pd.read_csv(path, sep=sep_for[suffix], nrows=0).columns)
    if suffix == "parquet":
        # Parquet schema can be read without materialising any rows.
        return set(pq.read_schema(path).names)
    if suffix in ("geojson", "gpkg"):
        # geopandas reads the file; rows=0 limits to the header schema.
        return set(gpd.read_file(path, rows=0).columns)
    raise NotImplementedError(f"Source format not supported: {suffix=}")


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

    Note on date columns:
        Optional ``until`` and ``since`` columns go through ``normalise_date_column``: datetimes and
        dates become ``YYYY-MM-DD`` strings, and empty or blank cells become null (no per-AOI
        override). The exporter then rejects any value that is not a ``YYYY-MM-DD`` date
        (``validate_date_cutoffs``).

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

    # The API takes per-AOI cutoffs as YYYY-MM-DD strings; see normalise_date_column.
    for col in (SINCE_COL_NAME, UNTIL_COL_NAME):
        if col in parcels_gdf.columns:
            parcels_gdf[col] = normalise_date_column(parcels_gdf[col])

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
            parcels_gdf = parcels_gdf.set_crs(source_crs)
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
                logger.warning(f"Dropping {num_dropped} rows with empty or invalid geometries, or ones with zero area")

    if len(parcels_gdf) == 0:
        raise RuntimeError(f"No valid parcels in {path=}")

    # Resolve "aoi_id is both an index level and a column label" up-front: a
    # downstream merge(on=id_column) would raise ValueError on this shape, and
    # pandas commonly produces it via parquet round-trip — pandas writes the
    # index name into parquet metadata even for a RangeIndex carrying no values,
    # and the read back materialises a named-but-data-less index alongside the
    # real column. The two ways to produce this shape need opposite resolutions:
    #   * RangeIndex named id_column: the index has no information, the column
    #     is authoritative — drop the index.
    #   * Real-data index named id_column with a same-named column: the index
    #     has values that may differ from the column; preserving the index is
    #     the only conservative choice — drop the column. (No known producer
    #     hits this path; if you find one, please open an issue.)
    if parcels_gdf.index.name == id_column and id_column in parcels_gdf.columns:
        if isinstance(parcels_gdf.index, pd.RangeIndex):
            parcels_gdf = parcels_gdf.reset_index(drop=True)
        else:
            parcels_gdf = parcels_gdf.drop(columns=[id_column])

    # Check that identifier is unique
    if parcels_gdf.index.name != id_column:
        if parcels_gdf.index.name is not None:
            # Preserve named index as a column (it may contain useful data)
            parcels_gdf = parcels_gdf.reset_index()
        else:
            # Unnamed RangeIndex — discard it to avoid creating a spurious "index" column
            # (also prevents ValueError if the input already has a column called "index")
            parcels_gdf = parcels_gdf.reset_index(drop=True)
        if id_column not in parcels_gdf:
            logger.info(f"Missing {AOI_ID_COLUMN_NAME} column in parcel data - generating unique IDs")
            parcels_gdf.index.name = id_column  # Set a new unique ordered index for reference
        else:  # The existing aoi_id column must be promoted to the index
            logger.info(f"Moving existing {AOI_ID_COLUMN_NAME} column to be the index")
            parcels_gdf = parcels_gdf.set_index(id_column)

    if parcels_gdf.index.duplicated().any():
        raise ValueError(f"Duplicate IDs found for {id_column=}")

    return parcels_gdf
