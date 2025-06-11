import argparse
import concurrent.futures
import os
from pathlib import Path
from typing import List, Optional
import json
import sys
from enum import Enum

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import traceback
import shapely.geometry  # Add this import for checking geometry types

warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")

from nmaipy import log, parcels
from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    SINCE_COL_NAME,
    UNTIL_COL_NAME,
    API_CRS,
    SURVEY_RESOURCE_ID_COL_NAME,
    DEFAULT_URL_ROOT,
    ADDRESS_FIELDS,
    BUILDING_STYLE_CLASS_IDS,
)
from nmaipy.feature_api import FeatureApi

import psutil
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import gc
import atexit
import signal


class Endpoint(Enum):
    FEATURE = "feature"
    ROLLUP = "rollup"


CHUNK_SIZE = 500
PROCESSES = 4
THREADS = 20

logger = log.get_logger()


def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi-file", help="Input AOI file path or S3 URL", type=str, required=True)
    parser.add_argument("--output-dir", help="Directory to store results", type=str, required=True)
    parser.add_argument(
        "--packs",
        help="List of AI packs",
        type=str,
        nargs="+",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--classes",
        help="List of Feature Class IDs (UUIDs)",
        type=str,
        nargs="+",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--include",
        help="List of additional data to include (e.g. roofSpotlightIndex, roofConditionConfidenceStats)",
        type=str,
        nargs="+",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--primary-decision",
        help="Primary feature decision method: largest_intersection|nearest",
        type=str,
        required=False,
        default="largest_intersection",
    )
    parser.add_argument(
        "--aoi-grid-min-pct",
        help="The minimum threshold (0-100) for how much of a grid cell (proportion of squares) must get a successful result (not 404) to return. Default is strict full coverage (100) required.",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument(
        "--aoi-grid-inexact",
        help="Permit inexact merging of large AOIs that get gridded, end up getting grid squares from multiple dates, then merging. Deduplication will work poorly for things like buildings.",
        action="store_true",
    )
    parser.add_argument(
        "--processes",
        help="Number of processes",
        type=int,
        required=False,
        default=PROCESSES,
    )
    parser.add_argument(
        "--threads",
        help="Number of threads",
        type=int,
        required=False,
        default=THREADS,
    )
    parser.add_argument(
        "--chunk-size",
        help="Number of AOIs to process in a single temporarily stored chunk file. Smaller files increase parallelism.",
        type=int,
        required=False,
        default=CHUNK_SIZE,
    )
    parser.add_argument(
        "--calc-buffers",
        help="Calculate buffered features of trees around buildings (compute expensive) within the query AOI (null otherwise).",
        action="store_true",
    )
    parser.add_argument(
        "--include-parcel-geometry",
        help="If set, parcel geometries will be in the output",
        action="store_true",
    )
    parser.add_argument(
        "--no-parcel-mode",
        help="If set, disable the API's parcel mode (which filters features based on parcel boundaries)",
        action="store_true",
    )
    parser.add_argument(
        "--save-features",
        help="If set, save the raw vectors as a geoparquet file for loading in GIS tools. This can be quite time consuming.",
        action="store_true",
    )
    parser.add_argument(
        "--save-buildings",
        help="If set, save a building-level geoparquet file with one row per building feature and associated attributes.",
        action="store_true",
    )
    parser.add_argument(
        "--rollup-format",
        help="csv | parquet: Whether to store output as .csv or .parquet (defaults to csv)",
        type=str,
        required=False,
        default="csv",
    )
    parser.add_argument(
        "--cache-dir",
        help="Location to store cache.",
        required=False,
    )
    parser.add_argument(
        "--no-cache",
        help="If set, turn off cache.",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite-cache",
        help="If set, ignore the existing cache and overwrite files as they are downloaded.",
        action="store_true",
    )
    parser.add_argument(
        "--compress-cache",
        help="If set, use gzip compression on each json payload in the cache.",
        action="store_true",
    )
    parser.add_argument(
        "--country",
        help="Country code for area calculations (au, us, ca, nz)",
        required=True,
    )
    parser.add_argument(
        "--alpha",
        help="Include alpha layers",
        action="store_true",
    )
    parser.add_argument(
        "--beta",
        help="Include beta layers",
        action="store_true",
    )
    parser.add_argument(
        "--prerelease",
        help="Include prerelease system versions",
        action="store_true",
    )
    parser.add_argument(
        "--only3d",
        help="Restrict date based queries to 3D data only",
        action="store_true",
    )
    parser.add_argument(
        "--since",
        help="Bulk limit on date for responses (earliest inclusive date returned). Presence of 'since' column in data takes precedent.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--until",
        help="Bulk limit on date for responses (earliest inclusive date returned). Presence of 'until' column in data takes precedent.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--endpoint",
        help="Select which endpoint gets used for rollups - 'feature' (default) or 'rollup'",
        type=str,
        required=False,
        default="feature",
    )
    parser.add_argument(
        "--url-root",
        help="Overwrite the root URL with a custom one.",
        type=str,
        required=False,
        default=DEFAULT_URL_ROOT,
    )
    parser.add_argument(
        "--system-version-prefix",
        help="Restrict responses to a specific system version generation (e.g. gen6-).",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--system-version",
        help="Restrict responses to a specific system version (e.g. gen6-glowing_grove-1.0).",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--api-key",
        help="API key to use (overrides API_KEY environment variable)",
        type=str,
        required=False,
    )
    parser.add_argument("--log-level", help="Log level (DEBUG, INFO, ...)", required=False, default="INFO", type=str)
    return parser.parse_args()


def cleanup_process_resources():
    """Helper to ensure processes are cleaned up"""
    import gc
    gc.collect()
    # Force cleanup of any remaining ProcessPoolExecutor threads
    concurrent.futures.process._threads_wakeups.clear()

def cleanup_thread_sessions(executor):
    """Helper to ensure thread sessions are properly closed"""
    if hasattr(executor, '_threads'):
        for thread in executor._threads:
            if hasattr(thread, '_local'):
                if hasattr(thread._local, 'session'):
                    try:
                        thread._local.session.close()
                    except:
                        pass

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**3  # Convert to GB
    return mem


class AOIExporter:
    def __init__(
        self,
        aoi_file='default_aoi_file',
        output_dir='default_output_dir',
        packs=None,
        classes=None,
        include=None,
        primary_decision='largest_intersection',
        aoi_grid_min_pct=100,
        aoi_grid_inexact=False,
        processes=PROCESSES,
        threads=THREADS,
        chunk_size=CHUNK_SIZE,
        calc_buffers=False,
        include_parcel_geometry=False,
        save_features=False,
        save_buildings=False,
        rollup_format='csv',
        cache_dir=None,
        no_cache=False,
        overwrite_cache=False,
        compress_cache=False,
        country='us',
        alpha=False,
        beta=False,
        prerelease=False,
        only3d=False,
        since=None,
        until=None,
        endpoint='feature',
        url_root=DEFAULT_URL_ROOT,
        system_version_prefix=None,
        system_version=None,
        log_level='INFO',
        api_key=None,  # Add API key parameter
        parcel_mode=True,  # Add parcel mode parameter with default True
    ):
        # Assign parameters to instance variables
        self.aoi_file = aoi_file
        self.output_dir = output_dir
        self.packs = packs
        self.classes = classes
        self.include = include
        self.primary_decision = primary_decision
        self.aoi_grid_min_pct = aoi_grid_min_pct
        self.aoi_grid_inexact = aoi_grid_inexact
        self.processes = processes
        self.threads = threads
        self.chunk_size = chunk_size
        self.calc_buffers = calc_buffers
        self.include_parcel_geometry = include_parcel_geometry
        self.save_features = save_features
        self.save_buildings = save_buildings
        self.rollup_format = rollup_format
        self.cache_dir = cache_dir
        self.no_cache = no_cache
        self.overwrite_cache = overwrite_cache
        self.compress_cache = compress_cache
        self.country = country
        self.alpha = alpha
        self.beta = beta
        self.prerelease = prerelease
        self.only3d = only3d
        self.since = since
        self.until = until
        self.endpoint = endpoint
        self.url_root = url_root
        self.system_version_prefix = system_version_prefix
        self.system_version = system_version
        self.log_level = log_level
        self.api_key_param = api_key  # Store the API key parameter
        self.parcel_mode = parcel_mode  # Store the parcel mode parameter

        # Configure logger
        log.configure_logger(self.log_level)
        self.logger = log.get_logger()

    def api_key(self) -> str:
        # Use provided API key if available, otherwise fall back to environment variable
        if hasattr(self, 'api_key_param') and self.api_key_param is not None:
            return self.api_key_param
        return os.getenv("API_KEY")

    def _stream_and_convert_features(self, feature_paths: List[Path], outpath_features: Path) -> Optional[gpd.GeoDataFrame]:
        """
        Stream feature chunks directly to a geoparquet file.
        This approach avoids loading all chunks into memory simultaneously.
        
        Args:
            feature_paths: List of paths to feature chunk parquet files
            outpath_features: Output path for final geoparquet file
            
        Returns:
            None since we don't need the GeoDataFrame in memory
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        import json
        
        pqwriter = None
        reference_columns = None  # Store column order from first chunk
        reference_schema = None   # Store PyArrow schema from first chunk
        
        # Stream chunks directly to geoparquet
        for i, cp in enumerate(tqdm(feature_paths, desc="Streaming chunks", file=sys.stdout)):
            try:
                df_feature_chunk = gpd.read_parquet(cp)
            except Exception as e:
                self.logger.error(f"Failed to read {cp}: {e}")
                continue
                
            if len(df_feature_chunk) > 0:
                # Store CRS and column schema from first chunk
                if reference_columns is None:
                    reference_columns = df_feature_chunk.columns

                # Validate and reorder columns for subsequent chunks
                current_columns = list(df_feature_chunk.columns)
                missing_cols = set(reference_columns) - set(current_columns)
                extra_cols = set(current_columns) - set(reference_columns)
                if missing_cols or extra_cols:
                    self.logger.warning(f"Chunk {i} schema mismatch detected:")
                    if missing_cols:
                        self.logger.warning(f"  Missing columns: {sorted(missing_cols)}")
                        # Add missing columns with null values
                        for col in missing_cols:
                            df_feature_chunk[col] = None
                    if extra_cols:
                        self.logger.warning(f"  Extra columns: {sorted(extra_cols)}")
                
                # Ensure column order matches reference for all chunks
                if list(current_columns) != list(reference_columns):
                    self.logger.debug(f"Chunk {i}: Column order differs from reference, reordering silently")
                    df_feature_chunk = df_feature_chunk[reference_columns]
                
                # Convert to regular pandas DataFrame for pyarrow, converting geometry to WKB
                geom_col = df_feature_chunk.geometry.to_wkb()
                df_feature_chunk = pd.DataFrame(df_feature_chunk)
                df_feature_chunk['geometry'] = geom_col

                # Convert to pyarrow table and stream
                table = pa.Table.from_pandas(df_feature_chunk, preserve_index=True)
                
                if pqwriter is None:
                    reference_schema = table.schema
                    
                    # Create geoparquet metadata from the start
                    geo_metadata = {
                        "version": "1.0.0",
                        "primary_column": "geometry",
                        "columns": {
                            "geometry": {
                                "encoding": "WKB",
                                "geometry_types": [],
                                "crs": API_CRS,
                                "edges": "planar",
                                "orientation": "counterclockwise",
                                "bbox": None
                            }
                        }
                    }
                    
                    # Add geoparquet metadata to schema
                    schema_with_geo = reference_schema.with_metadata({
                        b'geo': json.dumps(geo_metadata).encode('utf-8')
                    })
                    
                    pqwriter = pq.ParquetWriter(outpath_features, schema_with_geo)
                else:
                    # Cast to reference schema if needed
                    if table.schema != reference_schema:
                        try:
                            table = table.cast(reference_schema)
                        except Exception as e:
                            self.logger.error(f"Chunk {i}: Schema casting failed.")
                            self.logger.error(f"Error: {e}")
                            self.logger.error(f"Reference schema: {reference_schema}")
                            self.logger.error(f"Current schema: {table.schema}")
                pqwriter.write_table(table)
        
        # Close the writer
        if pqwriter is not None:
            pqwriter.close()
            
            # Log final status
            mem = psutil.virtual_memory()
            final_file_size_gb = outpath_features.stat().st_size / (1024**3)
            self.logger.debug(
                f"Successfully streamed to geoparquet without temporary files. "
                f"Memory: {(mem.total - mem.available)/1024**3:.2f}GB / {mem.total/1024**3:.2f}GB ({mem.percent:.1f}%). "
                f"Final file size: {final_file_size_gb:.2f}GB"
            )
            
            return None
        else:
            self.logger.warning("No feature data found to write")
            return None

    def process_chunk(self, chunk_id: str, aoi_gdf: gpd.GeoDataFrame, classes_df: pd.DataFrame):
        """
        Create a parcel rollup for a chunk of parcels.
        """
        feature_api = None
        start_mem = get_memory_usage()
        try:
            if self.cache_dir is None and not self.no_cache:
                cache_dir = Path(self.output_dir)
            else:
                cache_dir = Path(self.cache_dir) if self.cache_dir else Path(self.output_dir)

            if not self.no_cache:
                cache_path = cache_dir / "cache"
            else:
                cache_path = None

            chunk_path = Path(self.output_dir) / "chunks"
            outfile = chunk_path / f"rollup_{chunk_id}.parquet"
            outfile_features = chunk_path / f"features_{chunk_id}.parquet"
            outfile_errors = chunk_path / f"errors_{chunk_id}.parquet"
            if outfile.exists():
                return

            # Get additional parcel attributes from parcel geometry
            if isinstance(aoi_gdf, gpd.GeoDataFrame):
                rep_point = aoi_gdf.representative_point()
                aoi_gdf["query_aoi_lat"] = rep_point.y
                aoi_gdf["query_aoi_lon"] = rep_point.x

            # Get features
            feature_api = FeatureApi(
                api_key=self.api_key(),
                cache_dir=cache_path,
                overwrite_cache=self.overwrite_cache,
                compress_cache=self.compress_cache,
                threads=self.threads,
                alpha=self.alpha,
                beta=self.beta,
                prerelease=self.prerelease,
                only3d=self.only3d,
                url_root=self.url_root,
                system_version_prefix=self.system_version_prefix,
                system_version=self.system_version,
                aoi_grid_min_pct=self.aoi_grid_min_pct,
                aoi_grid_inexact=self.aoi_grid_inexact,
                parcel_mode=self.parcel_mode,
            )
            if self.endpoint == Endpoint.ROLLUP.value:
                self.logger.debug(f"Chunk {chunk_id}: Getting rollups for {len(aoi_gdf)} AOIs ({self.endpoint=})")
                rollup_df, metadata_df, errors_df = feature_api.get_rollup_df_bulk(
                    aoi_gdf,
                    region=self.country,
                    since_bulk=self.since,
                    until_bulk=self.until,
                    packs=self.packs,
                    classes=self.classes,
                    max_allowed_error_pct=100,
                )
                rollup_df.columns = FeatureApi._multi_to_single_index(rollup_df.columns)
                mem = psutil.virtual_memory()
                self.logger.debug(
                    f"Chunk {chunk_id} failed {len(errors_df)} of {len(aoi_gdf)} AOI requests. "
                    f"{len(rollup_df)} rollups returned on {len(rollup_df.index.unique())} unique {rollup_df.index.name}s. "
                    f"Memory: {(mem.total - mem.available)/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB ({mem.percent}%)"
                )
                if len(errors_df) > 0:
                    if "message" in errors_df:
                        self.logger.debug(errors_df.value_counts("message"))
                    else:
                        self.logger.debug(f"Found {len(errors_df)} errors")
                if len(errors_df) == len(aoi_gdf):
                    errors_df.to_parquet(outfile_errors)
                    return
            elif self.endpoint == Endpoint.FEATURE.value:
                self.logger.debug(f"Chunk {chunk_id}: Getting features for {len(aoi_gdf)} AOIs ({self.endpoint=})")
                features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(
                    aoi_gdf,
                    region=self.country,
                    since_bulk=self.since,
                    until_bulk=self.until,
                    packs=self.packs,
                    classes=self.classes,
                    include=self.include,
                    max_allowed_error_pct=100,
                )
                mem = psutil.virtual_memory()
                self.logger.debug(
                    f"Chunk {chunk_id} failed {len(errors_df)} of {len(aoi_gdf)} AOI requests. "
                    f"Memory: {(mem.total - mem.available)/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB ({mem.percent}%)"
                )
                if len(errors_df) > 0:
                    if "message" in errors_df:
                        self.logger.debug(errors_df.value_counts("message"))
                    else:
                        self.logger.debug(f"Found {len(errors_df)} errors")
                if len(errors_df) == len(aoi_gdf):
                    errors_df.to_parquet(outfile_errors)
                    return

                # Create rollup
                rollup_df = parcels.parcel_rollup(
                    aoi_gdf,
                    features_gdf,
                    classes_df,
                    country=self.country,
                    calc_buffers=self.calc_buffers,
                    primary_decision=self.primary_decision,
                )
            else:
                self.logger.error(f"Not a valid endpoint selection: {self.endpoint}")
                sys.exit(1)

            # Put it all together and save
            meta_data_columns = [
                "system_version",
                "link",
                "date",
                "survey_id",
                "survey_resource_id",
                "perspective",
                "postcat",
            ]
            # Validate columns
            for meta_data_column in meta_data_columns:
                if meta_data_column in aoi_gdf.columns:
                    metadata_df = metadata_df.rename(columns={meta_data_column: f"nmaipy_{meta_data_column}"})
                    meta_data_columns.remove(meta_data_column)

            final_df = metadata_df.merge(rollup_df, on=AOI_ID_COLUMN_NAME).merge(aoi_gdf, on=AOI_ID_COLUMN_NAME)
            parcel_columns = [c for c in aoi_gdf.columns if c != "geometry"]
            columns = (
                parcel_columns
                + [c for c in meta_data_columns if c in final_df.columns]
                + [c for c in final_df.columns if c not in parcel_columns + meta_data_columns + ["geometry"]]
            )
            final_df = final_df[columns]
            if self.include_parcel_geometry:
                columns.append("geometry")
            columns = [c for c in columns if c in final_df.columns]
            date2str = lambda d: str(d).replace("-", "")
            make_link = (
                lambda d: f"https://apps.nearmap.com/maps/#/@{d.query_aoi_lat},{d.query_aoi_lon},21.00z,0d/V/{date2str(d.date)}?locationMarker"
            )
            if self.endpoint == Endpoint.ROLLUP.value:
                if "query_aoi_lat" in final_df.columns and "query_aoi_lon" in final_df.columns:
                    final_df["link"] = final_df.apply(make_link, axis=1)
                final_df = final_df.drop(columns=["system_version", "date"])
            self.logger.debug(f"Chunk {chunk_id}: Writing {len(final_df)} rows for rollups and {len(errors_df)} for errors.")
            try:
                errors_df.to_parquet(outfile_errors)
            except Exception as e:
                self.logger.error(f"Chunk {chunk_id}: Failed writing errors_df ({len(errors_df)} rows) to {outfile_errors}.")
                self.logger.error(errors_df.shape)
                self.logger.error(errors_df)
            try:
                # Handle the geometry column separately to avoid conversion issues
                has_geometry = "geometry" in final_df.columns
                geometry_series = None
                
                if has_geometry:
                    # Store the geometry column separately
                    geometry_series = final_df["geometry"]
                    final_df = final_df.drop(columns=["geometry"])
                
                # Convert dtypes on the dataframe without geometry
                final_df = final_df.convert_dtypes()
                
                if has_geometry:
                    # Reattach the geometry column
                    final_df["geometry"] = geometry_series
                    # Create a proper GeoDataFrame
                    final_df = gpd.GeoDataFrame(final_df, geometry="geometry", crs=API_CRS)
                
                final_df.to_parquet(outfile)
            except Exception as e:
                self.logger.error(f"Chunk {chunk_id}: Failed writing final_df ({len(final_df)} rows) to {outfile}.")
                self.logger.error(f"Error type: {type(e).__name__}, Error message: {str(e)}")
                self.logger.error(final_df.shape)
                self.logger.error(final_df.dtypes)
                self.logger.error(final_df.head())
                self.logger.error(e)
            if self.save_features and (self.endpoint != Endpoint.ROLLUP.value):
                logger.debug(f"Chunk {chunk_id}: Saving {len(features_gdf)} features for {len(aoi_gdf)} AOIs")
                # Check for column name collisions between any two dataframes
                final_features_df = aoi_gdf.rename(columns=dict(geometry="aoi_geometry"))

                metadata_cols = set(metadata_df.columns)
                features_cols = set(features_gdf.columns)
                aoi_cols = set(final_features_df.columns)
                metadata_features_overlap = metadata_cols & features_cols - {AOI_ID_COLUMN_NAME}
                metadata_aoi_overlap = metadata_cols & aoi_cols - {AOI_ID_COLUMN_NAME}
                features_aoi_overlap = features_cols & aoi_cols - {AOI_ID_COLUMN_NAME}
                all_overlapping = metadata_features_overlap | metadata_aoi_overlap | features_aoi_overlap
                if all_overlapping:
                    self.logger.warning(
                        f"Column name collisions detected. The following columns exist in multiple dataframes "
                        f"and may be duplicated with '_x' and '_y' suffixes: {sorted(all_overlapping)}"
                    )
                final_features_df = gpd.GeoDataFrame(
                    metadata_df.merge(features_gdf, on=AOI_ID_COLUMN_NAME).merge(
                        final_features_df, on=AOI_ID_COLUMN_NAME
                    ),
                    crs=API_CRS,
                )
                if "aoi_geometry" in final_features_df.columns:
                    final_features_df["aoi_geometry"] = final_features_df.aoi_geometry.to_wkt()
                final_features_df["attributes"] = final_features_df.attributes.apply(json.dumps)
                if len(final_features_df) > 0:
                    try:
                        if not self.include_parcel_geometry and "aoi_geometry" in final_features_df.columns:
                            final_features_df = final_features_df.drop(columns=["aoi_geometry"])
                        final_features_df = final_features_df[
                            ~(final_features_df.geometry.is_empty | final_features_df.geometry.isna())
                        ]
                        # Ensure it's a proper GeoDataFrame before saving to parquet
                        if not isinstance(final_features_df, gpd.GeoDataFrame):
                            final_features_df = gpd.GeoDataFrame(final_features_df, geometry="geometry", crs=API_CRS)
                        else:
                            final_features_df = final_features_df.set_crs(API_CRS, allow_override=True)
                        final_features_df.to_parquet(outfile_features)
                    except Exception as e:
                        self.logger.error(
                            f"Failed to save features parquet file for chunk_id {chunk_id}. Errors saved to {outfile_errors}. Rollup saved to {outfile}."
                        )
                        self.logger.error(f"Error type: {type(e).__name__}, Error message: {str(e)}")
                        self.logger.error(e)
            self.logger.debug(f"Finished saving chunk {chunk_id}")
            
            # Force cleanup
            if feature_api:
                feature_api.cleanup()
                del feature_api
            if "rollup_df" in locals():
                del rollup_df
            if "features_gdf" in locals():
                del features_gdf
            if "metadata_df" in locals():
                del metadata_df
            if "errors_df" in locals():
                del errors_df
            if "final_df" in locals():
                del final_df
            if 'final_features_df' in locals():
                del final_features_df
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_id}: {e}")
            raise
        finally:
            # Only cleanup if feature_api has been declared as a variable
            if "feature_api" in locals():
                try:
                    feature_api.cleanup()
                except:
                    pass
                del feature_api
            gc.collect()

    def run(self):
        self.logger.debug("Starting parcel rollup")

        # Process a single AOI file
        aoi_path = self.aoi_file
        self.logger.info(f"Processing AOI file {aoi_path}")

        cache_path = Path(self.output_dir) / "cache"
        chunk_path = Path(self.output_dir) / "chunks"
        final_path = Path(self.output_dir) / "final"
        cache_path.mkdir(parents=True, exist_ok=True)
        chunk_path.mkdir(parents=True, exist_ok=True)
        final_path.mkdir(parents=True, exist_ok=True)

        # Get classes
        feature_api = FeatureApi(
                api_key=self.api_key(),
                alpha=self.alpha,
                beta=self.beta,
                prerelease=self.prerelease,
                only3d=self.only3d,
                parcel_mode=self.parcel_mode
            )
        if self.packs is not None:
            classes_df = feature_api.get_feature_classes(self.packs)
        else:
            classes_df = feature_api.get_feature_classes() # All classes
            if self.classes is not None:
                classes_df = classes_df[classes_df.index.isin(self.classes)]

        # Modify output file paths using the AOI file name
        outpath = final_path / f"{Path(aoi_path).stem}.{self.rollup_format}"
        outpath_features = final_path / f"{Path(aoi_path).stem}_features.parquet"
        outpath_buildings = final_path / f"{Path(aoi_path).stem}_buildings.{self.rollup_format}"

        # Check if all required outputs already exist
        outputs_exist = outpath.exists()
        if self.save_features:
            outputs_exist = outputs_exist and outpath_features.exists()
        if self.save_buildings:
            outputs_exist = outputs_exist and outpath_buildings.exists()
            
        if outputs_exist:
            self.logger.info(f"Output already exists, skipping {Path(aoi_path).stem}")
            return

        aoi_gdf = parcels.read_from_file(aoi_path, id_column=AOI_ID_COLUMN_NAME)

        if isinstance(aoi_gdf, gpd.GeoDataFrame):
            aoi_gdf = aoi_gdf.to_crs(API_CRS)
        else:
            self.logger.info("No geometry found in parcel data - using address fields")
            for field in ADDRESS_FIELDS:
                if field not in aoi_gdf:
                    self.logger.error(f"Missing field {field} in parcel data")
                    sys.exit(1)

        # Print out info around what is being inferred from column names:
        if SURVEY_RESOURCE_ID_COL_NAME in aoi_gdf:
            logger.info(
                f"{SURVEY_RESOURCE_ID_COL_NAME} will be used to get results from the exact Survey Resource ID, instead of using date based filtering."
            )
        else:
            logger.info(f"No {SURVEY_RESOURCE_ID_COL_NAME} column provided, so date based endpoint will be used.")
            if SINCE_COL_NAME in aoi_gdf:
                logger.info(
                    f'The column "{SINCE_COL_NAME}" will be used as the earliest permitted date (YYYY-MM-DD) for each Query AOI.'
                )
            elif self.since is not None:
                logger.info(f"The since date of {self.since} will limit the earliest returned date for all Query AOIs")
            else:
                logger.info("No earliest date will used")
            if UNTIL_COL_NAME in aoi_gdf:
                logger.info(
                    f'The column "{UNTIL_COL_NAME}" will be used as the latest permitted date (YYYY-MM-DD) for each Query AOI.'
                )
            elif self.until is not None:
                logger.info(f"The until date of {self.until} will limit the latest returned date for all Query AOIs")
            else:
                logger.info("No latest date will used")

        jobs = []
        num_chunks = max(len(aoi_gdf) // self.chunk_size, 1)
        self.logger.info(f"Exporting {len(aoi_gdf)} AOIs as {num_chunks} chunk files.")
        self.logger.info(f"Using endpoint '{self.endpoint}' for rollups.")

        # Split the parcels into chunks
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Geometry is in a geographic CRS.",
            )
            if isinstance(aoi_gdf, gpd.GeoDataFrame):
                aoi_gdf = aoi_gdf[aoi_gdf.area > 0]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'GeoDataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'GeoDataFrame.transpose' instead.",
            )
            warnings.filterwarnings(
                "ignore",
                message="'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead.",
            )
            chunks = np.array_split(aoi_gdf, num_chunks)
        processes = int(self.processes)
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                with ProcessPoolExecutor(processes) as executor:
                    try:
                        for i, batch in enumerate(chunks):
                            chunk_id = f"{Path(aoi_path).stem}_{str(i).zfill(4)}"
                            self.logger.debug(
                                f"Parallel processing chunk {chunk_id} - min {batch.index.name} is {batch.index.min()}, max {AOI_ID_COLUMN_NAME} is {batch.index.max()}"
                            )
                            jobs.append(
                                executor.submit(
                                    self.process_chunk,
                                    chunk_id,
                                    batch,
                                    classes_df,
                                )
                            )
                        for j in tqdm(jobs, desc="Processing chunks", file=sys.stdout):
                            try:
                                j.result()
                            except Exception as e:
                                self.logger.error(f"FAILURE TO COMPLETE JOB {j}, DROPPING DUE TO ERROR {e}")
                                self.logger.error(f"{sys.exc_info()}\t{traceback.format_exc()}")
                                cleanup_thread_sessions(executor)
                                executor.shutdown(wait=False)
                                raise
                        break  # Success - exit retry loop
                    finally:
                        cleanup_thread_sessions(executor)
                        executor.shutdown(wait=True)
                        cleanup_process_resources()
            except BrokenProcessPool as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Process pool broken, attempt {attempt + 1}/{max_retries}, retrying after {retry_delay}s delay...")
                    cleanup_process_resources()
                    time.sleep(retry_delay)
                    jobs = []  # Reset jobs list for retry
                else:
                    self.logger.error(f"Process pool broken after {max_retries} attempts, giving up")
                    cleanup_process_resources()
                    raise

        data = []
        data_features = []
        errors = []
        self.logger.info(f"Saving rollup data as {self.rollup_format} file to {outpath}")
        for i in range(num_chunks):
            chunk_filename = f"rollup_{Path(aoi_path).stem}_{str(i).zfill(4)}.parquet"
            cp = chunk_path / chunk_filename
            if cp.exists():
                try:
                    chunk = gpd.read_parquet(cp)
                except ValueError:
                    chunk = pd.read_parquet(cp)
                if len(chunk) > 0:
                    data.append(chunk)
            else:
                error_filename = f"errors_{Path(aoi_path).stem}_{str(i).zfill(4)}.parquet"
                if (chunk_path / error_filename).exists():
                    self.logger.debug(f"Chunk {i} rollup file missing, but error file found.")
                else:
                    self.logger.error(f"Chunk {i} rollup and error files missing. Try rerunning.")
                    sys.exit(1)
        if len(data) > 0:
            data = pd.concat([data for data in data if len(data) > 0])
            if "geometry" in data.columns:
                if not isinstance(data.geometry, gpd.GeoSeries):
                    data["geometry"] = gpd.GeoSeries.from_wkt(data.geometry)
                data = gpd.GeoDataFrame(data, crs=API_CRS)
            
        else:
            data = pd.DataFrame(data)
        if len(data) > 0:
            if self.rollup_format == "parquet":
                data.to_parquet(outpath, index=True)
            elif self.rollup_format == "csv":
                if "geometry" in data.columns:
                    if hasattr(data.geometry, "to_wkt") and callable(data.geometry.to_wkt):
                        # If it has a to_wkt method but isn't a GeoSeries
                        data["geometry"] = data.geometry.to_wkt()
                data.to_csv(outpath, index=True)
            else:
                self.logger.info("Invalid output format specified - reverting to csv")
                if "geometry" in data.columns:
                    if hasattr(data.geometry, "to_wkt") and callable(data.geometry.to_wkt):
                        # If it has a to_wkt method but isn't a GeoSeries
                        data["geometry"] = data.geometry.to_wkt()
                data.to_csv(outpath, index=True)
        outpath_errors = final_path / f"{Path(aoi_path).stem}_errors.csv"
        self.logger.info(f"Saving error data as .csv to {outpath_errors}")
        for cp in chunk_path.glob(f"errors_{Path(aoi_path).stem}_*.parquet"):
            errors.append(pd.read_parquet(cp))
        if len(errors) > 0:
            errors = pd.concat(errors)
            
            # Count error types by status code and message
            error_summary = []
            if 'status_code' in errors.columns:
                status_counts = errors['status_code'].value_counts()
                error_summary.append(f"status codes: {status_counts.to_dict()}")
            if 'message' in errors.columns:
                message_counts = errors['message'].value_counts()
                error_summary.append(f"messages: {message_counts.to_dict()}")
            
            if error_summary:
                self.logger.info(f"Processing completed with {len(errors)} total failures - {', '.join(error_summary)}")
            else:
                self.logger.info(f"Processing completed with {len(errors)} total failures")
        else:
            errors = pd.DataFrame(errors)
            self.logger.info("Processing completed with no failures")
        errors.to_csv(outpath_errors, index=True)
        if self.save_features:
            feature_paths = [p for p in chunk_path.glob(f"features_{Path(aoi_path).stem}_*.parquet")]
            self.logger.info(f"Saving feature data from {len(feature_paths)} geoparquet chunks to {outpath_features}")
            
            features_gdf = self._stream_and_convert_features(feature_paths, outpath_features)
                
            # If buildings export is enabled, process building features
            if self.save_buildings:
                self.logger.info(f"Saving building-level data as {self.rollup_format} to {outpath_buildings}")
                # Define geoparquet path for buildings
                outpath_buildings_geoparquet = final_path / f"{Path(aoi_path).stem}_building_features.parquet"
                
                buildings_gdf = parcels.extract_building_features(
                    parcels_gdf=aoi_gdf,
                    features_gdf=features_gdf,
                    country=self.country
                )
                if len(buildings_gdf) > 0:
                    # First, save the geoparquet version with intact geometries
                    self.logger.info(f"Saving building-level data as geoparquet to {outpath_buildings_geoparquet}")
                    try:
                        buildings_gdf.to_parquet(outpath_buildings_geoparquet)
                    except Exception as e:
                        self.logger.error(f"Failed to save buildings geoparquet file: {str(e)}")
                        
                    # Then convert geodataframe to plain dataframe for tabular output
                    # Keep geometry as WKT representation if needed
                    buildings_df = pd.DataFrame(buildings_gdf)
                    if "geometry" in buildings_df.columns:
                        buildings_df["geometry"] = buildings_df.geometry.apply(lambda geom: geom.wkt if geom else None)
                    
                    # Save in the same format as rollup
                    if self.rollup_format == "parquet":
                        buildings_df.to_parquet(outpath_buildings, index=True)
                    elif self.rollup_format == "csv":
                        buildings_df.to_csv(outpath_buildings, index=True)
                    else:
                        self.logger.info("Invalid output format specified for buildings - reverting to csv")
                        buildings_df.to_csv(outpath_buildings, index=True)
                else:
                    self.logger.info(f"No building features found for {Path(aoi_path).stem}")


def main():
        # Set higher file descriptor limits for running many processes in parallel.
    import resource
    import sys

    if sys.platform != 'win32':
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = 32000  # Same as ulimit -n 32000
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
            new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            logger.info(f"File descriptor limits - Previous: {soft}, New: {new_soft}, Hard limit: {hard}")
        except ValueError as e:
            # If desired limit is too high, try setting to hard limit
            logger.warning(f"Could not set file descriptor limit to {desired}, trying hard limit {hard}")
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
                new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
                logger.info(f"File descriptor limits - Previous: {soft}, New: {new_soft}, Hard limit: {hard}")
            except ValueError as e:
                logger.warning(f"Could not increase file descriptor limits: {e}")
    args = parse_arguments()
    exporter = AOIExporter(
        aoi_file=args.aoi_file,
        output_dir=args.output_dir,
        packs=args.packs,
        classes=args.classes,
        include=args.include,
        primary_decision=args.primary_decision,
        aoi_grid_min_pct=args.aoi_grid_min_pct,
        aoi_grid_inexact=args.aoi_grid_inexact,
        processes=args.processes,
        threads=args.threads,
        chunk_size=args.chunk_size,
        calc_buffers=args.calc_buffers,
        include_parcel_geometry=args.include_parcel_geometry,
        save_features=args.save_features,
        save_buildings=args.save_buildings,
        rollup_format=args.rollup_format,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
        overwrite_cache=args.overwrite_cache,
        compress_cache=args.compress_cache,
        country=args.country,
        alpha=args.alpha,
        beta=args.beta,
        prerelease=args.prerelease,
        only3d=args.only3d,
        since=args.since,
        until=args.until,
        endpoint=args.endpoint,
        url_root=args.url_root,
        system_version_prefix=args.system_version_prefix,
        system_version=args.system_version,
        log_level=args.log_level,
        api_key=args.api_key,  # Pass API key argument
        parcel_mode=not args.no_parcel_mode,  # Default to True unless --no-parcel-mode is set
    )
    exporter.run()

if __name__ == "__main__":
    main()

atexit.register(cleanup_process_resources)
signal.signal(signal.SIGTERM, lambda *args: cleanup_process_resources())
