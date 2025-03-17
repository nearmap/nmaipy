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
)
from nmaipy.feature_api import FeatureApi


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
        "--save-features",
        help="If set, save the raw vectors as a geoparquet file for loading in GIS tools. This can be quite time consuming.",
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


class AOIExporter:
    def __init__(
        self,
        aoi_file='default_aoi_file',
        output_dir='default_output_dir',
        packs=None,
        classes=None,
        primary_decision='largest_intersection',
        aoi_grid_min_pct=100,
        aoi_grid_inexact=False,
        processes=PROCESSES,
        threads=THREADS,
        chunk_size=CHUNK_SIZE,
        calc_buffers=False,
        include_parcel_geometry=False,
        save_features=False,
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
        api_key=None,
    ):
        # Assign parameters to instance variables
        self.aoi_file = aoi_file
        self.output_dir = output_dir
        self.packs = packs
        self.classes = classes
        self.primary_decision = primary_decision
        self.aoi_grid_min_pct = aoi_grid_min_pct
        self.aoi_grid_inexact = aoi_grid_inexact
        self.processes = processes
        self.threads = threads
        self.chunk_size = chunk_size
        self.calc_buffers = calc_buffers
        self.include_parcel_geometry = include_parcel_geometry
        self.save_features = save_features
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

        # Configure logger
        log.configure_logger(self.log_level)
        self.logger = log.get_logger()

    def api_key(self) -> str:
        if hasattr(self, 'api_key_param') and self.api_key_param is not None:
            return self.api_key_param
        return os.getenv("API_KEY")

    def process_chunk(self, chunk_id: str, aoi_gdf: gpd.GeoDataFrame, classes_df: pd.DataFrame):
        """
        Create a parcel rollup for a chunk of parcels.
        """
        feature_api = None
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
                self.logger.info(
                    f"Chunk {chunk_id} failed {len(errors_df)} of {len(aoi_gdf)} AOI requests. {len(rollup_df)} rollups returned on {len(rollup_df.index.unique())} unique {rollup_df.index.name}s."
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
                    max_allowed_error_pct=100,
                )
                self.logger.info(f"Chunk {chunk_id} failed {len(errors_df)} of {len(aoi_gdf)} AOI requests.")
                if len(errors_df) > 0:
                    if "message" in errors_df:
                        self.logger.debug(errors_df.value_counts("message"))
                    else:
                        self.logger.debug(f"Found {len(errors_df)} errors")
                if len(errors_df) == len(aoi_gdf):
                    errors_df.to_parquet(outfile_errors)
                    return

                # Filter features
                len_all_features = len(features_gdf)
                features_gdf = parcels.filter_features_in_parcels(features_gdf, aoi_gdf=aoi_gdf, region=self.country)
                len_filtered_features = len(features_gdf)

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
                + meta_data_columns
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
                final_df = final_df.convert_dtypes()
                if self.include_parcel_geometry:
                    final_df = gpd.GeoDataFrame(final_df, geometry="geometry", crs=API_CRS)
                final_df.to_parquet(outfile)
            except Exception as e:
                self.logger.error(f"Chunk {chunk_id}: Failed writing final_df ({len(final_df)} rows) to {outfile}.")
                self.logger.error(final_df.shape)
                self.logger.error(final_df)
                self.logger.error(e)
            if self.save_features and (self.endpoint != Endpoint.ROLLUP.value):
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
                        final_features_df.to_parquet(outfile_features)
                    except Exception as e:
                        self.logger.error(
                            f"Failed to save features parquet file for chunk_id {chunk_id}. Errors saved to {outfile_errors}. Rollup saved to {outfile}."
                        )
                        self.logger.error(e)
            self.logger.debug(f"Finished saving chunk {chunk_id}")
        finally:
            if feature_api:
                feature_api._sessions = []

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
                api_key=self.api_key(), alpha=self.alpha, beta=self.beta, prerelease=self.prerelease, only3d=self.only3d
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

        if outpath.exists() and (outpath_features.exists() or not self.save_features):
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
        with concurrent.futures.ProcessPoolExecutor(processes) as executor:
            try:
                for i, batch in enumerate(chunks):
                    # Use 'aoi_path' to construct chunk_id
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
                for j in jobs:
                    try:
                        j.result()
                    except Exception as e:
                        self.logger.error(f"FAILURE TO COMPLETE JOB {j}, DROPPING DUE TO ERROR {e}")
                        self.logger.error(f"{sys.exc_info()}\t{traceback.format_exc()}")
                        executor.shutdown(wait=False)
                        sys.exit(1)
            finally:
                executor.shutdown(wait=True)
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
            data = gpd.GeoDataFrame(pd.concat(data))
        else:
            data = pd.DataFrame(data)
        if len(data) > 0:
            if self.rollup_format == "parquet":
                data.to_parquet(outpath, index=True)
            elif self.rollup_format == "csv":
                if "geometry" in data.columns:
                    data["geometry"] = data.geometry.to_wkt()
                data.to_csv(outpath, index=True)
            else:
                self.logger.info("Invalid output format specified - reverting to csv")
                if "geometry" in data.columns:
                    data["geometry"] = data.geometry.to_wkt()
                data.to_csv(outpath, index=True)
        outpath_errors = final_path / f"{Path(aoi_path).stem}_errors.csv"
        self.logger.info(f"Saving error data as .csv to {outpath_errors}")
        for cp in chunk_path.glob(f"errors_{Path(aoi_path).stem}_*.parquet"):
            errors.append(pd.read_parquet(cp))
        if len(errors) > 0:
            errors = pd.concat(errors)
        else:
            errors = pd.DataFrame(errors)
        errors.to_csv(outpath_errors, index=True)
        if self.save_features:
            self.logger.info(f"Saving feature data as geoparquet to {outpath_features}")
            feature_paths = [p for p in chunk_path.glob(f"features_{Path(aoi_path).stem}_*.parquet")]
            for cp in tqdm(feature_paths, total=len(feature_paths)):
                try:
                    df_feature_chunk = gpd.read_parquet(cp)
                except Exception as e:
                    self.logger.error(f"Failed to read {cp}.")
                data_features.append(df_feature_chunk)
            if len(data_features) > 0:
                pd.concat(data_features).to_parquet(outpath_features)


def main():
    args = parse_arguments()
    exporter = AOIExporter(
        aoi_file=args.aoi_file,
        output_dir=args.output_dir,
        packs=args.packs,
        classes=args.classes,
        primary_decision=args.primary_decision,
        aoi_grid_min_pct=args.aoi_grid_min_pct,
        aoi_grid_inexact=args.aoi_grid_inexact,
        processes=args.processes,
        threads=args.threads,
        chunk_size=args.chunk_size,
        calc_buffers=args.calc_buffers,
        include_parcel_geometry=args.include_parcel_geometry,
        save_features=args.save_features,
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
        api_key=args.api_key,
    )
    exporter.run()

if __name__ == "__main__":
    main()
