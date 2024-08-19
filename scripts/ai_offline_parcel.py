import argparse
import concurrent.futures
import os
import random
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
    parser.add_argument("--parcel-dir", help="Directory with parcel files", type=str, required=True)
    parser.add_argument("--output-dir", help="Directory to store results", type=str, required=True)
    parser.add_argument(
        "--key-file",
        help="Path to file with API keys",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--config-file",
        help="Path to json file with config dictionary (min confidences, areas and ratios)",
        type=str,
        required=False,
        default=None,
    )
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
        "--workers",
        help="Number of processes",
        type=int,
        required=False,
        default=PROCESSES,
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
    parser.add_argument("--log-level", help="Log level (DEBUG, INFO, ...)", required=False, default="INFO", type=str)
    return parser.parse_args()


def api_key(key_file: Optional[str] = None) -> str:
    """
    Get an API key. If a key file is specified a random key from the file will be returned.
    """
    if key_file is None:
        return os.environ["API_KEY"]
    with open(key_file, "r") as f:
        keys = [line.rstrip() for line in f]
    return keys[random.randint(0, len(keys) - 1)]


def process_chunk(
    chunk_id: str,
    parcel_gdf: gpd.GeoDataFrame,
    classes_df: pd.DataFrame,
    output_dir: str,
    key_file: str,
    config: dict,
    country: str,
    packs: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    calc_buffers: Optional[bool] = False,
    include_parcel_geometry: Optional[bool] = False,
    save_features: Optional[bool] = True,
    primary_decision: str = "largest_intersection",
    overwrite_cache: Optional[bool] = False,
    compress_cache: Optional[bool] = False,
    no_cache: Optional[bool] = False,
    cache_dir: str = None,
    since_bulk: str = None,
    until_bulk: str = None,
    alpha: Optional[bool] = True,
    beta: Optional[bool] = True,
    prerelease: Optional[bool] = True,
    endpoint: [str] = Endpoint.FEATURE.value,
    url_root: Optional[str] = DEFAULT_URL_ROOT,
    system_version_prefix: Optional[str] = None,
    system_version: Optional[str] = None,
    threads: Optional[int] = THREADS,
):
    """
    Create a parcel rollup for a chuck of parcels.

    Args:
        chunk_id: Used to save data for the chunk
        parcel_gdf: Parcel set
        classes_df: Classes in output
        output_dir: Directory to save data to
        key_file: Path to API key file
        config: Dictionary of minimum areas and confidences.
        packs: AI packs to include. Defaults to all packs
        classes: List of feature class IDs (UUIDs) to include in the output.
        calc_buffers: Whether to calculate buffered features (compute expensive).
        include_parcel_geometry: Set to true to include parcel geometries in final output
        save_features: Whether to save the vectors for all features as a geospatial file.
        country: The country code for area calcs (au, us, ca, nz)
        primary_decision: The basis on which the primary feature is chosen (largest_intersection|nearest)
        compress_cache: Whether to use gzip compression (.json.gz) or save raw json text (.json).
        cache_dir: Place to store cache (absolute path of parent - "cache" and "rollup_cache" will be created within).
        since_bulk: Earliest date used to pull features
        until_bulk: Latest date used to pull features
        alpha: Return alpha layers
        beta: return beta layers
        prerelease: Return data from pre-release system versions
        endpoint: Which endpoint to use - feature|rollup. Uses either local geospatial ops, or relies on API logic.
        url_root: Overwrite the root URL with a custom one.
        system_version_prefix: Restrict responses to a specific system version generation.
        system_version: Restrict responses to a specific system version.
        threads: Number of threads to use for parallel processing.
    """
    if cache_dir is None and not no_cache:
        cache_dir = Path(output_dir)

    if not no_cache:
        cache_path = Path(cache_dir) / "cache"
    else:
        cache_path = None

    chunk_path = Path(output_dir) / "chunks"
    outfile = chunk_path / f"rollup_{chunk_id}.parquet"
    outfile_features = chunk_path / f"features_{chunk_id}.parquet"
    outfile_errors = chunk_path / f"errors_{chunk_id}.parquet"
    if outfile.exists():
        return

    # Get additional parcel attributes from parcel geometry
    if isinstance(parcel_gdf, gpd.GeoDataFrame):
        rep_point = parcel_gdf.representative_point()
        parcel_gdf["query_aoi_lat"] = rep_point.y
        parcel_gdf["query_aoi_lon"] = rep_point.x

    # Get features
    feature_api = FeatureApi(
        api_key=api_key(key_file),
        cache_dir=cache_path,
        overwrite_cache=overwrite_cache,
        compress_cache=compress_cache,
        workers=threads,
        alpha=alpha,
        beta=beta,
        prerelease=prerelease,
        url_root=url_root,
        system_version_prefix=system_version_prefix,
        system_version=system_version,
    )
    if endpoint == Endpoint.ROLLUP.value:
        logger.debug(f"Chunk {chunk_id}: Getting rollups for {len(parcel_gdf)} AOIs ({endpoint=})")
        rollup_df, metadata_df, errors_df = feature_api.get_rollup_df_bulk(
            parcel_gdf,
            region=country,
            since_bulk=since_bulk,
            until_bulk=until_bulk,
            packs=packs,
            classes=classes,
            instant_fail_batch=False,
        )
        logger.debug(f"Finished rollup for chunk {chunk_id} from feature endpoint.")
        rollup_df.columns = FeatureApi._multi_to_single_index(rollup_df.columns)
        logger.info(
            f"Chunk {chunk_id} failed {len(errors_df)} of {len(parcel_gdf)} AOI requests. {len(rollup_df)} rollups returned on {len(rollup_df[AOI_ID_COLUMN_NAME].unique())} unique {AOI_ID_COLUMN_NAME}s."
        )
        if len(errors_df) > 0:
            if "message" in errors_df:
                logger.debug(errors_df.value_counts("message"))
            else:
                logger.debug(f"Found {len(errors_df)} errors")
        if len(errors_df) == len(parcel_gdf):
            errors_df.to_parquet(outfile_errors)
            return
        logger.debug(f"Finished pulling rollup for chunk {chunk_id} from rollup endpoint.")
    elif endpoint == Endpoint.FEATURE.value:
        logger.debug(f"Chunk {chunk_id}: Getting features for {len(parcel_gdf)} AOIs({endpoint=})")
        features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(
            parcel_gdf,
            region=country,
            since_bulk=since_bulk,
            until_bulk=until_bulk,
            packs=packs,
            classes=classes,
            instant_fail_batch=False,
        )
        logger.info(f"Chunk {chunk_id} failed {len(errors_df)} of {len(parcel_gdf)} AOI requests.")
        if len(errors_df) > 0:
            if "message" in errors_df:
                logger.debug(errors_df.value_counts("message"))
            else:
                logger.debug(f"Found {len(errors_df)} errors")
        if len(errors_df) == len(parcel_gdf):
            errors_df.to_parquet(outfile_errors)
            return

        # Filter features
        len_all_features = len(features_gdf)
        features_gdf = parcels.filter_features_in_parcels(features_gdf, config=config, aoi_gdf=parcel_gdf, region=country)
        len_filtered_features = len(features_gdf)
        logger.debug(
            f"Chunk {chunk_id}:  Filtering removed {len_all_features-len_filtered_features} to leave {len_filtered_features} on {len(features_gdf[AOI_ID_COLUMN_NAME].unique())} unique {AOI_ID_COLUMN_NAME}s."
        )

        # Create rollup
        rollup_df = parcels.parcel_rollup(
            parcel_gdf,
            features_gdf,
            classes_df,
            country=country,
            calc_buffers=calc_buffers,
            primary_decision=primary_decision,
        )
    else:
        logger.error(f"Not a valid endpoint selection: {endpoint}")
        # End the program if the endpoint is not valid.
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
    ] # Some of these columns are not present in the Rollup API output
    # Validate that columns like survey_id and survey_resource_id in both the input file and API responses.
    for meta_data_column in meta_data_columns:
        if meta_data_column in parcel_gdf.columns:
            # Test they are identical in contents for non NaN values. If there is, rename the column with a prefix.
            metadata_df = metadata_df.rename(columns={meta_data_column: f"nmaipy_{meta_data_column}"})
            meta_data_columns.remove(meta_data_column)

    final_df = metadata_df.merge(rollup_df, on=AOI_ID_COLUMN_NAME).merge(parcel_gdf, on=AOI_ID_COLUMN_NAME)
    # Order the columns: parcel properties, meta data, data, parcel geometry.
    parcel_columns = [c for c in parcel_gdf.columns if c != "geometry"]
    columns = (
        parcel_columns
        + meta_data_columns
        + [c for c in final_df.columns if c not in parcel_columns + meta_data_columns + ["geometry"]]
    )
    if include_parcel_geometry:
        columns.append("geometry")
    # Filter out columns that are not in the final_df
    columns = [c for c in columns if c in final_df.columns]
    final_df = final_df[columns]
    date2str = lambda d: str(d).replace("-", "")
    make_link = (
        lambda d: f"https://apps.nearmap.com/maps/#/@{d.query_aoi_lat},{d.query_aoi_lon},21.00z,0d/V/{date2str(d.date)}?locationMarker"
    )
    if endpoint == Endpoint.ROLLUP.value:
        if "query_aoi_lat" in final_df.columns and "query_aoi_lon" in final_df.columns:
            final_df["link"] = final_df.apply(make_link, axis=1)
        final_df = final_df.drop(columns=["system_version", "date"])
    logger.debug(f"Chunk {chunk_id}: Writing {len(final_df)} rows for rollups and {len(errors_df)} for errors.")
    try:
        errors_df.to_parquet(outfile_errors)
    except Exception as e:
        logger.error(f"Chunk {chunk_id}: Failed writing errors_df ({len(errors_df)} rows) to {outfile_errors}.")
        logger.error(errors_df.shape)
        logger.error(errors_df)
    try:
        final_df = final_df.convert_dtypes()
        if include_parcel_geometry:
            final_df = gpd.GeoDataFrame(final_df, geometry="geometry", crs=API_CRS)
        final_df.to_parquet(outfile)
    except Exception as e:
        logger.error(f"Chunk {chunk_id}: Failed writing final_df ({len(final_df)} rows) to {outfile}.")
        logger.error(final_df.shape)
        logger.error(final_df)
        logger.error(e)

    if save_features and (endpoint != Endpoint.ROLLUP.value):
        # Save chunk's features as parquet, shift the parcel geometry to "aoi_geometry" if it exists.
        final_features_df = gpd.GeoDataFrame(
            metadata_df.merge(features_gdf, on=AOI_ID_COLUMN_NAME).merge(
                parcel_gdf.rename(columns=dict(geometry="aoi_geometry")), on=AOI_ID_COLUMN_NAME
            ),
            crs=API_CRS,
        )
        if "aoi_geometry" in final_features_df.columns:
            final_features_df["aoi_geometry"] = final_features_df.aoi_geometry.to_wkt()
        final_features_df["attributes"] = final_features_df.attributes.apply(json.dumps)
        if len(final_features_df) > 0:
            try:
                if not include_parcel_geometry and "aoi_geometry" in final_features_df.columns:
                    final_features_df = final_features_df.drop(columns=["aoi_geometry"])
                final_features_df = final_features_df[
                    ~(final_features_df.geometry.is_empty | final_features_df.geometry.isna())
                ]
                final_features_df.to_parquet(outfile_features)
            except Exception as e:
                logger.error(
                    f"Failed to save features parquet file for chunk_id {chunk_id}. Errors saved to {outfile_errors}. Rollup saved to {outfile}."
                )
                logger.error(e)
    logger.debug(f"Finished saving chunk {chunk_id}")


def main():
    args = parse_arguments()
    # Configure logger
    log.configure_logger(args.log_level)
    logger.debug("Starting parcel rollup")
    # Path setup
    parcel_paths = []
    for file_type in ["*.parquet", "*.csv", "*.psv", "*.tsv", "*.gpkg", "*.geojson"]:
        parcel_paths.extend(Path(args.parcel_dir).glob(file_type))
    parcel_paths.sort()
    logger.info(f"Running the following parcel files:")
    for parcel_path in parcel_paths:
        logger.info(f"\t{str(parcel_path)}")

    cache_path = Path(args.output_dir) / "cache"
    chunk_path = Path(args.output_dir) / "chunks"
    final_path = Path(args.output_dir) / "final"
    cache_path.mkdir(parents=True, exist_ok=True)
    chunk_path.mkdir(parents=True, exist_ok=True)
    final_path.mkdir(parents=True, exist_ok=True)

    # Get classes
    if args.packs is not None:
        classes_df = FeatureApi(api_key=api_key(args.key_file), alpha=args.alpha, beta=args.beta, prerelease=args.prerelease).get_feature_classes(
            args.packs
        )
    else:
        classes_df = FeatureApi(
            api_key=api_key(args.key_file), alpha=args.alpha, beta=args.beta, prerelease=args.prerelease
        ).get_feature_classes(args.packs)
        if args.classes is not None:
            # Remove classes in classes_df that are not in args.classes
            classes_df = classes_df[classes_df.index.isin(args.classes)]

    # Parse config
    if args.config_file is not None:
        # TODO: Add validation of the config file in future to strictly enforce valid feature class ids.
        with open(args.config_file, "r") as fp:
            config = json.load(fp)
    else:
        config = None

    # Loop over parcel files
    for f in parcel_paths:
        logger.info(f"Processing parcel file {f}")
        # If output exists, skip
        outpath = final_path / f"{f.stem}.{args.rollup_format}"
        outpath_features = final_path / f"{f.stem}_features.parquet"

        if outpath.exists() and (outpath_features.exists() or not args.save_features):
            logger.info(f"Output already exist, skipping {f.stem}")
            continue
        # Read parcel data
        parcels_gdf = parcels.read_from_file(f, id_column=AOI_ID_COLUMN_NAME)
        if isinstance(parcels_gdf, gpd.GeoDataFrame):
            parcels_gdf = parcels_gdf.to_crs(API_CRS)
        else:
            logger.info("No geometry found in parcel data - using address fields")
            for field in ADDRESS_FIELDS:
                if field not in parcels_gdf:
                    logger.error(f"Missing field {field} in parcel data")
                    sys.exit(1)

        # Print out info around what is being inferred from column names:
        if SURVEY_RESOURCE_ID_COL_NAME in parcels_gdf:
            logger.info(
                f"{SURVEY_RESOURCE_ID_COL_NAME} will be used to get results from the exact Survey Resource ID, instead of using date based filtering."
            )
        else:
            logger.info(f"No {SURVEY_RESOURCE_ID_COL_NAME} column provided, so date based endpoint will be used.")
            if SINCE_COL_NAME in parcels_gdf:
                logger.info(
                    f'The column "{SINCE_COL_NAME}" will be used as the earliest permitted date (YYYY-MM-DD) for each Query AOI.'
                )
            elif args.since is not None:
                logger.info(f"The since date of {args.since} will limit the earliest returned date for all Query AOIs")
            else:
                logger.info("No earliest date will used")
            if UNTIL_COL_NAME in parcels_gdf:
                logger.info(
                    f'The column "{UNTIL_COL_NAME}" will be used as the latest permitted date (YYYY-MM-DD) for each Query AOI.'
                )
            elif args.until is not None:
                logger.info(f"The until date of {args.until} will limit the latest returned date for all Query AOIs")
            else:
                logger.info("No latest date will used")

        if AOI_ID_COLUMN_NAME not in parcels_gdf:
            logger.warning(f"Missing {AOI_ID_COLUMN_NAME} column in parcel data - generating unique IDs")
            parcels_gdf[AOI_ID_COLUMN_NAME] = parcels_gdf.index

        jobs = []

        # Figure out how many chunks to divide the query AOI set into. Set 1 chunk as min.
        num_chunks = max(len(parcels_gdf) // CHUNK_SIZE, 1)
        logger.info(f"Exporting {len(parcels_gdf)} parcels as {num_chunks} chunk files.")
        logger.info(f"Using endpoint '{args.endpoint}' for rollups.")
        logger.debug(f"Splitting parcels into {num_chunks} chunks")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Geometry is in a geographic CRS.",
            )
            if isinstance(parcels_gdf, gpd.GeoDataFrame):
                parcels_gdf = parcels_gdf[parcels_gdf.area > 0]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'GeoDataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'GeoDataFrame.transpose' instead.",
            )
            warnings.filterwarnings(
                "ignore",
                message="'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead.",
            )
            chunks = np.array_split(parcels_gdf, num_chunks)
        if args.workers > 1:
            processes = int(args.workers)
            with concurrent.futures.ProcessPoolExecutor(processes) as executor:
                # Chunk parcels and send chunks to process pool
                for i, batch in enumerate(chunks):
                    chunk_id = f"{f.stem}_{str(i).zfill(4)}"
                    logger.debug(
                        (
                            f"Parallel processing chunk {chunk_id} - min {AOI_ID_COLUMN_NAME} is {batch[AOI_ID_COLUMN_NAME].min()}, max {AOI_ID_COLUMN_NAME} is {batch[AOI_ID_COLUMN_NAME].max()}"
                        )
                    )
                    jobs.append(
                        executor.submit(
                            process_chunk,
                            chunk_id,
                            batch,
                            classes_df,
                            args.output_dir,
                            args.key_file,
                            config,
                            args.country,
                            args.packs,
                            args.classes,
                            args.calc_buffers,
                            args.include_parcel_geometry,
                            args.save_features,
                            args.primary_decision,
                            args.overwrite_cache,
                            args.compress_cache,
                            args.no_cache,
                            args.cache_dir,
                            args.since,
                            args.until,
                            args.alpha,
                            args.beta,
                            args.prerelease,
                            args.endpoint,
                            args.url_root,
                            args.system_version_prefix,
                            args.system_version,
                        )
                    )
                for j in jobs:
                    try:
                        j.result()
                    except Exception as e:
                        logger.error(f"FAILURE TO COMPLETE JOB {j}, DROPPING DUE TO ERROR {e}")
                        logger.error(f"{sys.exc_info()}\t{traceback.format_exc()}")
                        # Shut down the rest of the jobs
                        executor.shutdown(wait=False)
                        sys.exit(1)
        else:
            # If we only have one worker, run in main process
            for i, batch in tqdm(enumerate(chunks), total=len(chunks)):
                chunk_id = f"{f.stem}_{str(i).zfill(4)}"
                logger.debug(
                    (
                        f"Processing chunk {chunk_id} - min {AOI_ID_COLUMN_NAME} is {batch[AOI_ID_COLUMN_NAME].min()}, max {AOI_ID_COLUMN_NAME} is {batch[AOI_ID_COLUMN_NAME].max()}"
                    )
                )
                process_chunk(
                    chunk_id,
                    batch,
                    classes_df,
                    args.output_dir,
                    args.key_file,
                    config,
                    args.country,
                    args.packs,
                    args.classes,
                    args.calc_buffers,
                    args.include_parcel_geometry,
                    args.save_features,
                    args.primary_decision,
                    args.overwrite_cache,
                    args.compress_cache,
                    args.no_cache,
                    args.cache_dir,
                    args.since,
                    args.until,
                    args.alpha,
                    args.beta,
                    args.prerelease,
                    args.endpoint,
                    args.url_root,
                    args.system_version_prefix,
                    args.system_version,
                )

        # Combine chunks and save
        data = []
        data_features = []
        errors = []
        # TODO: Add explicit check whether all chunks are found (some may have errored out). Currently fails silently and creates incomplete final files without further warning. List which chunks are missing.
        logger.info(f"Saving rollup data as {args.rollup_format} file to {outpath}")
        for i in range(num_chunks):  # Now attempt every chunk - so if one is missing, we error.
            chunk_filename = f"rollup_{f.stem}_{str(i).zfill(4)}.parquet"
            cp = chunk_path / chunk_filename
            if cp.exists():
                try:
                    chunk = gpd.read_parquet(cp)
                except ValueError:
                    # Probably because chunk doesn't have a geometry - no parcel boundary saved
                    chunk = pd.read_parquet(cp)
                if len(chunk) > 0:
                    data.append(chunk)
            else:
                error_filename = f"errors_{f.stem}_{str(i).zfill(4)}.parquet"
                if (chunk_path / error_filename).exists():
                    logger.debug(f"Chunk {i} rollup file missing, but error file found.")
                else:
                    logger.error(f"Chunk {i} rollup and error files missing. Try rerunning.")
                    # Break out of program to ensure we don't save a partially complete final csv.
                    sys.exit(1)

        if len(data) > 0:
            data = gpd.GeoDataFrame(pd.concat(data))
        else:
            data = pd.DataFrame(data)
        data = data.set_index(AOI_ID_COLUMN_NAME)
        if "index" in data.columns:
            data = data.drop(columns=["index"])
        if len(data) > 0:
            if args.rollup_format == "parquet":
                data.to_parquet(outpath, index=True)
            elif args.rollup_format == "csv":
                if "geometry" in data.columns:
                    data["geometry"] = data.geometry.to_wkt()
                data.to_csv(outpath, index=True)
            else:
                logger.info("Invalid output format specified - reverting to csv")
                data["geometry"] = data.geometry.to_wkt()
                data.to_csv(outpath, index=True)

        outpath_errors = final_path / f"{f.stem}_errors.csv"
        logger.info(f"Saving error data as .csv to {outpath_errors}")
        for cp in chunk_path.glob(f"errors_{f.stem}_*.parquet"):
            errors.append(pd.read_parquet(cp))
        if len(errors) > 0:
            errors = pd.concat(errors)
        else:
            errors = pd.DataFrame(errors)
        errors.to_csv(outpath_errors, index=True)

        if args.save_features:
            logger.info(f"Saving feature data as geoparquet to {outpath_features}")
            feature_paths = [p for p in chunk_path.glob(f"features_{f.stem}_*.parquet")]
            for cp in tqdm(feature_paths, total=len(feature_paths)):
                try:
                    df_feature_chunk = gpd.read_parquet(cp)
                except Exception as e:
                    logger.error(f"Failed to read {cp}.")
                data_features.append(df_feature_chunk)
            if len(data_features) > 0:
                pd.concat(data_features).to_parquet(outpath_features)


if __name__ == "__main__":
    main()
