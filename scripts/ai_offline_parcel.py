import argparse
import concurrent.futures
import os
import random
from pathlib import Path
from typing import List, Optional
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.wkt
from tqdm import tqdm

from nearmap_ai import log, parcels
from nearmap_ai.constants import AOI_ID_COLUMN_NAME, SINCE_COL_NAME, UNTIL_COL_NAME, API_CRS
from nearmap_ai.feature_api import FeatureApi

CHUNK_SIZE = 1000
PROCESSES = 20
THREADS = 4

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
        "--include-parcel-geometry",
        help="If set, parcel geometries will be in the output",
        action="store_true",
    )
    parser.add_argument(
        "--country",
        help="Country code for area calculations (au, us, ca, nz)",
        required=True,
    )
    parser.add_argument(
        "--bulk-mode",
        help="Use bulk mode API",
        required=False,
        type=bool,
        default=True,
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
    include_parcel_geometry: Optional[bool] = False,
    primary_decision: str = "largest_intersection",
    bulk_mode: Optional[bool] = True,
    since_bulk: str = None,
    until_bulk: str = None,
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
        include_parcel_geometry: Set to true to include parcel geometries in final output
        country: The country code for area calcs (au, us, ca, nz)
        primary_decision: The basis on which the primary feature is chosen (largest_intersection|nearest)
        since_bulk: Earliest date used to pull features
        until_bulk: LAtest date used to pull features
    """
    cache_path = Path(output_dir) / "cache"
    chunk_path = Path(output_dir) / "chunks"
    outfile = chunk_path / f"rollup_{chunk_id}.parquet"
    outfile_features = chunk_path / f"features_{chunk_id}.geojson"
    outfile_errors = chunk_path / f"errors_{chunk_id}.parquet"
    if outfile.exists():
        return

    # Get additional parcel attributes from parcel geometry
    parcel_gdf["query_aoi_lat"] = parcel_gdf.geometry.apply(lambda g: g.centroid.coords[0][1])
    parcel_gdf["query_aoi_lon"] = parcel_gdf.geometry.apply(lambda g: g.centroid.coords[0][0])

    # Get features
    feature_api = FeatureApi(api_key=api_key(key_file), bulk_mode=bulk_mode, cache_dir=cache_path, workers=THREADS)
    logger.debug(f"Chunk {chunk_id}: Getting features for {len(parcel_gdf)} AOIs")
    features_gdf, metadata_df, errors_df = feature_api.get_features_gdf_bulk(
        parcel_gdf, since_bulk=since_bulk, until_bulk=until_bulk, packs=packs
    )
    if errors_df is not None and parcel_gdf is not None and features_gdf is not None:
        logger.debug(
            f"Chunk {chunk_id} failed {len(errors_df)} of {len(parcel_gdf)} AOI requests. {len(features_gdf)} features returned."
        )
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
    features_gdf = parcels.filter_features_in_parcels(features_gdf, config=config)
    len_filtered_features = len(features_gdf)
    logger.debug(
        f"Chunk {chunk_id}:  Filtering removed {len_all_features-len_filtered_features} to leave {len_filtered_features}"
    )

    # Create rollup
    rollup_df = parcels.parcel_rollup(
        parcel_gdf, features_gdf, classes_df, country=country, primary_decision=primary_decision
    )

    # Put it all together and save
    final_df = metadata_df.merge(rollup_df, on=AOI_ID_COLUMN_NAME).merge(parcel_gdf, on=AOI_ID_COLUMN_NAME)

    # Order the columns: parcel properties, meta data, data, parcel geometry.
    parcel_columns = [c for c in parcel_gdf.columns if c != "geometry"]
    meta_data_columns = ["system_version", "link", "date"]
    columns = (
        parcel_columns
        + meta_data_columns
        + [c for c in final_df.columns if c not in parcel_columns + meta_data_columns + ["geometry"]]
    )
    if include_parcel_geometry:
        columns.append("geometry")
        final_df["geometry"] = final_df.geometry.apply(shapely.wkt.dumps)
    final_df = final_df[columns]

    logger.debug(f"Chunk {chunk_id}: Writing {len(final_df)} rows for rollups and {len(errors_df)} for errors.")
    errors_df.to_parquet(outfile_errors)
    final_df.to_parquet(outfile)

    # Save features as geojson, shift the parcel geometry to "aoi_geometry"
    final_features_df = gpd.GeoDataFrame(
        metadata_df.merge(features_gdf, on=AOI_ID_COLUMN_NAME).merge(
            parcel_gdf.rename(columns=dict(geometry="aoi_geometry")), on=AOI_ID_COLUMN_NAME
        ),
        crs=API_CRS,
    )
    final_features_df["aoi_geometry"] = final_features_df.aoi_geometry.apply(lambda d: d.wkt)
    final_features_df["attributes"] = final_features_df.attributes.apply(json.dumps)
    if len(final_features_df) > 0:
        try:
            if not include_parcel_geometry:
                final_features_df = final_features_df.drop(columns=["aoi_geometry"])
            final_features_df = final_features_df[
                ~(final_features_df.geometry.is_empty | final_features_df.geometry.isna())
            ]
            final_features_df.to_file(outfile_features, driver="GeoJSON")
        except Exception:
            logger.error(
                f"Failed to save features geojson for chunk_id {chunk_id}. Errors saved to {outfile_errors}. Rollup saved to {outfile}."
            )


def main():
    args = parse_arguments()
    # Configure logger
    log.configure_logger(args.log_level)
    logger.info("Starting parcel rollup")
    # Path setup
    parcel_paths = []
    for file_type in ["*.parquet", "*.csv", "*.psv", "*.tsv", "*.gpkg", "*.geojson"]:
        parcel_paths.extend(Path(args.parcel_dir).glob(file_type))
    parcel_paths.sort()
    logger.info(f"Running the following parcel files: {parcel_paths}")

    cache_path = Path(args.output_dir) / "cache"
    chunk_path = Path(args.output_dir) / "chunks"
    final_path = Path(args.output_dir) / "final"
    cache_path.mkdir(parents=True, exist_ok=True)
    chunk_path.mkdir(parents=True, exist_ok=True)
    final_path.mkdir(parents=True, exist_ok=True)

    # Get classes
    classes_df = FeatureApi(api_key=api_key(args.key_file)).get_feature_classes(args.packs)

    # Parse config
    if args.config_file is not None:
        #TODO: Add validation of the config file in future to strictly enforce valid feature class ids.
        with open(args.config_file, "r") as fp:
            config = json.load(fp)
    else:
        config = None

    # Loop over parcel files
    for f in parcel_paths:
        logger.info(f"Processing parcel file {f}")
        # If output exists, skip
        outpath = final_path / f"{f.stem}.csv"
        if outpath.exists():
            logger.info(f"Output already exist, skipping ({outpath})")
            continue
        outpath_features = final_path / f"{f.stem}_features.geojson"
        if outpath_features.exists():
            logger.info(f"Output already exist, skipping ({outpath_features})")
            continue
        # Read parcel data
        parcels_gdf = parcels.read_from_file(f).to_crs(API_CRS)

        logger.info(f"Exporting {len(parcels_gdf)} parcels.")

        # Print out info around what is being inferred from column names:
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

        jobs = []

        # Figure out how many chunks to divide the query AOI set into. Set 1 chunk as min.
        num_chunks = max(len(parcels_gdf) // CHUNK_SIZE, 1)

        if args.workers > 1:
            with concurrent.futures.ProcessPoolExecutor(PROCESSES) as executor:
                # Chunk parcels and send chunks to process pool
                for i, batch in enumerate(np.array_split(parcels_gdf, num_chunks)):
                    chunk_id = f"{f.stem}_{str(i).zfill(4)}"
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
                            args.include_parcel_geometry,
                            args.primary_decision,
                            args.bulk_mode,
                            args.since,
                            args.until,
                        )
                    )
                [j.result() for j in tqdm(jobs)]
        else:
            # If we only have one worker, run in main process
            for i, batch in tqdm(enumerate(np.array_split(parcels_gdf, num_chunks))):
                chunk_id = f"{f.stem}_{str(i).zfill(4)}"
                process_chunk(
                    chunk_id,
                    batch,
                    classes_df,
                    args.output_dir,
                    args.key_file,
                    config,
                    args.country,
                    args.packs,
                    args.include_parcel_geometry,
                    args.primary_decision,
                    args.bulk_mode,
                    args.since,
                    args.until,
                )

        # Combine chunks and save
        data = []
        data_features = []
        errors = []
        for cp in chunk_path.glob(f"rollup_{f.stem}_*.parquet"):
            data.append(pd.read_parquet(cp))
        pd.concat(data).to_csv(outpath, index=True)
        for cp in chunk_path.glob(f"features_{f.stem}_*.geojson"):
            data_features.append(gpd.read_file(cp))
        if len(data_features) > 0:
            pd.concat(data_features).to_file(outpath_features, driver="GeoJSON")
        for cp in chunk_path.glob(f"errors_{f.stem}_*.parquet"):
            errors.append(pd.read_parquet(cp))
        pd.concat(errors).to_csv(final_path / f"{f.stem}_errors.csv", index=True)
        logger.info(f"Save data to {outpath}")


if __name__ == "__main__":
    main()
