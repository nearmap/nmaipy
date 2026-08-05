"""
Damage Conflation Data Exporter

Command-line tool to export event-scoped, building-level conflated damage from the
Nearmap Damage Conflation API (ai/damage/v2) for multiple areas of interest (AOIs).

Given an ``--event-id`` and an AOI file, this exporter:
- Queries the conflation API per-AOI in parallel (pagination handles large AOIs)
- Always emits per-building damage polygons with flattened attributes
- Optionally (``--rollup``) emits a per-AOI rollup (one row per AOI: the input file's
  own columns, then rating counts + the primary building's attributes)
- Optionally (``--parcel-mode``) drops buildings that barely intersect their AOI,
  approximating the Feature API's parcelMode attribution client-side
- Caches API responses, tracks progress, and reports errors

Example usage:
    python -m nmaipy.damage_conflation_exporter \\
        --aoi-file parcels.geojson \\
        --output-dir data/damage_output \\
        --event-id 2f510853-5d55-50f4-9102-2c02de08190e \\
        --country us \\
        --processes 4 \\
        --rollup
"""

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd

from nmaipy import log, parcels, storage
from nmaipy.__version__ import __version__
from nmaipy.api_common import (
    combine_chunk_latency_stats,
    compute_global_latency_stats,
    format_error_summary_table,
    log_request_cache_summary,
    sanitize_error_message,
    save_chunk_latency_stats,
)
from nmaipy.base_exporter import BaseExporter
from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    API_CRS,
    API_WARMUP_INTERVAL_SECONDS,
    AREA_CRS,
    wrong_unit_area_columns,
)
from nmaipy.damage_conflation_api import DamageConflationApi
from nmaipy.reference_code import BUILDING_SMALL_MAX_AREA_SQM

logger = log.get_logger()

# Parcel-mode filter: a building is kept when at least this fraction of it lies
# inside the AOI, or when its in-AOI footprint is at least the small-building
# area. The area floor protects multi-parcel structures (e.g. a townhouse row is
# only ~25% on each parcel but has a substantial footprint on every one of them).
PARCEL_MODE_MIN_RATIO = 0.5
PARCEL_MODE_MIN_INTERSECTION_SQM = BUILDING_SMALL_MAX_AREA_SQM

# Ordered, curated columns for the per-building CSV export. The verbose JSON
# classRatios columns are kept in the geoparquet but omitted here for readability.
# Area column (area_sqm / area_sqft) is added per-country at export time.
_BUILDINGS_CSV_BASE_FIELDS = (
    AOI_ID_COLUMN_NAME,
    "feature_id",
    "hilbert_id",
    "damage_event_rating",
    "damage_event_confidence",
    "damage_event_raw_affected",
    "damage_event_raw_destroyed",
    "damage_event_raw_major",
    "damage_event_raw_minor",
    "damage_event_raw_no_damage",
    "damage_event_latest_capture_date",
    "damage_pre_event_rating",
    "damage_pre_event_confidence",
    "damage_pre_event_raw_affected",
    "damage_pre_event_raw_destroyed",
    "damage_pre_event_raw_major",
    "damage_pre_event_raw_minor",
    "damage_pre_event_raw_no_damage",
    "damage_pre_event_latest_capture_date",
    "confidence",
    "event_uuid",
    "event_name",
    "model_version",
    "presentation_version",
    "resource_id",
    "geomatched_address",
)


def filter_buildings_to_aoi(
    features_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    country: str,
    min_ratio: float = PARCEL_MODE_MIN_RATIO,
    min_intersection_sqm: float = PARCEL_MODE_MIN_INTERSECTION_SQM,
) -> gpd.GeoDataFrame:
    """Drop buildings that barely intersect their AOI (parcel-mode approximation).

    The conflation API is map-style: every building intersecting the AOI comes back,
    including neighbours' buildings that only clip the boundary. This is a client-side
    approximation of the Feature API's ``parcelMode`` attribution: a building is kept
    when at least ``min_ratio`` of its area lies inside the AOI, or when its in-AOI
    footprint is at least ``min_intersection_sqm`` (the floor keeps multi-parcel
    structures such as townhouse rows, which fail the ratio test on every parcel they
    span). Unlike parcelMode, nothing is clipped — kept buildings stay whole.

    Rows whose AOI has no geometry (address-mode input) or whose overlap cannot be
    computed are kept, never silently dropped.
    """
    if len(features_gdf) == 0 or "geometry" not in aoi_gdf.columns:
        return features_gdf

    crs = AREA_CRS.get(country.lower(), AREA_CRS["us"])
    buildings = features_gdf.geometry.to_crs(crs)
    aois = aoi_gdf.geometry.to_crs(crs)
    for series in (buildings, aois):
        invalid = ~series.is_valid
        if invalid.any():
            series[invalid] = series[invalid].make_valid()

    aoi_per_building = gpd.GeoSeries(features_gdf[AOI_ID_COLUMN_NAME].map(aois).values, index=buildings.index, crs=crs)
    intersection_sqm = buildings.intersection(aoi_per_building).area
    ratio = intersection_sqm / buildings.area
    keep = (ratio >= min_ratio) | (intersection_sqm >= min_intersection_sqm) | ratio.isna()
    return features_gdf[keep.values]


def parse_arguments():
    """Parse command line arguments for the damage conflation exporter."""
    parser = argparse.ArgumentParser(
        prog="nmaipy.damage_conflation_exporter",
        description="Export event-scoped conflated damage from the Nearmap Damage Conflation API",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--aoi-file",
        help="Input AOI file path (GeoJSON, Shapefile, GeoPackage, or CSV with WKT geometry)",
        type=str,
        required=True,
    )
    parser.add_argument("--output-dir", help="Directory to store results", type=str, required=True)
    parser.add_argument(
        "--event-id",
        help="UUID of the catastrophe event to query (from the Coverage API eventId tag)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-format",
        help="Output format: geoparquet (default), csv, or both",
        type=str,
        choices=["geoparquet", "csv", "both"],
        default="geoparquet",
    )
    parser.add_argument(
        "--rollup",
        help="Also emit a per-AOI rollup (one row per AOI: your input columns, rating counts + "
        "primary building). Most useful when AOIs are property-sized.",
        action="store_true",
    )
    parser.add_argument(
        "--parcel-mode",
        help="Treat each AOI as a parcel boundary and drop buildings that barely intersect it "
        f"(kept when >={PARCEL_MODE_MIN_RATIO:.0%} inside the AOI or the in-AOI footprint is "
        f">={PARCEL_MODE_MIN_INTERSECTION_SQM} sqm). Client-side approximation of the Feature "
        "API's parcelMode; nothing is clipped.",
        action="store_true",
    )
    parser.add_argument(
        "--primary-decision",
        help="Primary-building selection for the rollup: 'largest' (default) or 'optimal' "
        "(prefers the building containing the AOI's lat/lon, requires lat/lon columns).",
        type=str,
        choices=["largest", "optimal"],
        default="largest",
    )
    parser.add_argument("--cache-dir", help="Location to store cache (defaults to output-dir/cache)", type=str)
    parser.add_argument("--no-cache", help="Disable caching", action="store_true")
    parser.add_argument("--overwrite-cache", help="Overwrite existing cache files", action="store_true")
    parser.add_argument("--compress-cache", help="Use gzip compression for cache files", action="store_true")
    parser.add_argument(
        "--processes", help="Number of processes for parallel chunk processing (default: 4)", type=int, default=4
    )
    parser.add_argument(
        "--threads", help="Number of concurrent API requests within each process (default: 10)", type=int, default=10
    )
    parser.add_argument(
        "--chunk-size", help="Number of AOIs to process in a single chunk (default: 500)", type=int, default=500
    )
    parser.add_argument(
        "--api-warmup-interval",
        help=(
            f"Seconds between adding each parallel worker during API warmup "
            f"(default: {API_WARMUP_INTERVAL_SECONDS}). Set to 0 to disable warmup."
        ),
        type=float,
        default=API_WARMUP_INTERVAL_SECONDS,
    )
    parser.add_argument("--country", help="Country code (drives output area units)", type=str, default="us")
    parser.add_argument("--api-key", help="API key (overrides API_KEY environment variable)", type=str)
    parser.add_argument(
        "--log-level",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--include-aoi-geometry", help="Include original AOI geometry in output", action="store_true")
    return parser.parse_args()


class DamageConflationExporter(BaseExporter):
    """
    Exporter for bulk Damage Conflation data retrieval.

    Inherits chunking and parallel processing from BaseExporter. Per-building output
    is always produced; the per-AOI rollup is opt-in via ``rollup=True``.
    """

    def __init__(
        self,
        aoi_file: str,
        output_dir: str,
        event_id: str,
        output_format: str = "geoparquet",
        rollup: bool = False,
        parcel_mode: bool = False,
        primary_decision: str = "largest",
        cache_dir: str = None,
        no_cache: bool = False,
        overwrite_cache: bool = False,
        compress_cache: bool = False,
        processes: int = 4,
        threads: int = 10,
        chunk_size: int = 500,
        country: str = "us",
        api_key: str = None,
        log_level: str = "INFO",
        include_aoi_geometry: bool = False,
        api_warmup_interval_seconds: float = API_WARMUP_INTERVAL_SECONDS,
    ):
        super().__init__(
            output_dir=output_dir,
            processes=processes,
            chunk_size=chunk_size,
            log_level=log_level,
            api_warmup_interval_seconds=api_warmup_interval_seconds,
        )

        if not event_id:
            raise ValueError("event_id is required")

        self.aoi_file = aoi_file
        self.event_id = event_id
        self.output_format = output_format
        self.rollup = rollup
        self.parcel_mode = parcel_mode
        self.primary_decision = primary_decision
        self.cache_dir = str(cache_dir) if cache_dir else storage.join_path(self.output_dir, "cache")
        self.no_cache = no_cache
        self.overwrite_cache = overwrite_cache
        self.compress_cache = compress_cache
        self.threads = threads
        self.country = country
        self.api_key = api_key
        self.include_aoi_geometry = include_aoi_geometry

        if not self.no_cache:
            if storage.is_s3_path(self.cache_dir):
                self.logger.warning(
                    "API cache will be written to S3, which may be slow due to many small files. "
                    "Consider using --cache-dir to set a local cache directory, or --no-cache to disable caching."
                )
            storage.ensure_directory(self.cache_dir)

        self._save_config(
            {
                "aoi_file": str(aoi_file),
                "event_id": event_id,
                "output_format": output_format,
                "rollup": rollup,
                "parcel_mode": parcel_mode,
                "primary_decision": primary_decision,
                "cache_dir": str(self.cache_dir),
                "no_cache": no_cache,
                "overwrite_cache": overwrite_cache,
                "compress_cache": compress_cache,
                "processes": processes,
                "threads": threads,
                "chunk_size": chunk_size,
                "country": country,
                "include_aoi_geometry": include_aoi_geometry,
            },
            config_name="damage_conflation_export_config.json",
        )

    def get_chunk_output_file(self, chunk_id: str) -> str:
        """Path to the chunk's metadata file (used for cache checking)."""
        return storage.join_path(self.chunk_path, f"metadata_{chunk_id}.parquet")

    def process_chunk(self, chunk_id: str, aoi_gdf: gpd.GeoDataFrame, **kwargs):
        """Process a chunk of AOIs: query the conflation API and write chunk files."""
        BaseExporter.configure_worker_logging(self.log_level)
        logger = log.get_logger()

        chunk_start_time = datetime.now(timezone.utc).isoformat()
        chunk_start_monotonic = time.monotonic()

        try:
            storage.ensure_directory(self.chunk_path)

            outfile_features = storage.join_path(self.chunk_path, f"damage_features_{chunk_id}.parquet")
            outfile_metadata = storage.join_path(self.chunk_path, f"metadata_{chunk_id}.parquet")
            outfile_errors = storage.join_path(self.chunk_path, f"damage_errors_{chunk_id}.parquet")

            if storage.file_exists(outfile_metadata) and storage.validate_parquet(outfile_metadata):
                logger.debug(f"Chunk {chunk_id} already processed, skipping")
                return

            logger.debug(f"Chunk {chunk_id}: Processing {len(aoi_gdf)} AOIs")
            progress_counters = kwargs.get("progress_counters")

            cache_path = None if self.no_cache else self.cache_dir
            api = DamageConflationApi(
                event_id=self.event_id,
                api_key=self.api_key,
                cache_dir=cache_path,
                overwrite_cache=self.overwrite_cache,
                compress_cache=self.compress_cache,
                threads=self.threads,
                country=self.country,
                progress_counters=progress_counters,
            )

            features_gdf, metadata_df, errors_df = api.get_damage_bulk(aoi_gdf)

            logger.debug(
                f"Chunk {chunk_id}: Found {len(features_gdf)} buildings, "
                f"{len(metadata_df)} successful queries, {len(errors_df)} errors"
            )

            if len(features_gdf) > 0:
                storage.write_parquet(features_gdf, outfile_features)
            if len(metadata_df) > 0:
                storage.write_parquet(metadata_df, outfile_metadata)
            if len(errors_df) > 0:
                storage.write_parquet(errors_df, outfile_errors)

            # The rollup is intentionally NOT computed here. It is derived in the combine
            # step (_run_inner) from the fully-combined features, so it stays correct even
            # when --rollup is enabled on a resumed run whose chunks are already cached
            # (a per-chunk rollup would be skipped along with the cached chunk).

            latency_stats = api.get_latency_stats()
            if latency_stats is not None:
                latency_stats["chunk_id"] = chunk_id
                latency_stats["start_time"] = chunk_start_time
                latency_stats["end_time"] = datetime.now(timezone.utc).isoformat()
                latency_stats["total_duration_ms"] = (time.monotonic() - chunk_start_monotonic) * 1000
                save_chunk_latency_stats(latency_stats, self.chunk_path, chunk_id)

            api.cleanup()
            return {"chunk_id": chunk_id, "latency_stats": latency_stats}

        except Exception as e:
            logger.error(f"Chunk {chunk_id} failed: {e}")
            traceback.print_exc()
            raise

    def run(self):
        """Execute the damage conflation export workflow."""
        try:
            self._run_inner()
        finally:
            self._cleanup_staging()

    def _run_inner(self):
        self.logger.info(f"nmaipy version: {__version__}")
        self.logger.info(f"Starting damage conflation export from {self.aoi_file}")
        self.logger.info(f"Event: {self.event_id}")
        self.logger.info(f"Output directory: {self.output_dir}")

        self.logger.info("Loading AOI file...")
        aoi_gdf = parcels.read_from_file(self.aoi_file, id_column=AOI_ID_COLUMN_NAME)

        if "geometry" in aoi_gdf.columns and aoi_gdf.crs != API_CRS:
            self.logger.info(f"Reprojecting from {aoi_gdf.crs} to {API_CRS}")
            aoi_gdf = aoi_gdf.to_crs(API_CRS)

        self.logger.info(f"Loaded {len(aoi_gdf)} AOIs")

        chunks_to_process, skipped_chunks, skipped_aois, num_chunks = self.split_into_chunks(aoi_gdf, check_cache=True)
        initial_aoi_count = len(aoi_gdf) - skipped_aois
        latency_csv_path = storage.join_path(self.final_path, "latency_stats.csv")

        self.run_parallel(chunks_to_process, initial_aoi_count=initial_aoi_count, use_progress_tracking=True)

        # The spawned workers wrote chunk files to chunk_path; drop any cached s3fs listing
        # so the latency combine (and the chunk consolidation below) read S3 fresh.
        storage.invalidate_cache(self.chunk_path)

        all_latency_stats = combine_chunk_latency_stats(self.chunk_path, latency_csv_path)
        if all_latency_stats:
            global_stats = compute_global_latency_stats(all_latency_stats)
            if global_stats and global_stats.get("count", 0) > 0:
                self.logger.info(
                    f"Global latency stats: mean={global_stats['mean']:.0f}ms, "
                    f"P50={global_stats['p50']:.0f}ms, P90={global_stats['p90']:.0f}ms, "
                    f"P95={global_stats['p95']:.0f}ms, P99={global_stats['p99']:.0f}ms, n={global_stats['count']}"
                )
            log_request_cache_summary(self.logger, all_latency_stats)

        self.logger.info("Combining chunk results...")
        features_gdf = self.combine_chunk_files("damage_features", num_chunks, geo=True)
        metadata_df = self.combine_chunk_files("metadata", num_chunks)
        errors_df = self.combine_chunk_files("damage_errors", num_chunks)

        success_count = len(metadata_df)
        error_count = len(errors_df)
        self.logger.info(
            f"API queries complete: {success_count} successful, {error_count} errors, "
            f"{len(features_gdf)} total buildings found"
        )

        # Guard against silent empty consolidation: with chunks to read, all three combined
        # frames being empty means the sweep found nothing (e.g. a stale listing or a path
        # mismatch) rather than a genuinely empty export. Surface it loudly instead of
        # writing an all-empty rollup that looks like "no damage anywhere".
        if num_chunks > 0 and success_count == 0 and error_count == 0 and len(features_gdf) == 0:
            self.logger.warning(
                f"Consolidation read 0 records from {num_chunks} chunk(s) under {self.chunk_path}. "
                "Per-chunk files may be present but unreadable at combine time (stale listing / "
                "path mismatch). The rollup will be empty — investigate before trusting this output."
            )

        if error_count > 0:
            status_counts = errors_df["status_code"].value_counts() if "status_code" in errors_df.columns else None
            message_counts = None
            if "message" in errors_df.columns:
                message_counts = errors_df["message"].apply(sanitize_error_message).value_counts()
            error_table = format_error_summary_table(status_counts, message_counts)
            self.logger.info(f"Damage Conflation API: {error_count} failures{error_table}")

        # Filter before the rollup so counts and primary selection only see kept
        # buildings. Runs at combine time (not per-chunk) so cached chunks stay raw
        # and the flag can change on a resumed run.
        if self.parcel_mode and len(features_gdf) > 0:
            before = len(features_gdf)
            features_gdf = filter_buildings_to_aoi(features_gdf, aoi_gdf, self.country)
            self.logger.info(
                f"Parcel mode: dropped {before - len(features_gdf)} of {before} buildings "
                "that barely intersect their AOI"
            )

        # Derive the rollup from the fully-combined features (not per-chunk) so it is
        # always correct, including on a resumed run with cached chunks. Compute before
        # the include_aoi_geometry merge so the rollup is built from pristine features.
        rollup_df = pd.DataFrame()
        if self.rollup:
            rollup_df = parcels.conflation_rollup(
                aoi_gdf,
                features_gdf,
                country=self.country,
                successful_aoi_ids=set(metadata_df.index),
                primary_decision=self.primary_decision,
            )
            if len(rollup_df) > 0:
                # Lead with the input file's own columns (mirrors AOIExporter's rollup)
                # so customer-supplied columns survive into the output. On a name
                # collision the rollup schema wins; the input column gets `_input`.
                passthrough = pd.DataFrame(aoi_gdf).drop(columns=["geometry"], errors="ignore")
                rollup_df = passthrough.join(rollup_df, how="right", lsuffix="_input")

        if self.include_aoi_geometry and len(features_gdf) > 0:
            self.logger.info("Merging building data with AOI attributes...")
            aoi_for_merge = aoi_gdf.rename(columns={"geometry": "aoi_geometry"})
            features_gdf = features_gdf.merge(aoi_for_merge, left_on=AOI_ID_COLUMN_NAME, right_index=True, how="left")
            if "aoi_geometry" in features_gdf.columns:
                features_gdf["aoi_geometry"] = features_gdf["aoi_geometry"].apply(
                    lambda g: g.wkt if g is not None else None
                )

        self._save_outputs(features_gdf, rollup_df, metadata_df, errors_df, self.final_path)
        self.logger.info("Export complete!")

    def _save_outputs(self, features_gdf, rollup_df, metadata_df, errors_df, output_path):
        """Write the per-building output (always) and the per-AOI rollup (when enabled)."""
        if len(features_gdf) > 0:
            # Keep only the country-correct area family (drop e.g. area_sqm for US).
            drop_cols = [c for c in wrong_unit_area_columns(self.country) if c in features_gdf.columns]
            features_gdf = features_gdf.drop(columns=drop_cols)

            if self.output_format in ["geoparquet", "both"]:
                path = storage.join_path(output_path, "damage_buildings.parquet")
                self.logger.info(f"Saving {len(features_gdf)} buildings to {path}")
                storage.write_parquet(features_gdf, path, index=False)

            if self.output_format in ["csv", "both"]:
                path = storage.join_path(output_path, "damage_buildings.csv")
                self.logger.info(f"Saving {len(features_gdf)} buildings to {path}")
                df = pd.DataFrame(features_gdf)
                if "geometry" in df.columns:
                    df["geometry"] = df["geometry"].apply(lambda g: g.wkt if g is not None else None)
                area_col = "area_sqft" if "area_sqft" in df.columns else "area_sqm"
                fields = list(_BUILDINGS_CSV_BASE_FIELDS)
                fields.insert(fields.index("confidence"), area_col)
                fields.append("geometry")
                if "aoi_geometry" in df.columns:
                    fields.append("aoi_geometry")
                df[[c for c in fields if c in df.columns]].to_csv(path, index=False)
        else:
            self.logger.warning("No building data to save")

        if self.rollup and len(rollup_df) > 0:
            if self.output_format in ["geoparquet", "both"]:
                path = storage.join_path(output_path, "damage_rollup.parquet")
                self.logger.info(f"Saving rollup ({len(rollup_df)} AOIs) to {path}")
                storage.write_parquet(rollup_df, path, index=True)
            if self.output_format in ["csv", "both"]:
                path = storage.join_path(output_path, "damage_rollup.csv")
                self.logger.info(f"Saving rollup ({len(rollup_df)} AOIs) to {path}")
                rollup_df.to_csv(path, index=True)

        if len(metadata_df) > 0:
            metadata_df.to_csv(storage.join_path(output_path, "damage_metadata.csv"), index=True)
        # Consolidate per-chunk errors into final/ (status_code + message) so downstream
        # validation and customers can see failures — not just the log-only summary above.
        if len(errors_df) > 0:
            errors_path = storage.join_path(output_path, "damage_errors.csv")
            self.logger.info(f"Saving {len(errors_df)} errors to {errors_path}")
            errors_df.to_csv(errors_path, index=True)


def main():
    """Main entry point for the damage conflation exporter CLI."""
    args = parse_arguments()
    try:
        exporter = DamageConflationExporter(
            aoi_file=args.aoi_file,
            output_dir=args.output_dir,
            event_id=args.event_id,
            output_format=args.output_format,
            rollup=args.rollup,
            parcel_mode=args.parcel_mode,
            primary_decision=args.primary_decision,
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
            overwrite_cache=args.overwrite_cache,
            compress_cache=args.compress_cache,
            processes=args.processes,
            threads=args.threads,
            chunk_size=args.chunk_size,
            country=args.country,
            api_key=args.api_key,
            log_level=args.log_level,
            include_aoi_geometry=args.include_aoi_geometry,
            api_warmup_interval_seconds=args.api_warmup_interval,
        )
        exporter.run()
    except Exception as e:
        logger.error(f"Export failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
