"""
Roof Age Data Exporter

Command-line tool to export roof age data from the Nearmap Roof Age API for
multiple areas of interest (AOIs).

This exporter provides:
- Bulk parallel processing of multiple AOIs
- Progress tracking
- Error handling and reporting
- Output in GeoParquet and CSV formats
- Caching support to avoid redundant API calls

Example usage:
    python -m nmaipy.roof_age_exporter \\
        --aoi-file parcels.geojson \\
        --output-dir data/roof_age_output \\
        --country us \\
        --processes 4

Note: The Roof Age API is currently available for US properties only.
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from nmaipy import log, parcels
from nmaipy.__version__ import __version__
from nmaipy.api_common import format_error_summary_table, sanitize_error_message
from nmaipy.base_exporter import BaseExporter
from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS
from nmaipy.roof_age_api import RoofAgeApi

logger = log.get_logger()


def parse_arguments():
    """Parse command line arguments for roof age exporter"""
    parser = argparse.ArgumentParser(
        prog="nmaipy.roof_age_exporter",
        description="Export roof age data from Nearmap Roof Age API",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--aoi-file",
        help="Input AOI file path (GeoJSON, Shapefile, GeoPackage, or CSV with WKT geometry)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to store results",
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
        "--cache-dir",
        help="Location to store cache (defaults to output-dir/cache)",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--no-cache",
        help="Disable caching",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite-cache",
        help="Overwrite existing cache files",
        action="store_true",
    )
    parser.add_argument(
        "--compress-cache",
        help="Use gzip compression for cache files",
        action="store_true",
    )
    parser.add_argument(
        "--processes",
        help="Number of processes for parallel chunk processing (default: 4)",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--threads",
        help="Number of concurrent API requests within each process (default: 10)",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--chunk-size",
        help="Number of AOIs to process in a single chunk (default: 500)",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--country",
        help="Country code (must be 'us' for Roof Age API)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--api-key",
        help="API key (overrides API_KEY environment variable)",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--log-level",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--include-aoi-geometry",
        help="Include original AOI geometry in output",
        action="store_true",
    )
    return parser.parse_args()


class RoofAgeExporter(BaseExporter):
    """
    Exporter for bulk roof age data retrieval.

    Handles parallel processing, progress tracking, caching, and output generation.
    Inherits chunking and parallel processing infrastructure from BaseExporter.
    """

    def __init__(
        self,
        aoi_file: str,
        output_dir: str,
        output_format: str = "geoparquet",
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
    ):
        """
        Initialize RoofAgeExporter.

        Args:
            aoi_file: Path to input AOI file
            output_dir: Directory for output files
            output_format: Output format (geoparquet, csv, or both)
            cache_dir: Cache directory (defaults to output_dir/cache)
            no_cache: Disable caching
            overwrite_cache: Overwrite existing cache
            compress_cache: Use gzip compression for cache
            processes: Number of processes for parallel chunk processing
            threads: Number of concurrent API requests within each process
            chunk_size: Number of AOIs to process in a single chunk
            country: Country code (must be 'us')
            api_key: API key (optional, uses environment variable if not provided)
            log_level: Logging level
            include_aoi_geometry: Include AOI geometry in output
        """
        # Initialize base exporter (handles output_dir, processes, chunk_size, log_level)
        super().__init__(
            output_dir=output_dir,
            processes=processes,
            chunk_size=chunk_size,
            log_level=log_level,
        )

        # RoofAgeExporter-specific attributes
        self.aoi_file = aoi_file
        self.output_format = output_format
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / "cache"
        self.no_cache = no_cache
        self.overwrite_cache = overwrite_cache
        self.compress_cache = compress_cache
        self.threads = threads
        self.country = country
        self.api_key = api_key
        self.include_aoi_geometry = include_aoi_geometry

        # Validate country
        if self.country.lower() != "us":
            raise ValueError(
                f"Roof Age API is currently only available for US properties. "
                f"Got country='{self.country}'"
            )

        # Create cache directory if needed
        if not self.no_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_chunk_output_file(self, chunk_id: str) -> Path:
        """
        Get the path to the main output file for a chunk.

        Args:
            chunk_id: Unique identifier for this chunk

        Returns:
            Path to the chunk's metadata file (used for cache checking)
        """
        return self.chunk_path / f"metadata_{chunk_id}.parquet"

    def process_chunk(
        self,
        chunk_id: str,
        aoi_gdf: gpd.GeoDataFrame,
        **kwargs
    ):
        """
        Process a chunk of AOIs to extract roof age data.

        Args:
            chunk_id: Unique identifier for this chunk
            aoi_gdf: GeoDataFrame containing AOIs to process
            **kwargs: Additional parameters (unused for roof age, but required by base class)

        Returns:
            None (results are saved to chunk files)
        """
        # Configure logging for worker process
        BaseExporter.configure_worker_logging(self.log_level)
        logger = log.get_logger()

        try:
            # Ensure chunk output directory exists (BaseExporter creates self.chunk_path)
            self.chunk_path.mkdir(parents=True, exist_ok=True)

            # Define chunk output files
            outfile_roofs = self.chunk_path / f"roofs_{chunk_id}.parquet"
            outfile_metadata = self.chunk_path / f"metadata_{chunk_id}.parquet"
            outfile_errors = self.chunk_path / f"errors_{chunk_id}.parquet"

            # Check if chunk already processed
            if outfile_metadata.exists():
                logger.debug(f"Chunk {chunk_id} already processed, skipping")
                return

            logger.debug(f"Chunk {chunk_id}: Processing {len(aoi_gdf)} AOIs")

            # Get progress counters from kwargs (passed by BaseExporter)
            progress_counters = kwargs.get("progress_counters")

            # Initialize API client for this chunk
            cache_path = None if self.no_cache else self.cache_dir
            api = RoofAgeApi(
                api_key=self.api_key,
                cache_dir=cache_path,
                overwrite_cache=self.overwrite_cache,
                compress_cache=self.compress_cache,
                threads=self.threads,
                country=self.country,
                progress_counters=progress_counters,
            )

            # Query API for this chunk
            roofs_gdf, metadata_df, errors_df = api.get_roof_age_bulk(aoi_gdf)

            logger.debug(
                f"Chunk {chunk_id}: Found {len(roofs_gdf)} roofs, "
                f"{len(metadata_df)} successful queries, {len(errors_df)} errors"
            )

            # Save chunk results
            if len(roofs_gdf) > 0:
                roofs_gdf.to_parquet(outfile_roofs)
            if len(metadata_df) > 0:
                metadata_df.to_parquet(outfile_metadata)
            if len(errors_df) > 0:
                errors_df.to_parquet(outfile_errors)

        except Exception as e:
            logger.error(f"Chunk {chunk_id} failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def run(self):
        """Execute the roof age export workflow"""
        self.logger.info(f"nmaipy version: {__version__}")
        self.logger.info(f"Starting roof age export from {self.aoi_file}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Load AOIs
        self.logger.info("Loading AOI file...")
        aoi_gdf = parcels.read_from_file(self.aoi_file, id_column=AOI_ID_COLUMN_NAME)

        # Ensure correct CRS if geometry mode (skip CRS check for address mode)
        if "geometry" in aoi_gdf.columns and aoi_gdf.crs != API_CRS:
            self.logger.info(f"Reprojecting from {aoi_gdf.crs} to {API_CRS}")
            aoi_gdf = aoi_gdf.to_crs(API_CRS)

        self.logger.info(f"Loaded {len(aoi_gdf)} AOIs")

        # Split into chunks and process in parallel (using BaseExporter methods)
        aoi_stem = Path(self.aoi_file).stem
        chunks_to_process, skipped_chunks, skipped_aois = self.split_into_chunks(
            aoi_gdf, aoi_stem, check_cache=True
        )

        # Calculate initial AOI count for progress tracking (excluding skipped)
        initial_aoi_count = len(aoi_gdf) - skipped_aois

        # Run parallel processing with progress tracking
        self.run_parallel(
            chunks_to_process,
            aoi_stem,
            initial_aoi_count=initial_aoi_count,
            use_progress_tracking=True,
        )

        # Combine chunk results
        self.logger.info("Combining chunk results...")
        roofs_list = []
        metadata_list = []
        errors_list = []

        # Calculate total number of chunks (including cached ones)
        num_chunks = max(len(aoi_gdf) // self.chunk_size, 1)

        for i in range(num_chunks):
            chunk_id = f"{aoi_stem}_{str(i).zfill(4)}"

            # Load roofs
            roofs_file = self.chunk_path / f"roofs_{chunk_id}.parquet"
            if roofs_file.exists():
                roofs_list.append(gpd.read_parquet(roofs_file))

            # Load metadata
            metadata_file = self.chunk_path / f"metadata_{chunk_id}.parquet"
            if metadata_file.exists():
                metadata_list.append(pd.read_parquet(metadata_file))

            # Load errors
            errors_file = self.chunk_path / f"errors_{chunk_id}.parquet"
            if errors_file.exists():
                errors_list.append(pd.read_parquet(errors_file))

        # Combine results
        if roofs_list:
            roofs_gdf = gpd.GeoDataFrame(pd.concat(roofs_list, ignore_index=False), crs=API_CRS)
        else:
            roofs_gdf = gpd.GeoDataFrame(columns=[AOI_ID_COLUMN_NAME, "geometry"], crs=API_CRS)

        if metadata_list:
            metadata_df = pd.concat(metadata_list, ignore_index=False)
        else:
            metadata_df = pd.DataFrame()

        if errors_list:
            errors_df = pd.concat(errors_list, ignore_index=False)
        else:
            errors_df = pd.DataFrame()

        # Report results
        success_count = len(metadata_df)
        error_count = len(errors_df)
        roof_count = len(roofs_gdf)
        self.logger.info(
            f"API queries complete: {success_count} successful, {error_count} errors, "
            f"{roof_count} total roofs found"
        )

        if error_count > 0:
            # Log error summary as ASCII table (same format as Feature API)
            status_counts = None
            message_counts = None
            if "status_code" in errors_df.columns:
                status_counts = errors_df["status_code"].value_counts()
            if "message" in errors_df.columns:
                # Sanitize URLs in messages before aggregating (truncate query params)
                sanitized_messages = errors_df["message"].apply(sanitize_error_message)
                message_counts = sanitized_messages.value_counts()

            error_table = format_error_summary_table(status_counts, message_counts)
            self.logger.info(f"Roof Age API: {error_count} failures{error_table}")

        # Merge with AOI attributes if requested
        if self.include_aoi_geometry and len(roofs_gdf) > 0:
            self.logger.info("Merging roof data with AOI attributes...")
            aoi_for_merge = aoi_gdf.rename(columns={"geometry": "aoi_geometry"})
            roofs_gdf = roofs_gdf.merge(
                aoi_for_merge,
                left_on=AOI_ID_COLUMN_NAME,
                right_index=True,
                how="left"
            )
            # Convert aoi_geometry to WKT for CSV compatibility
            if "aoi_geometry" in roofs_gdf.columns:
                roofs_gdf["aoi_geometry"] = roofs_gdf["aoi_geometry"].apply(
                    lambda g: g.wkt if g is not None else None
                )

        # Save outputs
        self._save_outputs(aoi_stem, roofs_gdf, metadata_df, errors_df, self.final_path)

        self.logger.info("Export complete!")

    def _save_outputs(
        self,
        file_stem: str,
        roofs_gdf: gpd.GeoDataFrame,
        metadata_df: pd.DataFrame,
        errors_df: pd.DataFrame,
        output_path: Path,
    ):
        """
        Save output files.

        Args:
            file_stem: Base filename (without extension)
            roofs_gdf: GeoDataFrame with roof features
            metadata_df: DataFrame with metadata
            errors_df: DataFrame with errors
            output_path: Directory to save final outputs
        """
        # Save roofs
        if len(roofs_gdf) > 0:
            if self.output_format in ["geoparquet", "both"]:
                roofs_path = output_path / f"{file_stem}_roofs.parquet"
                self.logger.info(f"Saving {len(roofs_gdf)} roofs to {roofs_path}")
                roofs_gdf.to_parquet(roofs_path, index=True)

            if self.output_format in ["csv", "both"]:
                roofs_path = output_path / f"{file_stem}_roofs.csv"
                self.logger.info(f"Saving {len(roofs_gdf)} roofs to {roofs_path}")
                # Convert geometry to WKT for CSV
                roofs_df = pd.DataFrame(roofs_gdf)
                if "geometry" in roofs_df.columns:
                    roofs_df["geometry"] = roofs_df["geometry"].apply(
                        lambda g: g.wkt if g is not None else None
                    )

                # Define public-facing fields for CSV export based on swagger spec
                # Excludes internal fields: hilbertId, timeline, resourceId, assessorDataDetails
                public_fields = [
                    AOI_ID_COLUMN_NAME,
                    "kind",
                    "installationDate",
                    "untilDate",
                    "trustScore",
                    "area",  # Not in swagger but useful metric
                    "evidenceType",
                    "evidenceTypeDescription",
                    "beforeInstallationCaptureDate",
                    "afterInstallationCaptureDate",
                    "minCaptureDate",
                    "maxCaptureDate",
                    "numberOfCaptures",
                    "roof_age_mapbrowser_url",  # Renamed from mapBrowserUrl with ?locationMarker appended
                    "assessorData",
                    "relevantPermits",
                    "geometry",
                ]
                # Include aoi_geometry at the end if present (from include_aoi_geometry flag)
                if "aoi_geometry" in roofs_df.columns:
                    public_fields.append("aoi_geometry")

                # Only keep columns that exist in the dataframe
                columns_to_save = [col for col in public_fields if col in roofs_df.columns]
                roofs_df = roofs_df[columns_to_save]

                roofs_df.to_csv(roofs_path, index=True)
        else:
            self.logger.warning("No roof data to save")

        # Save metadata
        if len(metadata_df) > 0:
            metadata_path = output_path / f"{file_stem}_metadata.csv"
            self.logger.info(f"Saving metadata to {metadata_path}")
            metadata_df.to_csv(metadata_path, index=True)

        # Save errors
        if len(errors_df) > 0:
            errors_path = output_path / f"{file_stem}_errors.csv"
            self.logger.info(f"Saving {len(errors_df)} errors to {errors_path}")
            errors_df.to_csv(errors_path, index=True)


def main():
    """Main entry point for roof age exporter CLI"""
    args = parse_arguments()

    try:
        exporter = RoofAgeExporter(
            aoi_file=args.aoi_file,
            output_dir=args.output_dir,
            output_format=args.output_format,
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
        )
        exporter.run()
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
