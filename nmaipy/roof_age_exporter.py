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
from tqdm import tqdm

from nmaipy import log, parcels
from nmaipy.__version__ import __version__
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
        "--threads",
        help="Number of concurrent API requests (default: 10)",
        type=int,
        default=10,
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


class RoofAgeExporter:
    """
    Exporter for bulk roof age data retrieval.

    Handles parallel processing, progress tracking, caching, and output generation.
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
        threads: int = 10,
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
            threads: Number of concurrent threads
            country: Country code (must be 'us')
            api_key: API key (optional, uses environment variable if not provided)
            log_level: Logging level
            include_aoi_geometry: Include AOI geometry in output
        """
        self.aoi_file = aoi_file
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / "cache"
        self.no_cache = no_cache
        self.overwrite_cache = overwrite_cache
        self.compress_cache = compress_cache
        self.threads = threads
        self.country = country
        self.api_key = api_key
        self.log_level = log_level
        self.include_aoi_geometry = include_aoi_geometry

        # Configure logging
        log.configure_logger(self.log_level)
        self.logger = log.get_logger()

        # Validate country
        if self.country.lower() != "us":
            raise ValueError(
                f"Roof Age API is currently only available for US properties. "
                f"Got country='{self.country}'"
            )

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.no_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Execute the roof age export workflow"""
        self.logger.info(f"Starting roof age export from {self.aoi_file}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Load AOIs
        self.logger.info("Loading AOI file...")
        aoi_gdf = parcels.read_from_file(self.aoi_file, id_column=AOI_ID_COLUMN_NAME)

        # Validate that we have geometries
        if not isinstance(aoi_gdf, gpd.GeoDataFrame):
            raise ValueError(
                "Roof Age API requires geometry (AOI polygons). "
                "Address-based queries are not yet supported in bulk mode."
            )

        # Ensure correct CRS
        if aoi_gdf.crs != API_CRS:
            self.logger.info(f"Reprojecting from {aoi_gdf.crs} to {API_CRS}")
            aoi_gdf = aoi_gdf.to_crs(API_CRS)

        self.logger.info(f"Loaded {len(aoi_gdf)} AOIs")

        # Initialize API client
        cache_path = None if self.no_cache else self.cache_dir
        api = RoofAgeApi(
            api_key=self.api_key,
            cache_dir=cache_path,
            overwrite_cache=self.overwrite_cache,
            compress_cache=self.compress_cache,
            threads=self.threads,
        )

        # Query API for all AOIs
        self.logger.info(f"Querying Roof Age API with {self.threads} concurrent threads...")
        roofs_gdf, metadata_df, errors_df = api.get_roof_age_bulk(aoi_gdf)

        # Report results
        success_count = len(metadata_df)
        error_count = len(errors_df)
        roof_count = len(roofs_gdf)
        self.logger.info(
            f"API queries complete: {success_count} successful, {error_count} errors, "
            f"{roof_count} total roofs found"
        )

        if error_count > 0:
            self.logger.warning(f"Failed queries: {error_count} / {len(aoi_gdf)}")
            # Log error details
            if "message" in errors_df.columns:
                error_summary = errors_df["message"].value_counts().to_dict()
                self.logger.warning(f"Error breakdown: {error_summary}")

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
        aoi_stem = Path(self.aoi_file).stem
        self._save_outputs(aoi_stem, roofs_gdf, metadata_df, errors_df)

        self.logger.info("Export complete!")

    def _save_outputs(
        self,
        file_stem: str,
        roofs_gdf: gpd.GeoDataFrame,
        metadata_df: pd.DataFrame,
        errors_df: pd.DataFrame,
    ):
        """
        Save output files.

        Args:
            file_stem: Base filename (without extension)
            roofs_gdf: GeoDataFrame with roof features
            metadata_df: DataFrame with metadata
            errors_df: DataFrame with errors
        """
        # Save roofs
        if len(roofs_gdf) > 0:
            if self.output_format in ["geoparquet", "both"]:
                roofs_path = self.output_dir / f"{file_stem}_roofs.parquet"
                self.logger.info(f"Saving {len(roofs_gdf)} roofs to {roofs_path}")
                roofs_gdf.to_parquet(roofs_path, index=True)

            if self.output_format in ["csv", "both"]:
                roofs_path = self.output_dir / f"{file_stem}_roofs.csv"
                self.logger.info(f"Saving {len(roofs_gdf)} roofs to {roofs_path}")
                # Convert geometry to WKT for CSV
                roofs_df = pd.DataFrame(roofs_gdf)
                if "geometry" in roofs_df.columns:
                    roofs_df["geometry"] = roofs_df["geometry"].apply(
                        lambda g: g.wkt if g is not None else None
                    )
                roofs_df.to_csv(roofs_path, index=True)
        else:
            self.logger.warning("No roof data to save")

        # Save metadata
        if len(metadata_df) > 0:
            metadata_path = self.output_dir / f"{file_stem}_metadata.csv"
            self.logger.info(f"Saving metadata to {metadata_path}")
            metadata_df.to_csv(metadata_path, index=True)

        # Save errors
        if len(errors_df) > 0:
            errors_path = self.output_dir / f"{file_stem}_errors.csv"
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
            threads=args.threads,
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
