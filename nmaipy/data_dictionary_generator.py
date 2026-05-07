"""Generate ``<filename>_data_dictionary.csv`` next to every tabular AI-data
output in an export ``final/`` directory. Columns are looked up via
``column_metadata.lookup_column``; eligible files come from the
``output_files`` registry, gated on the export's ``tabular_file_format``.
Per-file failures are logged and isolated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from nmaipy import output_files, storage
from nmaipy.column_metadata import (
    UNKNOWN_SENTINEL,
    USER_INPUT_DESCRIPTION,
    USER_INPUT_SOURCE,
    lookup_column,
)

logger = logging.getLogger(__name__)

_DD_COLUMNS = [
    "column_name",
    "description",
    "allowed_values",
    "dtype",
    "source",
    "min",
    "max",
    "precision",
]


class DataDictionaryGenerator:
    """Generate per-output ``*_data_dictionary.csv`` files for an export directory."""

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Path to the export directory. Either the full export root
                (``output_dir/final``) or the ``final/`` directory itself; the
                constructor finds the right one in the same way ``ReadmeGenerator``
                does.
        """
        self.output_dir = str(output_dir)
        self._is_s3 = storage.is_s3_path(self.output_dir)

        # Resolve the actual directory containing the AI-data files.
        potential_final = storage.join_path(self.output_dir, "final")
        if not self._is_s3 and Path(potential_final).exists() and Path(potential_final).is_dir():
            self.final_dir = potential_final
        elif self._is_s3 and storage.glob_files(potential_final, "*"):
            self.final_dir = potential_final
        else:
            self.final_dir = self.output_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_and_save(self) -> list[str]:
        """Generate all data dictionaries for the export. Returns list of written paths."""
        ext = self._tabular_extension()
        area_unit = self._detect_area_unit()
        extras = self._description_extras()
        input_columns = self._input_columns()
        written: list[str] = []
        for filepath, class_label in output_files.tabular_ai_files(self.final_dir, ext):
            try:
                out_path = self._build_and_write(filepath, class_label, area_unit, extras, input_columns)
            except Exception as exc:
                logger.warning(f"Data dictionary generation failed for {filepath}: {exc}")
                continue
            if out_path is not None:
                written.append(out_path)
        return written

    # ------------------------------------------------------------------
    # Config-driven detection
    # ------------------------------------------------------------------

    def _load_export_config(self) -> dict:
        """Load export_config.json (or roof_age_export_config.json) if present."""
        for config_name in ("export_config.json", "roof_age_export_config.json"):
            cfg_path = storage.join_path(self.final_dir, config_name)
            if storage.file_exists(cfg_path):
                try:
                    with storage.open_file(cfg_path, "r") as fh:
                        return json.load(fh)
                except Exception:
                    continue
        return {}

    def _detect_area_unit(self) -> str:
        """Return ``'sqft'`` (US) or ``'sqm'`` (elsewhere) based on export config; default sqft."""
        country = self._load_export_config().get("parameters", {}).get("country", "").lower()
        return "sqm" if country and country != "us" else "sqft"

    def _tabular_extension(self) -> str:
        """Return ``'csv'`` or ``'parquet'`` based on the export's tabular_file_format setting."""
        fmt = self._load_export_config().get("parameters", {}).get("tabular_file_format", "csv")
        return fmt if fmt in ("csv", "parquet") else "csv"

    def _input_columns(self) -> set[str]:
        """Return column names from the input AOI file, or empty set on any failure.

        Used to recognise pass-through user columns (e.g. ``external_id``,
        ``address``) so the data dictionary can describe them as input columns
        rather than rendering the unknown sentinel. Reads only the header.
        """
        aoi_file = self._load_export_config().get("parameters", {}).get("aoi_file")
        if not aoi_file:
            return set()
        try:
            return _read_header_columns(str(aoi_file))
        except Exception as exc:
            logger.debug(f"Could not read input AOI columns from {aoi_file!r}: {exc}")
            return set()

    def _description_extras(self) -> dict[str, str]:
        """Build the extra-substitution dict used by ``column_metadata.lookup_column``.

        Currently surfaces the export's ``primary_decision`` strategy so that
        descriptions like ``is_primary`` can render the actual selection
        method ("optimal" / "nearest" / "largest_intersection") instead of
        referencing a CLI flag the customer never sees.
        """
        params = self._load_export_config().get("parameters", {})
        method = params.get("primary_decision") or "optimal"
        method_descriptions = {
            "optimal": "the optimal strategy (geocoded point preferred, falling back to largest intersection)",
            "nearest": "the nearest-to-centroid strategy",
            "largest_intersection": "the largest-intersection strategy",
        }
        return {
            "primary_strategy": method,
            "primary_strategy_description": method_descriptions.get(method, method),
        }

    # ------------------------------------------------------------------
    # Schema reads
    # ------------------------------------------------------------------

    def _read_columns(self, filepath: str) -> list[str]:
        """Return ordered column names from a CSV or parquet file."""
        name = storage.basename(filepath).lower()
        if name.endswith(".csv"):
            with storage.open_file(filepath, "r") as fh:
                df_head = pd.read_csv(fh, nrows=0)
            return list(df_head.columns)
        if name.endswith(".parquet"):
            try:
                with storage.open_file(filepath, "rb") as fh:
                    schema = pq.read_schema(fh)
                return list(schema.names)
            except Exception:
                # Fallback for any case where pq.read_schema can't take the file handle.
                df_head = pd.read_parquet(filepath, columns=[])
                return list(df_head.columns)
        raise ValueError(f"Unsupported file extension for {filepath!r}.")

    # ------------------------------------------------------------------
    # Per-file dictionary build + write
    # ------------------------------------------------------------------

    def _build_and_write(
        self,
        filepath: str,
        class_label: str,
        area_unit: str,
        extras: dict[str, str],
        input_columns: set[str],
    ) -> Optional[str]:
        columns = self._read_columns(filepath)
        rows = [self._row_for_column(name, area_unit, class_label, extras, input_columns) for name in columns]
        if not rows:
            return None
        out_path = self._dictionary_path_for(filepath)
        df = pd.DataFrame(rows, columns=_DD_COLUMNS)
        # utf-8-sig writes a BOM so Excel on macOS renders Unicode correctly.
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

    def _row_for_column(
        self,
        name: str,
        area_unit: str,
        class_label: str,
        extras: dict[str, str],
        input_columns: set[str],
    ) -> dict[str, str]:
        """Build a single dictionary row for one column."""
        meta = lookup_column(name, area_unit=area_unit, class_label=class_label, extras=extras)
        # Pass-through columns from the user's input file aren't in the seeded
        # metadata. Recognise them so the dictionary describes them rather
        # than surfacing the unknown sentinel.
        if meta.description == UNKNOWN_SENTINEL and name in input_columns:
            description = USER_INPUT_DESCRIPTION
            source = USER_INPUT_SOURCE
        else:
            description = meta.description
            source = meta.source
        return {
            "column_name": name,
            "description": description,
            "allowed_values": meta.allowed_values,
            "dtype": meta.dtype,
            "source": source,
            "min": meta.min,
            "max": meta.max,
            "precision": meta.precision,
        }

    @staticmethod
    def _dictionary_path_for(filepath: str) -> str:
        """``rollup.csv`` → ``rollup_data_dictionary.csv``; ``foo.parquet`` → ``foo_data_dictionary.csv``."""
        directory = filepath.rsplit("/", 1)[0] if "/" in filepath else "."
        name = storage.basename(filepath)
        stem = name.rsplit(".", 1)[0]
        return storage.join_path(directory, f"{stem}_data_dictionary.csv")


def _read_header_columns(path: str) -> set[str]:
    """Return the set of column names from an input AOI file by reading just the header.

    Supports the same formats as ``aoi_io.read_from_file`` (CSV/PSV/TSV,
    GeoJSON, GeoPackage, Parquet). Raises on unsupported formats; callers
    catch and degrade gracefully.
    """
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    sep_for = {"csv": ",", "psv": "|", "tsv": "\t"}
    if suffix in sep_for:
        with storage.open_file(path, "r") as fh:
            df = pd.read_csv(fh, sep=sep_for[suffix], nrows=0)
        return set(df.columns)
    if suffix == "parquet":
        with storage.open_file(path, "rb") as fh:
            schema = pq.read_schema(fh)
        return set(schema.names)
    if suffix in ("geojson", "json", "gpkg"):
        # geopandas can read these; cheaper than alternatives and the file
        # is typically small enough that loading it once is fine.
        import geopandas as gpd

        gdf = gpd.read_file(path, rows=0)
        return set(gdf.columns)
    raise ValueError(f"Unsupported input AOI format: {path!r}")


def generate_dictionaries(output_dir: str) -> list[str]:
    """Generate dictionaries for an export, swallowing top-level errors. Returns written paths."""
    try:
        return DataDictionaryGenerator(output_dir=output_dir).generate_and_save()
    except Exception as exc:
        logger.warning(f"Data dictionary generation failed for {output_dir}: {exc}")
        return []
