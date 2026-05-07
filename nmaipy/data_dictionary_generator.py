"""Data dictionary generator.

Generates a ``<filename>_data_dictionary.csv`` next to every tabular AI-data
output file in an nmaipy export ``final/`` directory. Each dictionary lists
every column with description, allowed values, dtype, source, min, max, and
precision — sourced from ``nmaipy/data/column_metadata.json`` via
``column_metadata.lookup_column``.

The set of files we emit dictionaries for is driven by the
``nmaipy.output_files`` registry, gated on the export's ``tabular_file_format``
(csv|parquet) so geoparquet companions and parallel format variants don't get
redundant dictionaries.

Mirrors the shape of ``ReadmeGenerator``: construct with ``output_dir``, call
``generate_and_save()``. Failure is isolated and logged per-file; the export's
primary contract is the data files themselves, and the README atomic write
remains the "all done" sentinel.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from nmaipy import output_files, storage
from nmaipy.column_metadata import lookup_column

logger = logging.getLogger(__name__)

# CSV column order in the generated data dictionaries. Matches the field
# sequence requested in the INDS-2080 ticket.
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
        written: list[str] = []
        for filepath, class_label in output_files.tabular_ai_files(self.final_dir, ext):
            try:
                out_path = self._build_and_write(filepath, class_label, area_unit)
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

    def _build_and_write(self, filepath: str, class_label: str, area_unit: str) -> Optional[str]:
        columns = self._read_columns(filepath)
        rows = [self._row_for_column(name, area_unit, class_label) for name in columns]
        if not rows:
            return None
        out_path = self._dictionary_path_for(filepath)
        df = pd.DataFrame(rows, columns=_DD_COLUMNS)
        # ``encoding="utf-8-sig"`` writes a BOM so Excel on macOS renders Unicode
        # (em-dashes, etc.) correctly when opening the .csv. Pandas + fsspec
        # handles ``s3://`` URIs natively — same idiom as the rollup write
        # (exporter.py ``data.to_csv(outpath, ...)``).
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

    def _row_for_column(self, name: str, area_unit: str, class_label: str) -> dict[str, str]:
        """Build a single dictionary row for one column."""
        meta = lookup_column(name, area_unit=area_unit, class_label=class_label)
        return {
            "column_name": name,
            "description": meta.description,
            "allowed_values": meta.allowed_values,
            "dtype": meta.dtype,
            "source": meta.source,
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


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def generate_dictionaries(output_dir: str) -> list[str]:
    """Generate dictionaries for an export. Returns list of written paths.

    Wrapper around ``DataDictionaryGenerator(output_dir).generate_and_save()``
    that swallows top-level errors and logs a warning, mirroring the
    failure-isolation contract requested in the design.
    """
    try:
        return DataDictionaryGenerator(output_dir=output_dir).generate_and_save()
    except Exception as exc:
        logger.warning(f"Data dictionary generation failed for {output_dir}: {exc}")
        return []
