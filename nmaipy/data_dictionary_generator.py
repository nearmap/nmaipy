"""Data dictionary generator.

Generates a ``<filename>_data_dictionary.csv`` next to every CSV/parquet output
file containing AI data in an nmaipy export ``final/`` directory. Each
dictionary lists every column with description, allowed values, dtype, source,
min, max, and precision — sourced from ``nmaipy/data/column_metadata.json``.

Mirrors the shape of ``ReadmeGenerator``: construct with ``output_dir``, call
``generate_and_save()``. Designed to be safe to call after the README has been
written; failure is isolated and logged (the export's primary contract is the
data files themselves).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from nmaipy import storage
from nmaipy.column_metadata import ColumnMeta, lookup_column

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

# Files in ``final/`` that should NOT receive a data dictionary. Operational
# telemetry, error files, and config files are excluded per ticket scope.
_SKIP_BASENAMES = {
    "README.md",
    ".DS_Store",
    "export_config.json",
    "roof_age_export_config.json",
    "classes_availability.json",
    "latency_stats.csv",
    "feature_api_errors.csv",
    "feature_api_errors.parquet",
    "roof_age_errors.csv",
    "roof_age_errors.parquet",
    "errors.csv",  # Roof Age standalone errors
    "metadata.csv",  # Roof Age standalone — only aoi_id + resource_id
}

_SUPPORTED_EXTENSIONS = (".csv", ".parquet")

# Filename stem (with optional ``_features`` suffix stripped) → human-readable class
# label used in column descriptions. Per-class files map to their class. The rollup
# is parcel-level (one row per parcel), so its label is ``"parcel"``. ``features``
# is mixed-class and falls back to the generic ``"feature"``.
_FILENAME_CLASS_LABELS = {
    "building": "building",
    "building_lifecycle": "building lifecycle",
    "roof": "roof",
    "roof_instance": "roof age instance",
    "solar_panel": "solar panel",
    "swimming_pool": "swimming pool",
    "rollup": "parcel",
}


def _filename_to_class_label(filename: str) -> str:
    """Infer a human-readable class label from an output filename.

    ``building_features.parquet`` → ``"building"``;
    ``roof_instance.csv``         → ``"roof age instance"``;
    ``rollup.csv``                → ``"parcel"`` (one row per parcel);
    ``features.parquet``          → ``"feature"`` (mixed-class, generic).
    """
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if stem.endswith("_features"):
        stem = stem[: -len("_features")]
    return _FILENAME_CLASS_LABELS.get(stem, "feature")


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
        files = self._discover_files()
        area_unit = self._detect_area_unit()
        written: list[str] = []
        for filepath in files:
            try:
                out_path = self._build_and_write(filepath, area_unit)
            except Exception as exc:
                logger.warning(f"Data dictionary generation failed for {filepath}: {exc}")
                continue
            if out_path is not None:
                written.append(out_path)
        return written

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_files(self) -> list[str]:
        """Return paths of files in ``final/`` that should receive a data dictionary."""
        candidates = storage.glob_files(self.final_dir, "*")
        eligible: list[str] = []
        for path in sorted(candidates):
            name = storage.basename(path)
            if name in _SKIP_BASENAMES:
                continue
            if name.endswith("_data_dictionary.csv"):
                # Don't recursively generate dictionaries for previously-generated dictionaries.
                continue
            if not name.lower().endswith(_SUPPORTED_EXTENSIONS):
                continue
            eligible.append(path)
        return eligible

    # ------------------------------------------------------------------
    # Area unit detection (mirrors ReadmeGenerator behaviour)
    # ------------------------------------------------------------------

    def _detect_area_unit(self) -> str:
        """Detect 'sqft' (US) or 'sqm' (elsewhere) from export config; default sqft."""
        for config_name in ("export_config.json", "roof_age_export_config.json"):
            cfg_path = storage.join_path(self.final_dir, config_name)
            if storage.file_exists(cfg_path):
                try:
                    with storage.open_file(cfg_path, "r") as fh:
                        cfg = json.load(fh)
                    country = cfg.get("parameters", {}).get("country", "").lower()
                    if country and country != "us":
                        return "sqm"
                    return "sqft"
                except Exception:
                    continue
        return "sqft"

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

    def _build_and_write(self, filepath: str, area_unit: str) -> Optional[str]:
        columns = self._read_columns(filepath)
        class_label = _filename_to_class_label(storage.basename(filepath))
        rows = [self._row_for_column(name, area_unit, class_label) for name in columns]
        out_path = self._dictionary_path_for(filepath)
        self._write_csv(out_path, rows)
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

    @staticmethod
    def _write_csv(path: str, rows: list[dict[str, str]]) -> None:
        """Atomic write: write to ``<path>.tmp`` then replace. Skips if rows is empty.

        Prepends a UTF-8 BOM (``\\ufeff``) so Excel — which still defaults to
        Latin-1/MacRoman on macOS when opening .csv files without explicit hints
        — recognises the encoding and renders em-dashes and other Unicode
        characters correctly. Other tools (pandas, csv module, VS Code, Numbers)
        ignore or silently consume the BOM.
        """
        if not rows:
            return
        tmp_path = f"{path}.tmp"
        with storage.open_file(tmp_path, "w") as fh:
            fh.write("﻿")
            writer = csv.DictWriter(fh, fieldnames=_DD_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        # Atomic replace. ``storage.open_file`` for non-S3 paths writes to the
        # local filesystem; for S3 it has already uploaded on close. Either way,
        # rename the temp into place so a partial write never overwrites a
        # previous good dictionary.
        if storage.is_s3_path(path):
            # S3 doesn't support atomic rename; the close above completed the
            # upload, so just remove the tmp marker.
            try:
                storage.remove_file(tmp_path) if hasattr(storage, "remove_file") else None
            except Exception:
                pass
            # Re-write to the final path. Acceptable for S3 since ListObjects
            # is eventually consistent anyway.
            with storage.open_file(path, "w") as fh:
                fh.write("﻿")
                writer = csv.DictWriter(fh, fieldnames=_DD_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        else:
            Path(tmp_path).replace(Path(path))


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
