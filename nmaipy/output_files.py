"""Registry of files the nmaipy exporter produces. Consumed by
``readme_generator`` and ``data_dictionary_generator`` so output file metadata
lives in one place.

``kind`` drives downstream behaviour:
- ``ai_data``     → has columns; gets a data dictionary.
- ``geometry``    → geoparquet with geometry; listed in README, no dictionary.
- ``operational`` → telemetry / errors; listed in README, no dictionary.
- ``config``      → input metadata; not listed in the README's file table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from nmaipy import storage
from nmaipy.column_metadata import PER_CLASS_LABELS


@dataclass(frozen=True)
class FileSpec:
    """Metadata for one file produced by the exporter."""

    description: str
    kind: str
    class_label: Optional[str] = None
    is_per_class: bool = False
    list_in_readme: bool = True


# Static-name files. Per-class files are derived from PER_CLASS_LABELS in
# ``file_spec_for``. Extension for ``ai_data`` files is the export's
# ``tabular_file_format``; other entries are fixed-extension.
STATIC_FILES: dict[str, FileSpec] = {
    "rollup": FileSpec(
        description="Property-level summary with one row per parcel containing aggregated statistics",
        kind="ai_data",
        # "parcel" triggers parcel-aggregate scope phrasing in column_metadata.lookup_column.
        class_label="parcel",
    ),
    "features": FileSpec(
        description="All detected features with geometry (GeoParquet)",
        kind="geometry",
    ),
    "feature_api_errors": FileSpec(
        description="Properties where Feature API calls failed",
        kind="operational",
    ),
    "roof_age_errors": FileSpec(
        description="Properties where Roof Age API calls failed",
        kind="operational",
    ),
    "latency_stats": FileSpec(
        description="API call timing statistics",
        kind="operational",
    ),
    "classes_availability": FileSpec(
        description="Available feature classes per AOI",
        kind="operational",
    ),
    "export_config": FileSpec(
        description="Parameters used to produce this export",
        kind="config",
        list_in_readme=False,
    ),
    "roof_age_export_config": FileSpec(
        description="Parameters used to produce this Roof Age export",
        kind="config",
        list_in_readme=False,
    ),
}


def file_spec_for(filename: str) -> Optional[FileSpec]:
    """Return the `FileSpec` for a given basename, or None if not in the registry.

    Recognises:
    - ``<stem>.<ext>`` for stems in `STATIC_FILES` (any extension).
    - ``<cname>.csv`` / ``<cname>.parquet`` for cnames in `PER_CLASS_LABELS`
      (per-class tabular AI data).
    - ``<cname>_features.parquet`` for cnames in `PER_CLASS_LABELS` (per-class
      geometry / GeoParquet).
    - ``*_data_dictionary.csv`` — auto-generated sidecar; not listed.
    """
    if filename.endswith("_data_dictionary.csv"):
        return FileSpec(
            description="Auto-generated data dictionary describing the columns of the matching output file",
            kind="documentation",
            list_in_readme=False,
        )
    if "." not in filename:
        return None
    stem, ext = filename.rsplit(".", 1)
    if stem in STATIC_FILES:
        return STATIC_FILES[stem]
    if stem in PER_CLASS_LABELS and ext in ("csv", "parquet"):
        label = PER_CLASS_LABELS[stem]
        return FileSpec(
            description=f"Per-{label} data with feature attributes",
            kind="ai_data",
            class_label=label,
            is_per_class=True,
        )
    if ext == "parquet" and stem.endswith("_features"):
        cname = stem[: -len("_features")]
        if cname in PER_CLASS_LABELS:
            label = PER_CLASS_LABELS[cname]
            return FileSpec(
                description=f"{label.capitalize()} polygons with geometry (GeoParquet)",
                kind="geometry",
                class_label=label,
                is_per_class=True,
            )
    return None


def tabular_ai_files(final_dir: str, ext: str) -> list[tuple[str, str]]:
    """Return ``[(filepath, class_label), ...]`` for tabular AI-data files
    present at ``final_dir/<stem>.<ext>``. ``ext`` is ``csv`` or ``parquet``.
    """
    out: list[tuple[str, str]] = []
    for stem, spec in STATIC_FILES.items():
        if spec.kind != "ai_data":
            continue
        path = storage.join_path(final_dir, f"{stem}.{ext}")
        if storage.file_exists(path):
            out.append((path, spec.class_label or "parcel"))
    for cname, label in PER_CLASS_LABELS.items():
        path = storage.join_path(final_dir, f"{cname}.{ext}")
        if storage.file_exists(path):
            out.append((path, label))
    return out
