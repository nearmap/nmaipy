"""Registry of files the nmaipy exporter is allowed to produce.

Single source of truth used by `readme_generator` and `data_dictionary_generator`.
Adding a new output file should be a one-line entry here, not a distributed
update across blacklists/whitelists in multiple modules.

The `kind` field drives downstream behaviour:
- ``ai_data``     → has columns (rollup, per-class tabular); gets a data
                    dictionary.
- ``geometry``    → geoparquet with geometry; listed in README, no dictionary.
- ``operational`` → telemetry / errors; listed in README, no dictionary.
- ``config``      → input metadata; not listed in the README's file table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from nmaipy import storage


@dataclass(frozen=True)
class FileSpec:
    """Metadata for one file produced by the exporter."""

    description: str
    kind: str
    class_label: Optional[str] = None
    is_per_class: bool = False


# Static-name files. Stems only — extension is set by `tabular_file_format` for
# `ai_data` files (csv|parquet), fixed for the rest.
STATIC_FILES: dict[str, FileSpec] = {
    "rollup": FileSpec(
        description="Property-level summary with one row per property containing aggregated statistics",
        kind="ai_data",
        class_label="property",
    ),
    "buildings": FileSpec(
        description="Building-level summary (one row per building; present when --save-buildings is set)",
        kind="ai_data",
        class_label="building",
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
    ),
    "roof_age_export_config": FileSpec(
        description="Parameters used to produce this Roof Age export",
        kind="config",
    ),
}

# Per-class file stems → human-readable class label.
# Labels must match keys in `column_metadata._SCOPE_PHRASES` so that the
# `{class_label}` substitution path in `lookup_column` resolves consistently.
PER_CLASS_LABELS: dict[str, str] = {
    "building": "building",
    "building_lifecycle": "building lifecycle",
    "roof": "roof",
    "roof_instance": "roof age instance",
    "swimming_pool": "swimming pool",
    "solar_panel": "solar panel",
}


def file_spec_for(filename: str) -> Optional[FileSpec]:
    """Return the `FileSpec` for a given basename, or None if not in the registry.

    Recognises:
    - ``<stem>.<ext>`` for stems in `STATIC_FILES` (any extension).
    - ``<cname>.csv`` / ``<cname>.parquet`` for cnames in `PER_CLASS_LABELS`
      (per-class tabular AI data).
    - ``<cname>_features.parquet`` for cnames in `PER_CLASS_LABELS` (per-class
      geometry / GeoParquet).
    """
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
    """Enumerate tabular AI-data files that exist on disk under `final_dir`.

    Args:
        final_dir: Path to the export's ``final/`` directory (local or ``s3://``).
        ext: Tabular extension to look for — ``"csv"`` or ``"parquet"``. Comes from
            the export config's ``tabular_file_format``.

    Returns:
        ``[(filepath, class_label), ...]`` for each registry entry whose file
        exists at ``final_dir/<stem>.<ext>``. Used by the data dictionary
        generator to emit one dictionary per logical class file (no duplication
        across CSV+parquet variants, no geoparquet companions).
    """
    out: list[tuple[str, str]] = []
    for stem, spec in STATIC_FILES.items():
        if spec.kind != "ai_data":
            continue
        path = storage.join_path(final_dir, f"{stem}.{ext}")
        if storage.file_exists(path):
            out.append((path, spec.class_label or "property"))
    for cname, label in PER_CLASS_LABELS.items():
        path = storage.join_path(final_dir, f"{cname}.{ext}")
        if storage.file_exists(path):
            out.append((path, label))
    return out
