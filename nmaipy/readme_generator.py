"""Dynamic README generator for nmaipy exports.

Generates customer-facing documentation based on actual files and columns
present in an export directory, rather than using a static template.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from nmaipy import output_files, storage
from nmaipy.__version__ import __version__
from nmaipy.column_metadata import (
    ADDRESS_QUERY_COLUMNS,
    COMMON_COLUMNS,
    DEFENSIBLE_SPACE_ZONE_COLUMNS,
    DOMINANT_ROOF_MATERIAL_COLUMNS,
    DOMINANT_ROOF_TYPES_COLUMNS,
    ROOF_AGE_COLUMNS,
    RSI_COLUMNS,
    evidence_type_legend,
    lookup_column,
)
from nmaipy.constants import NEAREST_TOLERANCE_METERS
from nmaipy.reference_code import BUILDING_SMALL_MAX_AREA_SQM

logger = logging.getLogger(__name__)

# Files always skipped from the README's file listing.
_NEVER_LIST = {"README.md", ".DS_Store"}

_AREA_UNIT_LONG_NAMES = {"sqft": "square feet", "sqm": "square metres"}


def _render_columns_table(column_names: Iterable[str], area_unit: str = "", class_label: str = "parcel") -> list[str]:
    """Render the named columns as a 7-column markdown table.

    Each name is resolved via ``column_metadata.lookup_column`` so that
    ``{unit}`` / ``{class_label}`` / ``{scope_phrase}`` substitution stays
    consistent with the data dictionary. Templated names (``area_{unit}``)
    are resolved against ``area_unit`` before lookup; empty min/max/unit/
    example fields render as ``—``.
    """
    lines = [
        "| Column | Type | Min | Max | Unit | Example | Description |",
        "|--------|------|-----|-----|------|---------|-------------|",
    ]
    for raw_name in column_names:
        resolved_name = raw_name.replace("{unit}", area_unit)
        meta = lookup_column(resolved_name, area_unit=area_unit, class_label=class_label)
        dtype = meta.dtype or "—"
        mn = meta.min or "—"
        mx = meta.max or "—"
        unit = meta.unit or "—"
        example = meta.example or "—"
        desc = meta.description or ""
        lines.append(f"| `{resolved_name}` | {dtype} | {mn} | {mx} | {unit} | {example} | {desc} |")
    return lines


class ReadmeGenerator:
    """Generates dynamic README.md based on actual export files and columns."""

    def __init__(self, output_dir):
        """
        Initialize the README generator.

        Args:
            output_dir: Path to export output directory (string or Path, may be S3 URI).
                Can be either:
                - The parent directory containing a final/ subdirectory
                - The final directory itself containing the export files
        """
        self.output_dir = str(output_dir)
        self._config = None
        self._is_s3 = storage.is_s3_path(self.output_dir)

        # Determine the actual final directory
        # Check if output_dir/final exists, otherwise assume output_dir is the final dir
        potential_final = storage.join_path(self.output_dir, "final")
        if not self._is_s3 and Path(potential_final).exists() and Path(potential_final).is_dir():
            self.final_dir = potential_final
        elif self._is_s3 and storage.glob_files(potential_final, "*"):
            self.final_dir = potential_final
        else:
            self.final_dir = self.output_dir

    def _load_export_config(self) -> dict:
        """Load export_config.json if it exists. Result is cached after first call."""
        if self._config is not None:
            return self._config
        config_path = storage.join_path(self.final_dir, "export_config.json")
        if storage.file_exists(config_path):
            try:
                self._config = storage.read_json(config_path)
                return self._config
            except Exception:
                pass
        self._config = {}
        return self._config

    def generate_and_save(self) -> str:
        """
        Generate README and save to final directory.

        Writes atomically: content is first written to README.md.tmp and then
        moved to README.md. The exporter's run_inner uses README.md presence as
        a "fully complete" marker that skips rebuild on re-run; a torn write
        would falsely signal completion. Atomic move (os.replace on local,
        s3fs.mv on S3) guarantees README.md only ever appears with the full
        generated content.

        Returns:
            Path to generated README.md file (string, may be S3 URI).
        """
        content = self._generate()
        readme_path = storage.join_path(self.final_dir, "README.md")
        tmp_path = readme_path + ".tmp"
        # encoding="utf-8" is required: README content contains em-dashes and
        # other non-ASCII characters, and on Windows open() defaults to cp1252
        # which can't encode them — would crash with UnicodeEncodeError.
        with storage.open_file(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        storage.move_file(tmp_path, readme_path)
        return readme_path

    def _generate(self) -> str:
        """Generate the README markdown content."""
        files = self._discover_files()
        classes = self._detect_classes(files)
        rollup_columns = self._get_rollup_columns(files)

        has_address_query = self._has_address_query_columns(rollup_columns)
        has_dominant = self._has_dominant_columns(rollup_columns)
        has_rsi = self._has_rsi_columns(rollup_columns)
        has_roof_age = self._has_roof_age_columns(rollup_columns)
        has_defensible_space = self._has_defensible_space_columns(rollup_columns)
        area_unit = self._detect_area_unit()

        sections = []
        sections.append(self._generate_header())
        sections.append(self._generate_files_table(files))
        sections.append(self._generate_classes_section(classes))
        sections.append(self._generate_class_hierarchy_section(classes))
        sections.append(self._generate_column_patterns_section(classes, area_unit))
        sections.append(self._generate_common_columns_section())

        if has_address_query:
            sections.append(self._generate_address_query_section(rollup_columns))
        if has_dominant:
            sections.append(self._generate_dominant_section(area_unit))
        if has_rsi:
            sections.append(self._generate_rsi_section())
        if has_roof_age:
            sections.append(self._generate_roof_age_section())
        if has_defensible_space:
            sections.append(self._generate_defensible_space_section(area_unit))

        sections.append(self._generate_data_notes(area_unit))
        sections.append(self._generate_footer())

        return "\n".join(sections)

    def _discover_files(self) -> list[str]:
        """List files in ``final/`` worth showing in the README's file table.

        Inclusion is governed by the ``output_files`` registry's
        ``list_in_readme`` flag; ``_NEVER_LIST`` covers files outside the
        registry. Unrecognised files are listed with a generic description so
        unexpected outputs are visible.
        """
        all_files = storage.glob_files(self.final_dir, "*")
        files = []
        for f in sorted(all_files):
            name = storage.basename(f)
            if name in _NEVER_LIST:
                continue
            spec = output_files.file_spec_for(name)
            if spec is not None and not spec.list_in_readme:
                continue
            files.append(f)
        return files

    def _detect_classes(self, files: list[str]) -> list[dict]:
        """
        Detect feature classes from filenames.

        Returns list of dicts with 'name' (display) and 'column' (snake_case).
        Prefers CSV files; falls back to _features.parquet files if no CSVs found.
        """
        skip_names = {
            "rollup",
            "feature_api_errors",
            "roof_age_errors",
            "latency_stats",
        }
        classes = []
        seen = set()

        def _get_stem(filepath):
            """Get filename without extension."""
            name = storage.basename(filepath)
            return name.rsplit(".", 1)[0] if "." in name else name

        def _get_suffix(filepath):
            """Get file extension including the dot."""
            name = storage.basename(filepath)
            return "." + name.rsplit(".", 1)[1] if "." in name else ""

        # Primary: detect from per-class tabular files (CSV or Parquet)
        for f in files:
            suffix = _get_suffix(f)
            if suffix not in (".csv", ".parquet"):
                continue
            name = _get_stem(f)

            if name in skip_names:
                continue

            # Skip GeoParquet companion files and known non-class files
            if name.endswith("_features") or name == "features":
                continue

            column_name = name
            display_name = name.replace("_", " ").title()

            if column_name not in seen:
                seen.add(column_name)
                classes.append({"name": display_name, "column": column_name})

        # Fallback: detect from _features.parquet files if no tabular class files found
        if not classes:
            for f in files:
                if _get_suffix(f) != ".parquet" or not _get_stem(f).endswith("_features"):
                    continue
                name = _get_stem(f)

                # Strip _features suffix to get class name
                name = name[: -len("_features")]
                if not name:
                    continue

                if name in skip_names:
                    continue

                column_name = name
                display_name = name.replace("_", " ").title()

                if column_name not in seen:
                    seen.add(column_name)
                    classes.append({"name": display_name, "column": column_name})

        return classes

    def _get_rollup_columns(self, files: list[str]) -> set[str]:
        """Get column names from the rollup file (CSV or Parquet)."""
        rollup_csv = None
        rollup_parquet = None
        for f in files:
            name = storage.basename(f)
            if name == "rollup.csv":
                rollup_csv = f
            elif name == "rollup.parquet":
                rollup_parquet = f

        if rollup_csv:
            try:
                df = pd.read_csv(rollup_csv, nrows=0)
                return set(df.columns)
            except Exception:
                pass

        if rollup_parquet:
            try:
                schema = pq.read_schema(rollup_parquet)
                return set(schema.names)
            except Exception:
                pass

        return set()

    def _detect_area_unit(self) -> str:
        """Detect area unit from export config country. Returns 'sqft' or 'sqm'."""
        config = self._load_export_config()
        country = config.get("parameters", {}).get("country", "").lower()
        if country and country != "us":
            return "sqm"
        return "sqft"

    def _has_address_query_columns(self, columns: set[str]) -> bool:
        """Check if address or query coordinate columns are present."""
        return "streetAddress" in columns or "query_aoi_lat" in columns

    def _has_dominant_columns(self, columns: set[str]) -> bool:
        """Check if dominant material/shape summary columns are present.

        The rollup emits these columns under the ``primary_roof_`` scope (e.g.
        ``primary_roof_dominant_roof_material_description``); per-class roof
        files carry the same columns unscoped.
        """
        prefixes = (
            "dominant_roof_material_",
            "dominant_roof_types_",
            "primary_roof_dominant_roof_material_",
            "primary_roof_dominant_roof_types_",
        )
        return any(c.startswith(p) for c in columns for p in prefixes)

    def _has_rsi_columns(self, columns: set[str]) -> bool:
        """Check if Roof Spotlight Index columns are present."""
        return any("roof_spotlight_index" in c for c in columns)

    def _has_roof_age_columns(self, columns: set[str]) -> bool:
        """Check if roof age columns are present."""
        return any("roof_age_" in c for c in columns)

    def _has_defensible_space_columns(self, columns: set[str]) -> bool:
        """Check if defensible space columns are present."""
        return any("defensible_space_zone_" in c for c in columns)

    def _generate_header(self) -> str:
        """Generate the README header."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"""# Nearmap AI Export

This folder contains AI-generated property data from Nearmap aerial imagery.

**Generated:** {date_str} | **nmaipy:** v{__version__}

---
"""

    def _generate_files_table(self, files: list[str]) -> str:
        """Generate the files table section."""
        lines = [
            "## Files in This Export",
            "",
            "| File Name | Description |",
            "|-----------|-------------|",
        ]

        for f in files:
            name = storage.basename(f)
            description = self._get_file_description(name)
            lines.append(f"| `{name}` | {description} |")

        lines.append("")
        return "\n".join(lines)

    def _get_file_description(self, filename: str) -> str:
        """Description for ``filename`` from the registry, or a generic fallback."""
        spec = output_files.file_spec_for(filename)
        if spec is not None:
            return spec.description
        logger.debug("No registry entry for %r; using generic description.", filename)
        return "Export data file"

    def _generate_classes_section(self, classes: list[dict]) -> str:
        """Generate the feature classes section."""
        if not classes:
            return ""

        lines = [
            "## Feature Classes in This Export",
            "",
            "Column names use snake_case versions of class names:",
            "",
            "| Class Name | Column Prefix |",
            "|------------|---------------|",
        ]

        for cls in classes:
            lines.append(f"| {cls['name']} | `{cls['column']}_` |")

        lines.append("")
        return "\n".join(lines)

    def _generate_class_hierarchy_section(self, classes: list[dict]) -> str:
        """Generate the class hierarchy & relationships section.

        Always renders the full canonical 4-layer hierarchy when at least one
        of the structural classes (Building Lifecycle / Building / Roof /
        Roof Instance) is present in the export. Unrelated exports (e.g.
        vegetation-only) skip the section entirely.
        """
        structural_columns = {"building_lifecycle", "building", "roof", "roof_instance"}
        present_columns = {c["column"] for c in classes}
        if not (present_columns & structural_columns):
            return ""

        # Canonical hierarchy. Edge labels reflect the API's actual linkage
        # mechanism between adjacent layers; render in full so customers see
        # the conceptual structure even when their export only includes some
        # of the layers.
        layers = [
            (
                "building_lifecycle",
                "Building Lifecycle",
                "stable building identity, linked across surveys",
                "parent_id",
            ),
            ("building", "Building", "building footprint", "spatial IoU"),
            ("roof", "Roof", "roof footprint", "spatial IoU"),
            ("roof_instance", "Roof Instance", "roof clipped to parcel — the unit roof age is reported on", None),
        ]
        max_label_len = max(len(label) for _, label, _, _ in layers)
        tree_lines: list[str] = []
        for i, (_, label, note, edge) in enumerate(layers):
            tree_lines.append(f"  {label.ljust(max_label_len)}  ({note})")
            if edge is not None:
                tree_lines.append(f"  {' ' * max_label_len}    │  {edge}")
                tree_lines.append(f"  {' ' * max_label_len}    ▼")

        bullets = [
            "- **Building Lifecycle ↔ Building** is a 1-hop traversal of the API's `parent_id`.",
            "- **Building ↔ Roof** and **Roof ↔ Roof Instance** use spatial Intersection over "
            "Union (IoU), not `parent_id`. The roof's API `parentId` points to a deprecated "
            "Building class, so nmaipy re-links via geometry, with an IoU-based threshold to "
            "assign a parent.",
            "- Each parent feature gets a `primary_child_*_id` (the child with the highest IoU); "
            "each child gets a `parent_*_id`.",
            "- When the primary roof has structural damage that masks its polygon, RSI and "
            "similar scores fall back through Roof → Building → Building Lifecycle.",
        ]

        lines = [
            "## Class Hierarchy & Relationships",
            "",
            "nmaipy connects structural classes so that primary-feature columns and per-class "
            "files can be cross-referenced. The canonical hierarchy is shown below; only the "
            "layers actually requested in your export will have corresponding files in `final/`.",
            "",
            "```",
            *tree_lines,
            "```",
            "",
            "**How the layers are linked:**",
            "",
            *bullets,
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _hierarchy_edge_label(parent_col: str, child_col: str) -> str:
        """Return the link-mechanism label for a parent → child edge in the hierarchy tree."""
        if parent_col == "building_lifecycle" and child_col == "building":
            return "parent_id"
        return "spatial IoU"

    def _generate_column_patterns_section(self, classes: list[dict], area_unit: str) -> str:
        """Generate the column naming patterns section."""
        # Get first class for examples, default to 'roof'
        example_class = classes[0]["column"] if classes else "roof"
        # Fidelity is only populated on roof / building / building_under_construction;
        # pick a class from this set for the fidelity rows so the example is real.
        # Fall back to "roof" (still illustrative even if not present in the export).
        present_columns = {c["column"] for c in classes}
        fidelity_example = next(
            (c for c in ("roof", "building", "building_under_construction") if c in present_columns),
            "roof",
        )

        # Get primary selection method from export config
        config = self._load_export_config()
        primary_method = config.get("parameters", {}).get("primary_decision") or "optimal"

        # Human-readable descriptions for each method
        method_descriptions = {
            "optimal": "optimal (prioritise geocoded point, falling back to largest intersection area with the parcel)",
            "nearest": "nearest to the property centroid",
            "largest_intersection": "largest intersection area with parcel",
        }
        method_desc = method_descriptions.get(primary_method, primary_method)

        # Detailed algorithm prose per method. Numeric thresholds bind to the
        # actual code constants so the rendered README stays in sync if those
        # defaults ever change.
        small = BUILDING_SMALL_MAX_AREA_SQM
        tol = NEAREST_TOLERANCE_METERS
        method_logic = {
            "optimal": (
                "First tries to select the feature whose footprint contains the input geocoded "
                f"point, or the nearest feature within {tol:g} m of it, preferring features above "
                f"{small:g} m² to avoid sheds and other small outbuildings. If no feature qualifies "
                "(e.g. the point falls in open space or no point was provided), falls back to the "
                "feature with the largest area in the parcel."
            ),
            "nearest": (
                "Selects the feature whose footprint contains the input geocoded point, or the "
                f"nearest feature within {tol:g} m of it, preferring features above {small:g} m² "
                "to avoid sheds and other small outbuildings. Returns no primary feature if "
                "neither condition is met."
            ),
            "largest_intersection": ("Selects the feature with the largest area in the parcel."),
        }
        method_paragraph = method_logic.get(primary_method, "")
        u = area_unit
        u_long = _AREA_UNIT_LONG_NAMES.get(u, u)

        lines = [
            "## Column Naming Patterns",
            "",
            "### Rollup Columns (rollup file)",
            "",
            "For each feature class, the following columns are generated:",
            "",
            "| Pattern | Example | Type | Min | Max | Unit | Description |",
            "|---------|---------|------|-----|-----|------|-------------|",
            f"| `{{class}}_present` | `{example_class}_present` | Y/N | — | — | — | Feature was detected |",
            f"| `{{class}}_count` | `{example_class}_count` | int | 0 | — | — | Number of features detected |",
            f"| `{{class}}_confidence` | `{example_class}_confidence` | float (quantised uint8) | 0.0 | 1.0 | — | Combined confidence score indicating likelihood that any features of this class are present |",
            f"| `{{class}}_total_area_{u}` | `{example_class}_total_area_{u}` | float | 0 | — | {u_long} | Total area of all features |",
            f"| `{{class}}_total_clipped_area_{u}` | `{example_class}_total_clipped_area_{u}` | float | 0 | — | {u_long} | Total area clipped to parcel boundary |",
            f"| `{{class}}_total_unclipped_area_{u}` | `{example_class}_total_unclipped_area_{u}` | float | 0 | — | {u_long} | Total unclipped feature area |",
            f"| `{{class}}_fidelity` | `{fidelity_example}_fidelity` | float | 0.0 | 1.0 | — | Quality of the shape of the vectorized footprint polygon (only for structural classes — building, roof) |",
            "",
            "### Primary Feature Columns",
            "",
            f"The **{method_desc}** feature of each class has additional `primary_` columns.",
            "",
            f"**How the primary feature is selected ({primary_method}):** {method_paragraph}",
            "",
            "| Pattern | Example | Type | Min | Max | Unit | Description |",
            "|---------|---------|------|-----|-----|------|-------------|",
            f"| `primary_{{class}}_area_{u}` | `primary_{example_class}_area_{u}` | float | 0 | — | {u_long} | Area of primary feature |",
            f"| `primary_{{class}}_clipped_area_{u}` | `primary_{example_class}_clipped_area_{u}` | float | 0 | — | {u_long} | Clipped area of primary feature |",
            f"| `primary_{{class}}_confidence` | `primary_{example_class}_confidence` | float (quantised uint8) | 0.0 | 1.0 | — | Calibrated confidence measuring likelihood that the primary feature exists |",
            f"| `primary_{{class}}_feature_id` | `primary_{example_class}_feature_id` | string | — | — | — | Unique ID of primary feature (does not persist across surveys) |",
            f"| `primary_{{class}}_fidelity` | `primary_{fidelity_example}_fidelity` | float | 0.0 | 1.0 | — | Quality of the shape of the vectorized footprint polygon (only for structural classes — building, roof) |",
            "",
        ]

        return "\n".join(lines)

    def _generate_common_columns_section(self) -> str:
        """Generate the common columns section."""
        lines = [
            "## Common Columns",
            "",
            "These columns appear in most output files:",
            "",
            *_render_columns_table(COMMON_COLUMNS.keys()),
            "",
        ]
        return "\n".join(lines)

    def _generate_address_query_section(self, rollup_columns: set[str]) -> str:
        """Generate the address/query columns section."""
        present = [col for col in ADDRESS_QUERY_COLUMNS if col in rollup_columns]
        lines = [
            "## Address & Query Columns",
            "",
            "These columns are present based on the query mode used:",
            "",
            *_render_columns_table(present),
            "",
        ]
        return "\n".join(lines)

    def _generate_dominant_section(self, area_unit: str) -> str:
        """Generate the dominant material/shape summary section."""
        lines = [
            "## Dominant Roof Material & Shape Columns",
            "",
            "These columns summarise the dominant roof material and shape for the primary roof.",
            "",
            "**How dominant is computed.** Each roof is decomposed into material and shape "
            "components (each with a class, area, and ratio of the roof's total area). The "
            "**dominant** component is the one with the highest ratio. For **materials**, the "
            "dominant material is reported only when its ratio is at least 0.5; below that "
            'threshold the column is `unknown`. For **shapes**, the same "highest ratio wins" '
            "rule applies but no ratio threshold is enforced — a shape is dominant if it has the "
            "highest ratio *and* a non-zero area; otherwise `unknown`. (Shape ratios can overlap "
            "or gap and do not sum to 1, which is why no shape ratio column is emitted.) "
            "Deprecated `flat (deprecated)` and `shed` shape classes are excluded from selection.",
            "",
            "### Dominant Material",
            "",
            *_render_columns_table(DOMINANT_ROOF_MATERIAL_COLUMNS.keys(), area_unit),
            "",
            "If no single material has a ratio >= 0.5, the dominant material is reported as `unknown` with null statistics.",
            "",
            "### Dominant Shape",
            "",
            *_render_columns_table(DOMINANT_ROOF_TYPES_COLUMNS.keys(), area_unit),
            "",
            "If all shape components have zero area, the dominant shape is reported as `unknown` with null statistics.",
            "Shape does not include a ratio column because roof shapes can overlap or gap, so ratios do not sum to 1.",
            "",
        ]
        return "\n".join(lines)

    def _generate_rsi_section(self) -> str:
        """Generate the Roof Spotlight Index section."""
        lines = [
            "## Roof Spotlight Index (RSI) Columns",
            "",
            "RSI provides a roof condition assessment score:",
            "",
            *_render_columns_table(RSI_COLUMNS.keys()),
        ]

        lines.append(
            """
**RSI Score Interpretation:**
- **91-100**: Very good
- **81-90**: Good
- **71-80**: Moderate
- **61-70**: Fair
- **51-60**: Low
- **26-50**: Poor
- **0-25**: Needs replacement

The `roof_spotlight_index` columns in this export contain the **resolved best RSI** per roof —
using the roof's own score when available, or falling back to the building lifecycle score when
structural damage is present. When structural damage causes holes in the roof polygon, the API
calculates RSI on the building lifecycle polygon instead to avoid falsely inflating the score.

This fallback only applies when building lifecycle features are included in the export
(e.g. via `--packs building damage_non_postcat`). Without building lifecycle data, only
roofs without structural damage will have an RSI value.

For more details, see: https://help.nearmap.com/kb/articles/1641-nearmap-roof-spotlight-index-rsi
"""
        )

        return "\n".join(lines)

    def _generate_roof_age_section(self) -> str:
        """Generate the Roof Age columns section."""
        lines = [
            "## Roof Age Columns",
            "",
            "These columns are present when roof age estimation was requested:",
            "",
        ]

        # Pull dataset / cutoff selection from export_config.json so the README is
        # self-describing about *which* roof age dataset was queried and whether a
        # historical cutoff was applied.
        config = self._load_export_config()
        params = config.get("parameters", {}) if isinstance(config, dict) else {}
        dataset = params.get("roof_age_dataset")
        resource_id = params.get("roof_age_resource_id")
        until = params.get("until")
        since = params.get("since")
        if dataset or resource_id or until or since:
            lines.append("**Dataset and historical query for this export:**")
            lines.append("")
            if dataset is not None:
                lines.append(f"- `roof_age_dataset`: `{dataset}`")
            if resource_id is not None and resource_id != dataset:
                lines.append(f"- Resolved resource id: `{resource_id}`")
            if until is not None:
                lines.append(
                    f"- `untilAsOfDate` cutoff: `{until}` "
                    "(per-AOI `until` column values, when present, override this for each row)"
                )
            if since is not None:
                lines.append(
                    f"- `sinceAsOfDate` cutoff: `{since}` "
                    "(per-AOI `since` column values, when present, override this for each row)"
                )
            if until is None and since is None:
                lines.append("- No historical cutoff applied — responses reflect the dataset's most recent snapshot.")
            lines.append("")
            lines.append(
                "Note: cutoff parameters are not supported on the A.0 dataset. The `model_version` "
                "column in each row records the actual model that produced that record — for resilient "
                "downstream code, prefer `model_version` over the dataset alias when reasoning about "
                "model behaviour."
            )
            lines.append("")

        lines.extend(_render_columns_table(ROOF_AGE_COLUMNS.keys()))

        # Evidence Type legend — sourced from column_metadata.json so the README
        # and any downstream consumers see the same descriptions.
        legend = evidence_type_legend()
        lines.append("")
        lines.append("**Evidence Type code → description:**")
        lines.append("")
        for code in sorted(legend, key=int):
            lines.append(f"- **Type {code}**: {legend[code]}")
        lines.append("")
        lines.append(
            "Higher evidence types indicate more robust information sources. "
            "**Trust Score (0-100)** quantifies reliability based on the same evidence "
            "quality factors (imagery, permits, change detection, capture count)."
        )
        lines.append("")
        lines.append("For more details, see:")
        lines.append("- https://help.nearmap.com/kb/articles/1810-nearmap-roof-age")
        lines.append("- https://help.nearmap.com/kb/articles/1811-evidence-type-and-trust-score")
        lines.append("")

        return "\n".join(lines)

    def _generate_defensible_space_section(self, area_unit: str) -> str:
        """Generate the Defensible Space columns section."""
        u = area_unit
        # Zone 0 is shown as a representative example; the API returns zones 0/1/2.
        example_columns = [
            f"primary_defensible_space_zone_0_zone_area_{u}",
            f"primary_defensible_space_zone_0_defensible_space_area_{u}",
            "primary_defensible_space_zone_0_coverage_ratio",
            f"primary_defensible_space_zone_0_risk_object_area_{u}",
            f"primary_defensible_space_zone_0_medium_and_high_vegetation_with_woody_vegetation_area_{u}",
            "primary_defensible_space_zone_0_medium_and_high_vegetation_with_woody_vegetation_ratio",
            f"primary_defensible_space_zone_0_roof_area_{u}",
            "primary_defensible_space_zone_0_roof_ratio",
            f"primary_defensible_space_zone_0_yard_debris_area_{u}",
            "primary_defensible_space_zone_0_yard_debris_ratio",
            "defensible_space_model_version",
        ]
        lines = [
            "## Defensible Space Columns",
            "",
            "Defensible space data describes the area around structures in concentric zones,",
            "measuring how much clear (defensible) space exists versus risk objects like vegetation and debris.",
            "",
            "For zone definitions (boundary distances, class descriptions) and methodology, see the",
            "authoritative Nearmap documentation:",
            "[Defensible Space Overview](https://help.nearmap.com/kb/articles/1853-defensible-space-overview).",
            "",
            "### Column Prefixes",
            "",
            "Two sets of defensible space columns are produced, one row per zone returned by the API",
            "(the `{N}` in each prefix is the `zoneId` from the API response):",
            "",
            "| Prefix | Scope |",
            "|--------|-------|",
            "| `primary_defensible_space_zone_{N}_` | Defensible space for the primary roof feature only |",
            "| `aggregate_defensible_space_zone_{N}_` | Defensible space aggregated across the entire parcel (all structures) |",
            "",
            "### Columns Per Zone (illustrated for zone 0; analogous columns exist for zones 1, 2, etc.)",
            "",
            *_render_columns_table(example_columns, u),
        ]

        lines.append(
            """
**Risk object classes** (vegetation, roof, yard debris) are always present with 0.0 defaults
when a class is not detected in a given zone. This ensures consistent column presence across all rows.

**Coverage ratio** indicates what fraction of the zone is clear/defensible space.
A ratio of 1.0 means the entire zone is defensible; 0.0 means it is fully occupied by risk objects.
"""
        )

        return "\n".join(lines)

    def _generate_data_notes(self, area_unit: str) -> str:
        """Generate the data notes section."""
        if area_unit == "sqm":
            area_desc = "Square meters (sqm)"
        else:
            area_desc = "Square feet (sqft)"

        return f"""## Data Notes

- **Coordinates**: WGS84 (EPSG:4326)
- **Areas**: {area_desc}
- **Confidence scores**: Range from 0.0 (low) to 1.0 (high)
- **Dates**: ISO 8601 format (YYYY-MM-DD)
- **GeoParquet files**: Can be opened with QGIS, ArcGIS, or Python (geopandas)
- **`*_skipped` columns**: The Feature API may return `{{skipped: true}}` for an include section (e.g. RSI, peril scores, defensible space) when it decides not to evaluate that section for a given row. nmaipy surfaces this as a per-include `*_skipped` boolean — `Y` means "API declined to evaluate", `N` means evaluated normally, absent means the include wasn't requested. When a row has `*_skipped = Y`, the corresponding score / area / ratio columns will be null because the API didn't compute them, not because the underlying feature is absent.

"""

    def _generate_footer(self) -> str:
        """Generate the README footer."""
        return """---

*Generated by nmaipy*
"""
