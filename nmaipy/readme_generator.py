"""Dynamic README generator for nmaipy exports.

Generates customer-facing documentation based on actual files and columns
present in an export directory, rather than using a static template.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from nmaipy import storage
from nmaipy.__version__ import __version__

# Known output filenames mapped to descriptions.
FILE_PATTERNS = {
    "rollup.csv": "Property-level summary with one row per property containing aggregated statistics",
    "rollup.parquet": "Property-level summary with one row per property containing aggregated statistics (Parquet format)",
    "features.parquet": "All detected features with geometry (GeoParquet)",
    "feature_api_errors.csv": "Properties where Feature API calls failed",
    "feature_api_errors.parquet": "Feature API errors with geometry (GeoParquet)",
    "roof_age_errors.csv": "Properties where Roof Age API calls failed",
    "roof_age_errors.parquet": "Roof Age API errors with geometry (GeoParquet)",
    "latency_stats.csv": "API call timing statistics",
    "buildings.csv": "Building-level summary data",
    "buildings.parquet": "Building-level summary data (Parquet format)",
}

# Files to exclude from documentation
EXCLUDE_FILES = {"README.md", ".DS_Store", "export_config.json"}

# Common columns always present
COMMON_COLUMNS = {
    "aoi_id": "Property identifier matching original input row order (0-indexed). If your input had an aoi_id column, it is preserved; otherwise generated from row index.",
    "mesh_date": "Date of AI model processing (YYYY-MM-DD)",
    "system_version": "AI model version used for detection",
    "link": "URL to view property in Nearmap MapBrowser",
}

# Address and query columns produced by nmaipy
ADDRESS_QUERY_COLUMNS = {
    "streetAddress": "Street address from input (used for address-based API queries)",
    "city": "City from input (used for address-based API queries)",
    "state": "State from input (used for address-based API queries)",
    "zip": "ZIP/postal code from input (used for address-based API queries)",
    "query_aoi_lat": "Representative latitude of the query AOI geometry",
    "query_aoi_lon": "Representative longitude of the query AOI geometry",
    "lat": "Input latitude used for primary feature selection",
    "lon": "Input longitude used for primary feature selection",
}

# Roof Spotlight Index columns
RSI_COLUMNS = {
    "roof_spotlight_index": "Roof condition score (0-100, higher = better condition)",
    "roof_spotlight_index_confidence": "Confidence in RSI score (0.0-1.0)",
}

# Roof Age columns
# Dominant material/shape summary columns
DOMINANT_ROOF_MATERIAL_COLUMNS = {
    "dominant_roof_material_feature_class": "Feature class UUID of the dominant roof material",
    "dominant_roof_material_description": "Snake_case name of the dominant material (or 'unknown' if ratio < 0.5)",
    "dominant_roof_material_area_{unit}": "Area of the dominant material component",
    "dominant_roof_material_ratio": "Ratio of the dominant material relative to total roof area (0.0-1.0)",
    "dominant_roof_material_confidence": "Confidence in the dominant material classification (0.0-1.0)",
}

DOMINANT_ROOF_TYPES_COLUMNS = {
    "dominant_roof_types_feature_class": "Feature class UUID of the dominant roof shape",
    "dominant_roof_types_description": "Snake_case name of the dominant shape (or 'unknown' if no detected area)",
    "dominant_roof_types_area_{unit}": "Area of the dominant shape component",
    "dominant_roof_types_confidence": "Confidence in the dominant shape classification (0.0-1.0)",
}

ROOF_AGE_COLUMNS = {
    "roof_age_installation_date": "Estimated roof installation date (YYYY-MM-DD)",
    "roof_age_as_of_date": "Reference date for age calculation",
    "roof_age_years_as_of_date": "Calculated roof age in years",
    "roof_age_trust_score": "Reliability of age estimate (0-100, higher = more reliable)",
    "roof_age_evidence_type": "How age was determined (numeric code)",
    "roof_age_evidence_type_description": "Human-readable evidence description",
    "roof_age_before_installation_capture_date": "Last capture before roof installation",
    "roof_age_after_installation_capture_date": "First capture after roof installation",
    "roof_age_min_capture_date": "Earliest imagery capture date analyzed",
    "roof_age_max_capture_date": "Latest imagery capture date analyzed",
    "roof_age_number_of_captures": "Number of aerial captures analyzed",
    "roof_age_map_browser_url": "URL to view roof age timeline in MapBrowser",
    "roof_age_model_version": "Version of the roof age prediction model used",
    "roof_age_kind": "Type of estimate: 'roof' (section) or 'parcel' (property-level)",
    "roof_age_relevant_permits": "Whether relevant building permits were found (Y/N)",
    "roof_age_relevant_permits_details": "Details of relevant building permits (JSON, present when permits found)",
    "roof_age_assessor_data": "Whether assessor records were found (Y/N)",
    "roof_age_assessor_data_details": "Details of assessor records (JSON, present when records found)",
}


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

        Returns:
            Path to generated README.md file (string, may be S3 URI).
        """
        content = self._generate()
        readme_path = storage.join_path(self.final_dir, "README.md")
        with storage.open_file(readme_path, "w") as f:
            f.write(content)
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
        area_unit = self._detect_area_unit()

        sections = []
        sections.append(self._generate_header())
        sections.append(self._generate_files_table(files))
        sections.append(self._generate_classes_section(classes))
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

        sections.append(self._generate_data_notes(area_unit))
        sections.append(self._generate_footer())

        return "\n".join(sections)

    def _discover_files(self) -> list[str]:
        """Discover all files in the final/ directory, excluding certain files."""
        all_files = storage.glob_files(self.final_dir, "*")
        files = []
        for f in sorted(all_files):
            name = storage.basename(f)
            if name not in EXCLUDE_FILES:
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
            "buildings",
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
        """Check if dominant material/shape summary columns are present."""
        return any(c.startswith("dominant_roof_material_") or c.startswith("dominant_roof_types_") for c in columns)

    def _has_rsi_columns(self, columns: set[str]) -> bool:
        """Check if Roof Spotlight Index columns are present."""
        return any("roof_spotlight_index" in c for c in columns)

    def _has_roof_age_columns(self, columns: set[str]) -> bool:
        """Check if roof age columns are present."""
        return any("roof_age_" in c for c in columns)

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
        """Get description for a file based on exact filename or pattern matching."""
        if filename in FILE_PATTERNS:
            return FILE_PATTERNS[filename]

        # Per-class feature geometry files: {class}_features.parquet
        if filename.endswith("_features.parquet"):
            class_name = filename[: -len("_features.parquet")].replace("_", " ").title()
            return f"{class_name} polygons with geometry (GeoParquet)"

        # Per-class attribute files: {class}.csv or {class}.parquet
        if filename.endswith(".csv"):
            class_name = filename[: -len(".csv")].replace("_", " ").title()
            return f"Per-{class_name.lower()} data with feature attributes"

        if filename.endswith(".parquet") and not filename.endswith("_features.parquet"):
            class_name = filename[: -len(".parquet")].replace("_", " ").title()
            return f"Per-{class_name.lower()} data with feature attributes (Parquet format)"

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

    def _generate_column_patterns_section(self, classes: list[dict], area_unit: str) -> str:
        """Generate the column naming patterns section."""
        # Get first class for examples, default to 'roof'
        example_class = classes[0]["column"] if classes else "roof"

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
        u = area_unit

        lines = [
            "## Column Naming Patterns",
            "",
            "### Rollup Columns (rollup file)",
            "",
            "For each feature class, the following columns are generated:",
            "",
            "| Pattern | Example | Description |",
            "|---------|---------|-------------|",
            f"| `{{class}}_present` | `{example_class}_present` | Y/N - feature was detected |",
            f"| `{{class}}_count` | `{example_class}_count` | Number of features detected |",
            f"| `{{class}}_confidence` | `{example_class}_confidence` | Combined confidence score (0.0-1.0) |",
            f"| `{{class}}_total_area_{u}` | `{example_class}_total_area_{u}` | Total area of all features |",
            f"| `{{class}}_total_clipped_area_{u}` | `{example_class}_total_clipped_area_{u}` | Total area clipped to parcel boundary |",
            f"| `{{class}}_total_unclipped_area_{u}` | `{example_class}_total_unclipped_area_{u}` | Total unclipped feature area |",
            "",
            "### Primary Feature Columns",
            "",
            f"The **{method_desc}** feature of each class has additional `primary_` columns:",
            "",
            "| Pattern | Example | Description |",
            "|---------|---------|-------------|",
            f"| `primary_{{class}}_area_{u}` | `primary_{example_class}_area_{u}` | Area of primary feature |",
            f"| `primary_{{class}}_clipped_area_{u}` | `primary_{example_class}_clipped_area_{u}` | Clipped area of primary feature |",
            f"| `primary_{{class}}_confidence` | `primary_{example_class}_confidence` | Confidence of primary feature |",
            f"| `primary_{{class}}_feature_id` | `primary_{example_class}_feature_id` | Unique ID of primary feature |",
            f"| `primary_{{class}}_fidelity` | `primary_{example_class}_fidelity` | Detection fidelity score |",
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
            "| Column | Description |",
            "|--------|-------------|",
        ]

        for col, desc in COMMON_COLUMNS.items():
            lines.append(f"| `{col}` | {desc} |")

        lines.append("")
        return "\n".join(lines)

    def _generate_address_query_section(self, rollup_columns: set[str]) -> str:
        """Generate the address/query columns section."""
        lines = [
            "## Address & Query Columns",
            "",
            "These columns are present based on the query mode used:",
            "",
            "| Column | Description |",
            "|--------|-------------|",
        ]

        for col, desc in ADDRESS_QUERY_COLUMNS.items():
            if col in rollup_columns:
                lines.append(f"| `{col}` | {desc} |")

        lines.append("")
        return "\n".join(lines)

    def _generate_dominant_section(self, area_unit: str) -> str:
        """Generate the dominant material/shape summary section."""
        lines = [
            "## Dominant Roof Material & Shape Columns",
            "",
            "These columns summarise the dominant (largest area) roof material and shape for the primary roof:",
            "",
            "### Dominant Material",
            "",
            "| Column | Description |",
            "|--------|-------------|",
        ]

        for col, desc in DOMINANT_ROOF_MATERIAL_COLUMNS.items():
            col_name = col.replace("{unit}", area_unit)
            lines.append(f"| `{col_name}` | {desc.replace('{unit}', area_unit)} |")

        lines.extend(
            [
                "",
                "If no single material has a ratio >= 0.5, the dominant material is reported as `unknown` with null statistics.",
                "",
                "### Dominant Shape",
                "",
                "| Column | Description |",
                "|--------|-------------|",
            ]
        )

        for col, desc in DOMINANT_ROOF_TYPES_COLUMNS.items():
            col_name = col.replace("{unit}", area_unit)
            lines.append(f"| `{col_name}` | {desc.replace('{unit}', area_unit)} |")

        lines.extend(
            [
                "",
                "If all shape components have zero area, the dominant shape is reported as `unknown` with null statistics.",
                "Shape does not include a ratio column because roof shapes can overlap or gap, so ratios do not sum to 1.",
                "",
            ]
        )

        return "\n".join(lines)

    def _generate_rsi_section(self) -> str:
        """Generate the Roof Spotlight Index section."""
        lines = [
            "## Roof Spotlight Index (RSI) Columns",
            "",
            "RSI provides a roof condition assessment score:",
            "",
            "| Column | Description |",
            "|--------|-------------|",
        ]

        for col, desc in RSI_COLUMNS.items():
            lines.append(f"| `{col}` | {desc} |")

        lines.append(
            """
**RSI Score Interpretation:**
- **81-100**: Excellent condition (minimal/no visible defects)
- **61-80**: Good condition
- **41-60**: Fair condition
- **21-40**: Poor condition
- **0-20**: Very poor condition (severe defects)

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
            "| Column | Description |",
            "|--------|-------------|",
        ]

        for col, desc in ROOF_AGE_COLUMNS.items():
            lines.append(f"| `{col}` | {desc} |")

        lines.append(
            """
**Evidence Type (0-8):**
- **Type 8**: Multiple clear images with detected roof change, plus corroborating permits/assessor data
- **Type 7**: Multiple clear images with detected roof change, no corroborating data
- **Type 6**: Multiple clear images, no change detected but other strong evidence
- **Type 2**: No clear imagery, but building permits or assessor records available
- **Type 0**: Minimal supporting evidence

Higher evidence types indicate more robust information sources.

**Trust Score (0-100):**
- Higher scores indicate more reliable estimates
- Based on evidence quality (permits, imagery change detection, number of captures)

For more details, see:
- https://help.nearmap.com/kb/articles/1810-nearmap-roof-age
- https://help.nearmap.com/kb/articles/1811-evidence-type-and-trust-score
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

"""

    def _generate_footer(self) -> str:
        """Generate the README footer."""
        return """---

*Generated by nmaipy*
"""
