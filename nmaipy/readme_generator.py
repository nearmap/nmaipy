"""Dynamic README generator for nmaipy exports.

Generates customer-facing documentation based on actual files and columns
present in an export directory, rather than using a static template.
"""

import json
from datetime import datetime
from pathlib import Path

from nmaipy.__version__ import __version__

# File patterns (suffixes) mapped to descriptions.
# The README displays full filenames (e.g., "parcels_aoi_rollup.csv"),
# these patterns are used to match and describe them.
FILE_PATTERNS = {
    "_aoi_rollup.csv": "Property-level summary with one row per property containing aggregated statistics",
    "_aoi_rollup.parquet": "Property-level summary with one row per property containing aggregated statistics (Parquet format)",
    "_building.csv": "Per-building data with building-specific attributes",
    "_building_features.parquet": "Building polygons with geometry (GeoParquet)",
    "_roof.csv": "Per-roof data with roof condition and optional child roof age",
    "_roof_features.parquet": "Roof polygons with geometry (GeoParquet)",
    "_roof_instance.csv": "Per-roof-instance data for roof age analysis",
    "_roof_instance_features.parquet": "Roof instance polygons with geometry (GeoParquet)",
    "_features.parquet": "All detected features with geometry (GeoParquet)",
    "_feature_api_errors.csv": "Properties where Feature API calls failed",
    "_feature_api_errors.parquet": "Feature API errors with geometry (GeoParquet)",
    "_roof_age_errors.csv": "Properties where Roof Age API calls failed",
    "_roof_age_errors.parquet": "Roof Age API errors with geometry (GeoParquet)",
    "_latency_stats.csv": "API call timing statistics",
    "_buildings.csv": "Building-level summary data",
    "_buildings.parquet": "Building-level summary data (Parquet format)",
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
    "roof_age_mapbrowser_url": "URL to view roof age timeline in MapBrowser",
    "roof_age_kind": "Type of estimate: 'roof' (section) or 'parcel' (property-level)",
}


class ReadmeGenerator:
    """Generates dynamic README.md based on actual export files and columns."""

    def __init__(self, output_dir: Path):
        """
        Initialize the README generator.

        Args:
            output_dir: Path to export output directory. Can be either:
                - The parent directory containing a final/ subdirectory
                - The final directory itself containing the export files
        """
        self.output_dir = Path(output_dir)

        # Determine the actual final directory
        # Check if output_dir/final exists, otherwise assume output_dir is the final dir
        potential_final = self.output_dir / "final"
        if potential_final.exists() and potential_final.is_dir():
            self.final_dir = potential_final
        else:
            self.final_dir = self.output_dir

    def _load_export_config(self) -> dict:
        """Load export_config.json if it exists."""
        config_path = self.final_dir / "export_config.json"
        if config_path.exists():
            try:
                return json.loads(config_path.read_text())
            except Exception:
                pass
        return {}

    def generate_and_save(self) -> Path:
        """
        Generate README and save to final directory.

        Returns:
            Path to generated README.md file.
        """
        content = self._generate()
        readme_path = self.final_dir / "README.md"
        readme_path.write_text(content)
        return readme_path

    def _generate(self) -> str:
        """Generate the README markdown content."""
        files = self._discover_files()
        prefix = self._get_file_prefix(files)
        classes = self._detect_classes(files, prefix)
        rollup_columns = self._get_rollup_columns(files, prefix)

        has_address_query = self._has_address_query_columns(rollup_columns)
        has_rsi = self._has_rsi_columns(rollup_columns)
        has_roof_age = self._has_roof_age_columns(rollup_columns)
        area_unit = self._detect_area_unit()

        sections = []
        sections.append(self._generate_header())
        sections.append(self._generate_files_table(files, prefix))
        sections.append(self._generate_classes_section(classes))
        sections.append(self._generate_column_patterns_section(classes, area_unit))
        sections.append(self._generate_common_columns_section())

        if has_address_query:
            sections.append(self._generate_address_query_section(rollup_columns))
        if has_rsi:
            sections.append(self._generate_rsi_section())
        if has_roof_age:
            sections.append(self._generate_roof_age_section())

        sections.append(self._generate_data_notes(area_unit))
        sections.append(self._generate_footer())

        return "\n".join(sections)

    def _discover_files(self) -> list[Path]:
        """Discover all files in the final/ directory, excluding certain files."""
        if not self.final_dir.exists():
            return []
        files = []
        for f in sorted(self.final_dir.iterdir()):
            if f.is_file() and f.name not in EXCLUDE_FILES:
                files.append(f)
        return files

    def _get_file_prefix(self, files: list[Path]) -> str:
        """Detect the common filename prefix (e.g., 'parcels_')."""
        for f in files:
            if f.name.endswith("_aoi_rollup.csv"):
                return f.stem.replace("_aoi_rollup", "") + "_"
            if f.name.endswith("_aoi_rollup.parquet"):
                return f.stem.replace("_aoi_rollup", "") + "_"
        # Fallback: try to find common prefix from CSV files
        csv_files = [f for f in files if f.suffix == ".csv"]
        if csv_files:
            for f in csv_files:
                if "_roof.csv" in f.name:
                    return f.name.replace("roof.csv", "")
                if "_building.csv" in f.name:
                    return f.name.replace("building.csv", "")
        return ""

    def _detect_classes(self, files: list[Path], prefix: str) -> list[dict]:
        """
        Detect feature classes from filenames.

        Returns list of dicts with 'name' (display) and 'column' (snake_case).
        """
        classes = []
        seen = set()

        for f in files:
            if f.suffix != ".csv":
                continue
            name = f.stem
            if prefix:
                name = name.replace(prefix.rstrip("_"), "", 1).lstrip("_")

            # Skip non-class files
            skip_patterns = [
                "aoi_rollup",
                "feature_api_errors",
                "roof_age_errors",
                "latency_stats",
                "buildings",
            ]
            if any(p in name for p in skip_patterns):
                continue

            # Skip parquet companion files
            if name.endswith("_features"):
                continue

            # Convert to display name
            column_name = name
            display_name = name.replace("_", " ").title()

            if column_name not in seen:
                seen.add(column_name)
                classes.append({"name": display_name, "column": column_name})

        return classes

    def _get_rollup_columns(self, files: list[Path], prefix: str) -> set[str]:
        """Get column names from the rollup CSV file."""
        rollup_file = None
        for f in files:
            if f.name.endswith("_aoi_rollup.csv"):
                rollup_file = f
                break

        if not rollup_file or not rollup_file.exists():
            return set()

        with open(rollup_file, "r") as fh:
            header = fh.readline().strip()
            return set(header.split(","))

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

    def _generate_files_table(self, files: list[Path], prefix: str) -> str:
        """Generate the files table section."""
        lines = ["## Files in This Export", "", "| File Name | Description |", "|-----------|-------------|"]

        for f in files:
            description = self._get_file_description(f.name, prefix)
            lines.append(f"| `{f.name}` | {description} |")

        lines.append("")
        return "\n".join(lines)

    def _get_file_description(self, filename: str, prefix: str) -> str:
        """Get description for a file based on pattern matching."""
        # Check exact matches first
        if filename in FILE_PATTERNS:
            return FILE_PATTERNS[filename]

        # Check suffix patterns
        for pattern, description in FILE_PATTERNS.items():
            if pattern.startswith("_") and filename.endswith(pattern):
                return description

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
            "### Rollup Columns (aoi_rollup file)",
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
