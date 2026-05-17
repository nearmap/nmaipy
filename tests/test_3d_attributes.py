"""Test 3D attribute flattening for features."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from nmaipy.exporter import AOIExporter
from nmaipy.feature_api import FeatureApi


@pytest.fixture
def test_output_dir():
    """Fixture to create and clean up test output directory."""
    # Use a temporary directory for CI/CD, but a known location for local debugging
    if os.environ.get("CI"):
        # In CI, use temp directory
        temp_dir = tempfile.mkdtemp(prefix="3d_test_")
        output_dir = Path(temp_dir)
    else:
        # Locally, use tests/data for easier debugging
        output_dir = Path(__file__).parent / "data" / "3d_test_output"
        output_dir.mkdir(parents=True, exist_ok=True)

    yield output_dir

    # Cleanup after test
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def salt_lake_aoi(test_output_dir):
    """Fixture to create test AOI for Salt Lake City."""
    test_polygon = Polygon(
        [
            (-111.8905, 40.7610),
            (-111.8903, 40.7610),
            (-111.8903, 40.7612),
            (-111.8905, 40.7612),
            (-111.8905, 40.7610),
        ]
    )

    test_aoi = gpd.GeoDataFrame({"aoi_id": ["salt_lake_3d_test"], "geometry": [test_polygon]}, crs="EPSG:4326")

    aoi_file = test_output_dir / "test_aoi.geojson"
    test_aoi.to_file(aoi_file, driver="GeoJSON")

    return aoi_file


@pytest.fixture
def salt_lake_aoi_large(test_output_dir):
    """A ~1.2km x 1.2km AOI centered on (40.7611, -111.8904).

    Area ≈ 1.44 sqkm, comfortably above MAX_AOI_AREA_SQM_BEFORE_GRIDDING (1 sqkm),
    so any export of this AOI triggers gridding (~3x3 = 9 cells at the default
    GRID_SIZE_DEGREES). Used to exercise the gridded prefer3d code path that the
    unit-test suite (single-AOI, no gridding) doesn't reach.
    """
    test_polygon = Polygon(
        [
            (-111.8975, 40.7557),
            (-111.8833, 40.7557),
            (-111.8833, 40.7665),
            (-111.8975, 40.7665),
            (-111.8975, 40.7557),
        ]
    )

    test_aoi = gpd.GeoDataFrame({"aoi_id": ["salt_lake_3d_test_large"], "geometry": [test_polygon]}, crs="EPSG:4326")

    aoi_file = test_output_dir / "test_aoi_large.geojson"
    test_aoi.to_file(aoi_file, driver="GeoJSON")

    return aoi_file


@pytest.mark.integration
@pytest.mark.live_api
def test_3d_attributes_flattening(test_output_dir, salt_lake_aoi):
    """Test that 3D attributes are properly flattened in feature exports."""

    # Run exporter with 3D packs
    exporter = AOIExporter(
        aoi_file=str(salt_lake_aoi),
        output_dir=str(test_output_dir),
        country="us",
        packs=["building", "building_char", "roof_char"],
        save_features=True,
        no_cache=True,
        processes=1,
        only3d=True,
    )

    exporter.run()

    # Check the features file
    features_file = test_output_dir / "final" / "features.parquet"
    assert features_file.exists(), "Features file should be created"

    # Load and verify
    gdf = gpd.read_parquet(features_file)

    # Check that we have features
    assert len(gdf) > 0, "Should have features in the output"

    # Check for flattened columns
    flattened_cols = [c for c in gdf.columns if "." in c]
    assert len(flattened_cols) > 0, "Should have flattened attribute columns"

    # Check that internalClassId is not present
    internal_cols = [c for c in gdf.columns if "internalClassId" in c.lower()]
    assert len(internal_cols) == 0, "internalClassId should be skipped"

    # Check for components serialized as JSON
    component_cols = [c for c in flattened_cols if "components" in c]
    if component_cols:
        for col in component_cols:
            non_null_values = gdf[gdf[col].notna()][col]
            if len(non_null_values) > 0:
                # Check that components are JSON strings
                sample = non_null_values.iloc[0]
                assert isinstance(sample, str), f"{col} should be a string"
                # Try to parse as JSON
                try:
                    parsed = json.loads(sample)
                    assert isinstance(parsed, list), f"{col} should be a JSON array"
                except json.JSONDecodeError:
                    pytest.fail(f"{col} should be valid JSON")

    # Check for expected 3D attributes on buildings
    building_rows = gdf[gdf["description"] == "Building"]
    if len(building_rows) > 0:
        # Look for height attributes
        height_cols = [c for c in flattened_cols if "height" in c.lower()]
        if height_cols:
            # At least one building should have height
            has_height = any(building_rows[col].notna().any() for col in height_cols)
            assert has_height or len(building_rows) < 2, "At least some buildings should have height attributes"

    # Check for roof attributes
    roof_rows = gdf[gdf["description"] == "Roof"]
    if len(roof_rows) > 0:
        # Look for pitch or material attributes
        roof_attr_cols = [c for c in flattened_cols if any(term in c.lower() for term in ["pitch", "material", "type"])]
        assert len(roof_attr_cols) > 0 or len(roof_rows) < 2, "Roofs should have attributes like pitch or material"


@pytest.mark.integration
@pytest.mark.live_api
def test_prefer3d_end_to_end_3d_path(test_output_dir, salt_lake_aoi):
    """End-to-end run with --prefer3d against an AOI/window known to have 3D coverage.

    The salt_lake_aoi (~40.7611, -111.8904) has a confirmed 3D capture on 2025-09-03;
    with no since/until restriction, the latest survey for this point is 3D, so the
    rollup row should carry a non-empty mesh_date and the 2D fallback path is NOT
    exercised here (covered separately by `test_prefer3d_end_to_end_2d_fallback`).
    """
    exporter = AOIExporter(
        aoi_file=str(salt_lake_aoi),
        output_dir=str(test_output_dir),
        country="us",
        packs=["building"],
        save_features=True,
        no_cache=True,
        processes=1,
        prefer3d=True,
    )

    exporter.run()

    rollup_files = list((test_output_dir / "final").glob("rollup.*"))
    assert rollup_files, "rollup file should be created"
    rollup_path = rollup_files[0]
    rollup = pd.read_parquet(rollup_path) if rollup_path.suffix == ".parquet" else pd.read_csv(rollup_path)

    assert "mesh_date" in rollup.columns, "rollup must expose the mesh_date metadata column"
    assert len(rollup) == 1
    mesh_date = rollup["mesh_date"].iloc[0]
    assert (
        mesh_date and str(mesh_date) != "" and str(mesh_date).lower() != "nan"
    ), f"latest survey at this AOI is the 2025-09-03 3D capture; mesh_date should be populated, got {mesh_date!r}"


@pytest.mark.integration
@pytest.mark.live_api
def test_prefer3d_end_to_end_2d_fallback(test_output_dir, salt_lake_aoi):
    """End-to-end run with --prefer3d against a date window that has only 2D coverage.

    At ~40.7611, -111.8904 the 2025-09-03 capture is 3D and the next survey (2026-03-04)
    is 2D-only. The window [2025-09-04, 2026-03-31] excludes the 3D survey and includes
    the 2D-only one, so the 3D-only first pass must 404 and the AOI must be resolved
    via the 2D fallback. mesh_date should be empty in the resulting rollup.
    """
    exporter = AOIExporter(
        aoi_file=str(salt_lake_aoi),
        output_dir=str(test_output_dir),
        country="us",
        packs=["building"],
        save_features=True,
        no_cache=True,
        processes=1,
        prefer3d=True,
        since="2025-09-04",
        until="2026-03-31",
    )

    exporter.run()

    rollup_files = list((test_output_dir / "final").glob("rollup.*"))
    assert rollup_files, "rollup file should be created"
    rollup_path = rollup_files[0]
    rollup = pd.read_parquet(rollup_path) if rollup_path.suffix == ".parquet" else pd.read_csv(rollup_path)

    assert "mesh_date" in rollup.columns
    assert len(rollup) == 1, "single AOI should produce a single rollup row even via the 2D fallback"
    mesh_date = rollup["mesh_date"].iloc[0]
    # Empty string from a 2D survey, or NaN if the metadata-level mesh_date was dropped
    # in favour of an absent per-feature mesh_date — either is a valid 2D signal.
    is_empty = (mesh_date == "") or (pd.isna(mesh_date))
    assert is_empty, f"window contains only 2D coverage; mesh_date should be empty, got {mesh_date!r}"

    # And the 2026-03-04 2D survey should be the one used.
    assert "survey_date" in rollup.columns
    survey_date = str(rollup["survey_date"].iloc[0])
    assert survey_date.startswith("2026-03-04"), f"expected the 2026-03-04 2D survey, got survey_date={survey_date!r}"


@pytest.mark.integration
@pytest.mark.live_api
def test_prefer3d_end_to_end_gridded(test_output_dir, salt_lake_aoi_large):
    """End-to-end gridded prefer3d run on a ~1.44 sqkm Salt Lake AOI.

    Exercises the gridded prefer3d code path that the unit-test suite (all single
    AOIs, no gridding) doesn't reach — gridded sub-requests bypassing the prefer3d
    dispatch via `in_gridding_mode=True`, the multi-cell metadata reconstruction in
    `_attempt_gridding`, and the merged-rollup write through `exporter._process_chunk`.

    Uses the 2D-only date window from `test_prefer3d_end_to_end_2d_fallback` so the
    3D-only first pass produces a wholesale grid failure for the AOI and the entire
    bulk falls back to a gridded 2D run. This catches regressions in the broader
    gridded prefer3d flow.

    Note: this does NOT exercise the specific partial-3D-coverage case that the
    unit-test `test_prefer3d_dedupes_partial_gridded_aoi_across_passes` covers —
    reproducing it live would require a region known to have mixed coverage AND
    `aoi_grid_min_pct < 100`, which we don't have a reliable fixture for.
    """
    exporter = AOIExporter(
        aoi_file=str(salt_lake_aoi_large),
        output_dir=str(test_output_dir),
        country="us",
        packs=["building"],
        save_features=True,
        no_cache=True,
        processes=1,
        prefer3d=True,
        since="2025-09-04",
        until="2026-03-31",
    )

    exporter.run()

    rollup_files = list((test_output_dir / "final").glob("rollup.*"))
    assert rollup_files, "rollup file should be created"
    rollup_path = rollup_files[0]
    rollup = pd.read_parquet(rollup_path) if rollup_path.suffix == ".parquet" else pd.read_csv(rollup_path)

    assert "mesh_date" in rollup.columns
    assert len(rollup) == 1, "single gridded AOI should produce a single rollup row, not duplicates"
    # 2D-only window → mesh_date must be empty (no 3D survey could be returned).
    mesh_date = rollup["mesh_date"].iloc[0]
    is_empty = (mesh_date == "") or pd.isna(mesh_date)
    assert is_empty, f"window contains only 2D coverage; mesh_date should be empty, got {mesh_date!r}"


@pytest.mark.integration
@pytest.mark.live_api
def test_attributes_list_handling():
    """Test that attributes list is properly handled in the flattening process."""

    # Get a small sample of real data
    feature_api = FeatureApi()

    polygon = Polygon(
        [
            (-111.8905, 40.7610),
            (-111.8904, 40.7610),
            (-111.8904, 40.7611),
            (-111.8905, 40.7611),
            (-111.8905, 40.7610),
        ]
    )

    features_gdf, _, _, _ = feature_api.get_features_gdf(
        polygon,
        until="2025-06-20",
        region="us",
        packs=["building", "building_char", "roof_char"],
    )

    if len(features_gdf) > 0 and "attributes" in features_gdf.columns:
        # Check that attributes is a list
        for idx, row in features_gdf.iterrows():
            attrs = row.get("attributes")
            if attrs is not None:
                assert isinstance(attrs, list), "Attributes should be a list"

                # Test the flattening logic (function is internal to exporter)
                # Replicate the flattening logic here for testing
                flattened = {}
                for i, attr_obj in enumerate(attrs):
                    if not isinstance(attr_obj, dict):
                        continue

                    desc = attr_obj.get("description", f"attr_{i}")

                    for key, value in attr_obj.items():
                        if key in ["description", "internalClassId"]:
                            continue

                        if key == "components" and isinstance(value, (list, dict)):
                            flattened[f"{desc}.components"] = json.dumps(value)
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if sub_value is not None:
                                    flattened[f"{desc}.{key}.{sub_key}"] = sub_value
                        elif value is not None:
                            flattened[f"{desc}.{key}"] = value

                # Check the flattened structure
                assert isinstance(flattened, dict), "Flattened result should be a dict"

                # Check that internalClassId is not in flattened keys
                for key in flattened.keys():
                    assert "internalClassId" not in key, f"Key {key} should not contain internalClassId"

                # Check that components are JSON strings
                for key, value in flattened.items():
                    if "components" in key:
                        assert isinstance(value, str), f"{key} should be a JSON string"
                        # Verify it's valid JSON
                        parsed = json.loads(value)
                        assert isinstance(parsed, list), f"{key} should contain a JSON array"

                break  # Just test one row with attributes
