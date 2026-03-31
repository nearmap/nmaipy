"""
Performance profiling test for parcel_rollup() and the full chunk closeout pipeline.

Run the fixture generator first (remove the skip decorator), then run the performance test.

    NMAIPY_PROFILE_CHUNK=0 pytest tests/test_parcel_rollup_performance.py::test_parcel_rollup_performance -s

cProfile output (if env var is set) is written to /tmp/nmaipy_profile_parcel_rollup.txt.
"""

import ast
import cProfile
import os
import pstats
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.wkt import loads as wkt_loads

from nmaipy import parcels
from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS, LAT_LONG_CRS, ROOF_INSTANCE_CLASS_ID
from nmaipy.feature_api import FeatureApi
from nmaipy.roof_age_api import RoofAgeApi

data_directory = Path(__file__).parent / "data"

PERF_PACKS = ["building", "building_structures", "roof_char"]
PERF_INCLUDE = ["roofSpotlightIndex"]
PERF_REGION = "us"
PERF_SINCE = "2022-06-29"
PERF_UNTIL = "2022-06-29"
PERF_FIXTURE = data_directory / "test_features_perf_nj_building_roof.csv"
PERF_THREADS = 15

PERF_AU_PACKS = None  # None = all packs
PERF_AU_REGION = "au"
PERF_AU_FIXTURE = data_directory / "test_features_perf_au_all_packs.csv"


@pytest.mark.skip("Comment out this line if you wish to regen the AU perf fixture")
def test_gen_perf_au_fixture(parcels_gdf: gpd.GeoDataFrame, cache_directory: Path):
    """
    Pull Feature API data for ~19 Sydney parcels and save to CSV.
    No roof age (AU not supported). Sydney grid tiles clip many roofs at parcel boundaries,
    exercising calculate_child_feature_attributes() far more than NJ residential parcels.
    """
    feature_api = FeatureApi(cache_dir=cache_directory, threads=PERF_THREADS)

    features_gdf, _, _ = feature_api.get_features_gdf_bulk(
        parcels_gdf,
        packs=PERF_AU_PACKS,
        region=PERF_AU_REGION,
    )

    features_gdf.reset_index().to_csv(PERF_AU_FIXTURE, index=False)
    print(f"\nWrote {len(features_gdf)} feature rows to {PERF_AU_FIXTURE}")


@pytest.mark.skip("Comment out this line if you wish to regen the perf fixture")
def test_gen_perf_fixture(parcels_2_gdf: gpd.GeoDataFrame, cache_directory: Path):
    """
    Pull Feature API + Roof Age API data for 100 NJ parcels and save to CSV.
    Uses 15 threads to match real chunk processing speed.
    Packs: building, building_structures, roof_char + roofSpotlightIndex include + roof age.
    """
    feature_api = FeatureApi(cache_dir=cache_directory, threads=PERF_THREADS)

    features_gdf, _, _ = feature_api.get_features_gdf_bulk(
        parcels_2_gdf,
        packs=PERF_PACKS,
        include=PERF_INCLUDE,
        region=PERF_REGION,
        since_bulk=PERF_SINCE,
        until_bulk=PERF_UNTIL,
    )

    roof_age_gdf, _, _ = RoofAgeApi(cache_dir=cache_directory).get_roof_age_bulk(parcels_2_gdf)

    combined = pd.concat([features_gdf.reset_index(), roof_age_gdf], ignore_index=True)
    combined.to_csv(PERF_FIXTURE, index=False)
    print(f"\nWrote {len(combined)} feature rows to {PERF_FIXTURE}")


def test_parcel_rollup_performance(parcels_2_gdf: gpd.GeoDataFrame):
    """
    Full pipeline performance test: API pull (15 threads) → parcel_rollup().

    Measures API phase and rollup phase separately so you can see the proportions
    that match a real export chunk (API-bound → CPU-bound dead zone).

    Set NMAIPY_PROFILE_CHUNK=1 to enable cProfile around parcel_rollup():
        NMAIPY_PROFILE_CHUNK=1 pytest tests/test_parcel_rollup_performance.py::test_parcel_rollup_performance -s

    cProfile output is written to /tmp/nmaipy_profile_parcel_rollup.txt.
    """
    if not PERF_FIXTURE.exists():
        pytest.skip(f"Perf fixture not generated yet — run test_gen_perf_fixture first: {PERF_FIXTURE}")

    # --- Load fixture (pre-saved API responses) ---
    import ast

    from shapely.wkt import loads as wkt_loads

    raw_df = pd.read_csv(PERF_FIXTURE)

    # Reconstruct features_gdf from the saved CSV (same pattern as conftest fixtures)
    has_attributes = "attributes" in raw_df.columns
    if has_attributes:
        raw_df["attributes"] = raw_df["attributes"].apply(
            lambda v: ast.literal_eval(v) if pd.notna(v) and isinstance(v, str) else v
        )

    features_gdf = gpd.GeoDataFrame(
        raw_df.drop("geometry", axis=1),
        geometry=raw_df["geometry"].apply(wkt_loads),
        crs=LAT_LONG_CRS,
    )
    if AOI_ID_COLUMN_NAME in features_gdf.columns:
        features_gdf = features_gdf.set_index(AOI_ID_COLUMN_NAME)

    # --- Build classes_df from the live API (same as exporter does) ---
    t_classes_start = time.monotonic()
    feature_api = FeatureApi(threads=PERF_THREADS)
    classes_df = feature_api.get_feature_classes(packs=PERF_PACKS)
    # Add Roof Instance class for roof age linkage (mirrors exporter logic)
    if ROOF_INSTANCE_CLASS_ID not in classes_df.index:
        roof_instance_row = pd.DataFrame(
            [{"description": "Roof Instance"}],
            index=pd.Index([ROOF_INSTANCE_CLASS_ID], name="id"),
        )
        classes_df = pd.concat([classes_df, roof_instance_row])
    t_classes_end = time.monotonic()

    print(f"\nclasses_df fetch: {t_classes_end - t_classes_start:.1f}s  ({len(classes_df)} classes)")
    print(f"features_gdf rows: {len(features_gdf)}, AOIs: {features_gdf.index.nunique()}")

    # --- Run parcel_rollup with optional cProfile ---
    do_profile = os.environ.get("NMAIPY_PROFILE_CHUNK", "0") != "0"

    t_rollup_start = time.monotonic()

    if do_profile:
        pr = cProfile.Profile()
        pr.enable()

    rollup_df = parcels.parcel_rollup(
        parcels_2_gdf,
        features_gdf,
        classes_df,
        country="us",
        primary_decision="largest_intersection",
    )

    if do_profile:
        pr.disable()
        profile_path = "/tmp/nmaipy_profile_parcel_rollup.txt"
        with open(profile_path, "w") as pf:
            pf.write("=== Top 40 by cumulative time ===\n")
            pstats.Stats(pr, stream=pf).sort_stats("cumulative").print_stats(40)
            pf.write("\n=== Top 40 by self (tottime) ===\n")
            pstats.Stats(pr, stream=pf).sort_stats("tottime").print_stats(40)
        print(f"cProfile output written to {profile_path}")

    t_rollup_end = time.monotonic()

    rollup_s = t_rollup_end - t_rollup_start
    print(
        f"parcel_rollup: {rollup_s:.1f}s for {len(parcels_2_gdf)} AOIs  ({rollup_s / len(parcels_2_gdf) * 1000:.1f}ms/AOI)"
    )
    print(f"rollup_df shape: {rollup_df.shape}")

    assert len(rollup_df) == len(
        parcels_2_gdf
    ), f"Expected {len(parcels_2_gdf)} rows in rollup_df, got {len(rollup_df)}"


def test_parcel_rollup_performance_au(parcels_gdf: gpd.GeoDataFrame):
    """
    Performance test for AU data — Sydney grid tiles clip many roofs at parcel boundaries,
    so calculate_child_feature_attributes() fires frequently.

    Also benchmarks the old per-element geometry loop vs the vectorised replacement so the
    speedup from that specific change is directly measurable on this data.

    Set NMAIPY_PROFILE_CHUNK=1 to enable cProfile:
        NMAIPY_PROFILE_CHUNK=1 pytest tests/test_parcel_rollup_performance.py::test_parcel_rollup_performance_au -s
    """
    if not PERF_AU_FIXTURE.exists():
        pytest.skip(f"AU perf fixture not generated yet — run test_gen_perf_au_fixture first: {PERF_AU_FIXTURE}")

    raw_df = pd.read_csv(PERF_AU_FIXTURE)
    if "attributes" in raw_df.columns:
        raw_df["attributes"] = raw_df["attributes"].apply(
            lambda v: ast.literal_eval(v) if pd.notna(v) and isinstance(v, str) else v
        )
    features_gdf = gpd.GeoDataFrame(
        raw_df.drop("geometry", axis=1),
        geometry=raw_df["geometry"].apply(wkt_loads),
        crs=LAT_LONG_CRS,
    )
    if AOI_ID_COLUMN_NAME in features_gdf.columns:
        features_gdf = features_gdf.set_index(AOI_ID_COLUMN_NAME)

    feature_api = FeatureApi(threads=PERF_THREADS)
    classes_df = feature_api.get_feature_classes(packs=PERF_AU_PACKS)

    print(f"\nfeatures_gdf rows: {len(features_gdf)}, AOIs: {features_gdf.index.nunique()}")

    # Report how many clipped roofs exist — these exercise calculate_child_feature_attributes
    if "is_clipped" in features_gdf.columns:
        n_clipped = features_gdf["is_clipped"].sum()
        print(f"Clipped features in dataset: {n_clipped} / {len(features_gdf)}")
    else:
        print("No is_clipped column found")

    # --- Full parcel_rollup benchmark ---
    do_profile = os.environ.get("NMAIPY_PROFILE_CHUNK", "0") != "0"
    t_start = time.monotonic()

    if do_profile:
        pr = cProfile.Profile()
        pr.enable()

    rollup_df = parcels.parcel_rollup(
        parcels_gdf,
        features_gdf,
        classes_df,
        country="au",
        primary_decision="largest_intersection",
    )

    if do_profile:
        pr.disable()
        profile_path = "/tmp/nmaipy_profile_parcel_rollup_au.txt"
        with open(profile_path, "w") as pf:
            pf.write("=== Top 40 by cumulative time ===\n")
            pstats.Stats(pr, stream=pf).sort_stats("cumulative").print_stats(40)
            pf.write("\n=== Top 40 by self (tottime) ===\n")
            pstats.Stats(pr, stream=pf).sort_stats("tottime").print_stats(40)
        print(f"cProfile output written to {profile_path}")

    rollup_s = time.monotonic() - t_start
    n_aois = len(parcels_gdf)
    print(f"parcel_rollup (AU): {rollup_s:.3f}s for {n_aois} AOIs  ({rollup_s / n_aois * 1000:.1f}ms/AOI)")

    assert len(rollup_df) == n_aois, f"Expected {n_aois} rows in rollup_df, got {len(rollup_df)}"
