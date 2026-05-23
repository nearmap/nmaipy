"""Micro-benchmark for _compute_all_per_class_data hot spots.

Drives the same function the chunk-processing loop calls, against a real
feature-chunk parquet pulled from a representative many-pack property
export. Times the function across a sweep of thread counts so the
sequential vs parallel roof_attrs build (and per-class loop) are
directly comparable. Also reports the groupby vs per-class boolean-filter
microbench independently so the Candidate 2 win is visible without
needing a before/after branch checkout.

Usage:
    python scripts/benchmark_per_class_compute.py \\
        --features path/to/features_NNNN.parquet \\
        --rollup   path/to/rollup_NNNN.parquet \\
        --country  au

Pulls feature + rollup parquets from any prior export's `chunks/` directory.
"""

import argparse
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

from nmaipy.constants import (
    AOI_ID_COLUMN_NAME,
    BUILDING_NEW_ID,
    PRIMARY_FEATURE_COLUMN_TO_CLASS,
    ROOF_ID,
    ROOF_INSTANCE_CLASS_ID,
)
from nmaipy.exporter import (
    _add_is_primary_column,
    _batch_project_geometries,
    _compute_all_per_class_data,
    _dataframe_to_records_with_index,
    _group_children_by_aoi,
)
from nmaipy.feature_attributes import flatten_roof_attributes


def _load_chunk(features_path: Path, rollup_path: Path) -> tuple:
    features = gpd.read_parquet(features_path)
    rollup = pd.read_parquet(rollup_path)

    if AOI_ID_COLUMN_NAME in rollup.columns:
        rollup = rollup.set_index(AOI_ID_COLUMN_NAME)
    primary_cols = [c for c in PRIMARY_FEATURE_COLUMN_TO_CLASS if c in rollup.columns]
    primary_ids_df = rollup[primary_cols].copy() if primary_cols else pd.DataFrame()

    chunk_gdf = _add_is_primary_column(features, primary_ids_df)
    return chunk_gdf, primary_ids_df


def _summarise_chunk(chunk_gdf: gpd.GeoDataFrame) -> None:
    print(f"chunk rows         : {len(chunk_gdf):,}")
    if "class_id" in chunk_gdf.columns:
        n_classes = chunk_gdf["class_id"].nunique()
        print(f"distinct classes   : {n_classes}")
        if ROOF_ID in chunk_gdf["class_id"].values:
            n_roofs = (chunk_gdf["class_id"] == ROOF_ID).sum()
            print(f"roof rows          : {n_roofs:,}")
        if BUILDING_NEW_ID in chunk_gdf["class_id"].values:
            n_buildings = (chunk_gdf["class_id"] == BUILDING_NEW_ID).sum()
            print(f"building rows      : {n_buildings:,}")
    print()


def _time_compute(chunk_gdf: gpd.GeoDataFrame, country: str, threads: int, runs: int) -> float:
    durations = []
    for _ in range(runs):
        t0 = time.monotonic()
        _compute_all_per_class_data(
            chunk_gdf=chunk_gdf,
            country=country,
            aoi_input_columns=[],
            threads=threads,
        )
        durations.append(time.monotonic() - t0)
    return min(durations)


def _microbench_roof_attrs(chunk_gdf: gpd.GeoDataFrame, country: str, runs: int = 2) -> None:
    """Isolate the roof_attrs_cache build phase and compare seq vs threaded."""
    if "class_id" not in chunk_gdf.columns:
        return
    roof_features_chunk = chunk_gdf[chunk_gdf["class_id"] == ROOF_ID]
    non_roof_chunk = chunk_gdf[chunk_gdf["class_id"] != ROOF_ID]
    if len(roof_features_chunk) == 0:
        print("(no roof rows, skipping roof_attrs microbench)")
        return

    child_by_aoi = _group_children_by_aoi(non_roof_chunk, None)
    roof_geoms_proj, child_proj_by_aoi = _batch_project_geometries(
        roof_features_chunk, child_by_aoi, country
    )
    roof_records = _dataframe_to_records_with_index(roof_features_chunk)

    def _one_roof(ridx, row):
        roof_aoi = row.get(AOI_ID_COLUMN_NAME)
        return flatten_roof_attributes(
            [row],
            country=country,
            child_features=child_by_aoi.get(roof_aoi) if roof_aoi is not None else None,
            parent_projected=roof_geoms_proj.iloc[ridx],
            children_projected=child_proj_by_aoi.get(roof_aoi),
        )

    def _seq() -> None:
        for ridx, row in enumerate(roof_records):
            _one_roof(ridx, row)

    def _parallel(threads: int) -> None:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(
                as_completed(
                    ex.submit(_one_roof, ridx, row) for ridx, row in enumerate(roof_records)
                )
            )

    print(f"roof_attrs build microbench ({len(roof_records):,} roofs, min of {runs} runs):")
    t_seq = min(_time(lambda: _seq()) for _ in range(runs))
    print(f"  threads=1   {t_seq:>8.3f}s")
    for n in (4, 8, 15):
        t = min(_time(lambda: _parallel(n)) for _ in range(runs))
        speedup = t_seq / t if t > 0 else float("inf")
        print(f"  threads={n:<3} {t:>8.3f}s  ({speedup:.2f}x)")
    print()


def _time(fn) -> float:
    t0 = time.monotonic()
    fn()
    return time.monotonic() - t0


def _microbench_class_lookup(chunk_gdf: gpd.GeoDataFrame, runs: int = 5) -> None:
    if "class_id" not in chunk_gdf.columns:
        print("(no class_id column, skipping lookup microbench)")
        return

    class_ids = list(chunk_gdf["class_id"].dropna().unique())

    t_bool = []
    for _ in range(runs):
        t0 = time.monotonic()
        for cid in class_ids:
            _ = chunk_gdf[chunk_gdf["class_id"] == cid]
        t_bool.append(time.monotonic() - t0)

    t_grp = []
    for _ in range(runs):
        t0 = time.monotonic()
        groups = dict(iter(chunk_gdf.groupby("class_id", sort=False)))
        for cid in class_ids:
            _ = groups.get(cid)
        t_grp.append(time.monotonic() - t0)

    print(f"class-lookup microbench (over {len(class_ids)} classes, min of {runs} runs):")
    print(f"  boolean filters : {min(t_bool):.3f}s")
    print(f"  groupby + dict  : {min(t_grp):.3f}s  ({min(t_bool) / max(min(t_grp), 1e-9):.1f}x faster)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--rollup", required=True, type=Path)
    ap.add_argument("--country", default="au")
    ap.add_argument("--threads", default="1,4,8,15", help="comma-separated thread counts to sweep")
    ap.add_argument("--runs", type=int, default=2, help="runs per thread count (report min)")
    args = ap.parse_args()

    print(f"Loading chunk: {args.features.name}")
    chunk_gdf, _primary = _load_chunk(args.features, args.rollup)
    _summarise_chunk(chunk_gdf)

    _microbench_class_lookup(chunk_gdf)
    _microbench_roof_attrs(chunk_gdf, args.country)

    thread_counts = [int(t) for t in args.threads.split(",")]
    print(f"_compute_all_per_class_data timings (min of {args.runs} runs):")
    print(f"  {'threads':>8}  {'wall (s)':>10}  {'speedup':>8}")
    baseline = None
    for n in thread_counts:
        wall = _time_compute(chunk_gdf, args.country, n, args.runs)
        if baseline is None:
            baseline = wall
            speedup = 1.0
        else:
            speedup = baseline / wall if wall > 0 else float("inf")
        print(f"  {n:>8}  {wall:>10.2f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
