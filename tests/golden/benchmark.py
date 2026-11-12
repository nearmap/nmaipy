"""Measure the peak memory of offline exports of the golden `us` parcels tiled to N AOIs, and append it to a CSV.

Each size runs `NearmapAIExporter` in a fresh interpreter against an in-process fake of the Feature and Roof Age APIs
(`tests.golden.tiled`), checks the export, then deletes it. The peaks are the main process's and the largest worker's
own peak resident memory (`VmHWM`), and the largest summed PSS of the export's whole process tree. Linux only.

Usage: python -m tests.golden.benchmark --work-dir DIR --results CSV [--aois N ...]
"""

import argparse
import csv
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import pandas as pd
import psutil

import nmaipy

from . import tiled
from .replay import CASES, peak_rss_kib, run_tiled

COLUMNS = (
    "date",
    "commit",
    "nmaipy_version",
    "python",
    "platform",
    "cpus",
    "aois",
    "peak_tree_pss_mib",
    "peak_main_rss_mib",
    "peak_worker_rss_mib",
    "wall_seconds",
)
SAMPLE_SECONDS = 0.2
SAMPLE_SLACK_KIB = 1024
REPO_ROOT = Path(__file__).parents[2]


def main(argv: list[str] | None = None) -> None:
    """Benchmark each `--aois` size in turn and append one row per size to `--results`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aois", type=int, nargs="+", default=[1000, 10000, 100000], metavar="N", help="sizes")
    parser.add_argument("--work-dir", type=Path, help="where each export, its log and its peak are written")
    parser.add_argument("--results", type=Path, help="CSV to append to (its header is written when it is absent)")
    parser.add_argument("--child", nargs=2, metavar=("N", "DIR"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        _child(int(args.child[0]), Path(args.child[1]))
        return
    if args.work_dir is None or args.results is None:
        parser.error("--work-dir and --results are required")
    if min(args.aois) < len(tiled.TEMPLATE_AOIS):
        parser.error(f"every size must be at least {len(tiled.TEMPLATE_AOIS)}, one whole block")
    for n in args.aois:
        row = measure(n, args.work_dir)
        append_row(args.results, row)
        print(row)


def measure(n: int, work_dir: Path) -> dict[str, object]:
    """Run, check and delete one `n`-AOI export under `work_dir`; return its CSV row."""
    export_dir, log_path, main_peak_path = _paths(n, work_dir)
    assert not export_dir.exists(), f"{export_dir} exists from an earlier run"
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "tests.golden.benchmark", "--child", str(n), str(work_dir)]
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        peak_tree_pss, sampled_peaks = _sample_tree(child, main_peak_path, export_dir / "markers")
    wall_seconds = time.monotonic() - start
    if child.returncode != 0:
        raise RuntimeError(f"the {n}-AOI export exited with {child.returncode}; its log is {log_path}")
    main_peak = json.loads(main_peak_path.read_text(encoding="utf-8"))
    assert main_peak["pid"] == child.pid, f"{main_peak_path} is from pid {main_peak['pid']}, not {child.pid}"
    worker_peaks = _worker_peaks(export_dir / "markers")
    recorded = {child.pid: main_peak["peak_rss_kib"], **worker_peaks}
    if over := {p: kib for p, kib in sampled_peaks.items() if p in recorded and kib > recorded[p] + SAMPLE_SLACK_KIB}:
        raise AssertionError(f"sampled VmHWM above the recorded peak (KiB): {over}; recorded {recorded}")
    validate(n, export_dir / "final")
    shutil.rmtree(export_dir)
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "nmaipy_version": nmaipy.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "aois": n,
        "peak_tree_pss_mib": round(peak_tree_pss / 2**20, 1),
        "peak_main_rss_mib": round(main_peak["peak_rss_kib"] / 1024, 1),
        "peak_worker_rss_mib": round(max(worker_peaks.values()) / 1024, 1),
        "wall_seconds": round(wall_seconds, 1),
    }


def validate(n: int, final: Path) -> None:
    """Fail unless the export in `final/` is `n` whole, distinct copies of the template's own answers."""
    aoi_ids = pd.read_csv(final / "rollup.csv", usecols=["aoi_id"], dtype=str)["aoi_id"]
    assert len(aoi_ids) == n, f"rollup.csv has {len(aoi_ids)} rows, not {n}"
    feature_errors = final / "feature_api_errors.csv"
    assert not feature_errors.exists() or pd.read_csv(feature_errors).empty, f"{feature_errors} has rows"
    misses = tiled.TiledFakeApi(CASES["us"]["until"]).roof_age_misses()
    roof_age_errors = final / "roof_age_errors.csv"
    got = set(pd.read_csv(roof_age_errors, dtype=str)["aoi_id"]) if roof_age_errors.exists() else set()
    want = {aoi_id for aoi_id in aoi_ids if tiled.template_aoi(aoi_id) in misses}
    assert got == want, f"roof_age_errors.csv: {len(got - want)} unexpected AOIs, {len(want - got)} missing"

    features = pd.read_parquet(final / "features.parquet", columns=["aoi_id", "feature_id"])
    rows = features.groupby("aoi_id").size()
    block_zero = {aoi_id: int(rows.get(f"b0_{aoi_id}", 0)) for aoi_id in tiled.TEMPLATE_AOIS}
    assert sum(block_zero.values()) > 0, "block 0 has no features"
    wrong = [a for a in aoi_ids if int(rows.get(a, 0)) != block_zero[tiled.template_aoi(a)]]
    assert not wrong, f"{len(wrong)} AOIs have a different feature count from block 0 (e.g. {wrong[0]})"
    blocks = features["aoi_id"].map(tiled.block)
    full_blocks = n // len(tiled.TEMPLATE_AOIS)
    distinct_zero = features.loc[blocks == 0, "feature_id"].nunique()
    distinct = features.loc[blocks < full_blocks, "feature_id"].nunique()
    assert distinct == full_blocks * distinct_zero, (
        f"{distinct} distinct feature ids in {full_blocks} whole blocks, not {full_blocks} x {distinct_zero}"
    )


def append_row(results: Path, row: dict[str, object]) -> None:
    """Append `row` to the `results` CSV, writing the header first when the file is absent."""
    new = not results.exists()
    with results.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def _child(n: int, work_dir: Path) -> None:
    export_dir, _, main_peak_path = _paths(n, work_dir)
    run_tiled(n, export_dir)
    main_peak_path.write_text(json.dumps({"pid": os.getpid(), "peak_rss_kib": peak_rss_kib()}), encoding="utf-8")


def _paths(n: int, work_dir: Path) -> tuple[Path, Path, Path]:
    return work_dir / f"export_{n}", work_dir / f"export_{n}.log", work_dir / f"export_{n}.main.json"


def _sample_tree(child: subprocess.Popen, main_peak_path: Path, marker_dir: Path) -> tuple[int, dict[int, int]]:
    """Sample the child's process tree until it exits: the largest summed PSS, and each pid's largest VmHWM.

    A pid's VmHWM is sampled only until it records its own peak, since it can still grow while the process exits.
    """
    root = psutil.Process(child.pid)
    peak_pss = 0
    peaks: dict[int, int] = {}
    while child.poll() is None:
        total = 0
        for process in _tree(root):
            recorded = main_peak_path if process.pid == child.pid else marker_dir / f"{process.pid}.hwm"
            try:
                pss = process.memory_full_info().pss
                peak = None if recorded.exists() else peak_rss_kib(process.pid)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            total += pss
            if peak is not None:
                peaks[process.pid] = max(peaks.get(process.pid, 0), peak)
        peak_pss = max(peak_pss, total)
        time.sleep(SAMPLE_SECONDS)
    return peak_pss, peaks


def _tree(root: psutil.Process) -> list[psutil.Process]:
    try:
        return [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return []


def _worker_peaks(marker_dir: Path) -> dict[int, int]:
    installed = {int(p.stem) for p in marker_dir.glob("*.installed")}
    peaks = {int(p.stem): int(p.read_text(encoding="utf-8")) for p in marker_dir.glob("*.hwm")}
    assert installed, f"no worker installed the fake in {marker_dir}"
    assert set(peaks) == installed, f"workers without a recorded peak: {sorted(installed - set(peaks))}"
    return peaks


if __name__ == "__main__":
    main()
