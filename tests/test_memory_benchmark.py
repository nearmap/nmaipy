import csv
import importlib.util
from pathlib import Path
import sys

import pytest

from .golden import benchmark

_SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_notes.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_notes", _SCRIPT)
assert _SPEC is not None, f"cannot load {_SCRIPT}"
assert _SPEC.loader is not None, f"cannot load {_SCRIPT}"
benchmark_notes = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_notes)

SECTION = "## Memory benchmark\n\nnew\n"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param("", SECTION, id="empty"),
        pytest.param(" \n\n", SECTION, id="whitespace-only"),
        pytest.param("# Notes\n\nText.\n\n\n", "# Notes\n\nText.\n\n## Memory benchmark\n\nnew\n", id="append"),
        pytest.param(
            "## Changes\n\n- a\n\n## Memory benchmark\n\nold\n### Details\n\nold\n## Install\n\npip install\n",
            "## Changes\n\n- a\n\n## Memory benchmark\n\nnew\n\n## Install\n\npip install\n",
            id="replace-before-heading",
        ),
        pytest.param(
            "## Changes\n\n- a\n\n## Memory benchmark\n\nold\n\n",
            "## Changes\n\n- a\n\n## Memory benchmark\n\nnew\n",
            id="replace-at-end",
        ),
        pytest.param(
            "## Changes\r\n\r\n- a\r\n\r\n## Memory benchmark\r\n\r\nold\r\n",
            "## Changes\n\n- a\n\n## Memory benchmark\n\nnew\n",
            id="crlf",
        ),
    ],
)
def test_merge_section(body: str, expected: str) -> None:
    once = benchmark_notes.merge_section(body, SECTION)
    assert once == expected, f"merge gave {once!r}, expected {expected!r}"
    twice = benchmark_notes.merge_section(once, SECTION)
    assert twice == once, f"merging twice gave {twice!r}, once gave {once!r}"


def test_main_writes_merged_body(tmp_path: Path) -> None:
    header = "date,commit,nmaipy_version,python,platform,cpus,aois,peak_tree_pss_mib,peak_main_rss_mib,"
    header += "peak_worker_rss_mib,wall_seconds"
    results = tmp_path / "results.csv"
    results.write_text(
        f"{header}\n"
        "2026-01-02,abc1234,5.1.0,3.14.0,Linux-x86_64,4,10,300.5,200.1,90.2,12.3\n"
        "2026-01-02,abc1234,5.1.0,3.14.0,Linux-x86_64,4,100,410.0,250.0,120.0,40.0\n",
        encoding="utf-8",
    )
    body = tmp_path / "body.md"
    body.write_text("## Changes\n\n- a\n\n## Memory benchmark\n\nold\n", encoding="utf-8")
    out = tmp_path / "out.md"
    url = "https://github.com/nearmap/nmaipy/actions/runs/1"
    benchmark_notes.main(["--body", str(body), "--results", str(results), "--run-url", url, "--out", str(out)])
    expected = (
        "## Changes\n\n- a\n\n"
        "## Memory benchmark\n\n"
        "Peak memory (MiB) and wall time (seconds) of the offline benchmark export, "
        "`python -m tests.golden.benchmark`.\n\n"
        "| aois | peak_main_rss_mib | peak_worker_rss_mib | peak_tree_pss_mib | wall_seconds |\n"
        "| ---: | ---: | ---: | ---: | ---: |\n"
        "| 10 | 200.1 | 90.2 | 300.5 | 12.3 |\n"
        "| 100 | 250.0 | 120.0 | 410.0 | 40.0 |\n\n"
        "Commit `abc1234`, Python `3.14.0`, platform `Linux-x86_64`, `4` CPUs.\n\n"
        "[Benchmark run](https://github.com/nearmap/nmaipy/actions/runs/1)\n"
    )
    written = out.read_text(encoding="utf-8")
    assert written == expected, f"wrote {written!r}, expected {expected!r}"


@pytest.mark.skipif(sys.platform != "linux", reason="PSS and VmHWM exist only on Linux")
def test_benchmark_two_blocks(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    benchmark.main(["--aois", "10", "--work-dir", str(tmp_path / "work"), "--results", str(results)])
    with results.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["aois"] for row in rows] == ["10"], f"expected one row for 10 AOIs, got {rows}"
    peaks = {column: float(rows[0][column]) for column in rows[0] if column.startswith("peak_")}
    assert len(peaks) == 3, f"expected three peak columns, got {peaks}"
    assert all(peak > 0 for peak in peaks.values()), f"a peak is not positive: {peaks}"
