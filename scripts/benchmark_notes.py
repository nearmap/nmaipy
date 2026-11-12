"""Merge memory benchmark results into a release body as a `## Memory benchmark` section.

The section replaces an existing `## Memory benchmark` section, up to the next `## ` heading, or is appended.
It uses only the standard library, so it runs under a plain `python -I`.

Usage: python scripts/benchmark_notes.py --body BODY.md --results RESULTS.csv --run-url URL --out OUT.md
"""

import argparse
import csv
from pathlib import Path

HEADING = "## Memory benchmark"
TABLE_COLUMNS = ("aois", "peak_main_rss_mib", "peak_worker_rss_mib", "peak_tree_pss_mib", "wall_seconds")
ENVIRONMENT_COLUMNS = ("commit", "python", "platform", "cpus")
DESCRIPTION = (
    "Peak memory (MiB) and wall time (seconds) of the offline benchmark export, `python -m tests.golden.benchmark`."
)


def main(argv: list[str] | None = None) -> None:
    """Write the release body with the benchmark section from `--results` merged in to `--out`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--body", type=Path, required=True, help="current release body (Markdown)")
    parser.add_argument("--results", type=Path, required=True, help="benchmark results CSV")
    parser.add_argument("--run-url", required=True, help="URL of the workflow run that produced the results")
    parser.add_argument("--out", type=Path, required=True, help="where to write the merged release body")
    args = parser.parse_args(argv)
    with args.results.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.results} has no result rows")
    body = args.body.read_text(encoding="utf-8")
    args.out.write_text(merge_section(body, render_section(rows, args.run_url)), encoding="utf-8")


def merge_section(body: str, section: str) -> str:
    """Return `body` (CRLF normalised to LF) with `section` replacing its `## Memory benchmark` section, or appended.

    The replaced span runs from the line that is exactly `## Memory benchmark` up to, not including, the next line
    that starts with `## `, or to the end. `section` must start with that heading line, and is always emitted ending
    in exactly one newline. An appended section follows the body after exactly one blank line; a replaced section
    that has a later heading after it is followed by exactly one blank line, then that heading. Text before and
    after the replaced span is kept unchanged, so merging the same section twice gives the same result as once.
    An empty or whitespace-only body gives just the section.
    """
    assert section.startswith(f"{HEADING}\n"), f"section must start with the line {HEADING!r}"
    section = section.rstrip("\n") + "\n"
    body = body.replace("\r\n", "\n")
    if not body.strip():
        return section
    lines = body.split("\n")
    if HEADING not in lines:
        return body.rstrip("\n") + "\n\n" + section
    start = lines.index(HEADING)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), None)
    before = "".join(f"{line}\n" for line in lines[:start])
    if end is None:
        return before + section
    return before + section + "\n" + "\n".join(lines[end:])


def render_section(rows: list[dict[str, str]], run_url: str) -> str:
    """Render benchmark result rows, in order, as a `## Memory benchmark` Markdown section ending in one newline."""
    assert rows, "no benchmark result rows"
    environment = {column: rows[0][column] for column in ENVIRONMENT_COLUMNS}
    for row in rows:
        row_environment = {column: row[column] for column in ENVIRONMENT_COLUMNS}
        assert row_environment == environment, f"rows differ in {ENVIRONMENT_COLUMNS}: {row_environment}, {environment}"
    lines = [
        HEADING,
        "",
        DESCRIPTION,
        "",
        _table_row(TABLE_COLUMNS),
        _table_row(["---:"] * len(TABLE_COLUMNS)),
        *(_table_row([row[column] for column in TABLE_COLUMNS]) for row in rows),
        "",
        f"Commit `{environment['commit']}`, Python `{environment['python']}`, platform `{environment['platform']}`, "
        + f"`{environment['cpus']}` CPUs.",
        "",
        f"[Benchmark run]({run_url})",
    ]
    return "\n".join(lines) + "\n"


def _table_row(cells: list[str] | tuple[str, ...]) -> str:
    return "| " + " | ".join(cells) + " |"


if __name__ == "__main__":
    main()
