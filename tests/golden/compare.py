"""Compare two exports by value, and write the canonical text form of an export used as golden `expected/`.

Usage: python -m tests.golden.compare LEFT RIGHT

Each side is an export folder (its `final/` when present) or a canonical folder written by `canonicalise_export`.
"""

import argparse
import csv
import io
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

RTOL = 1e-6
ATOL = 1e-9
GEOMETRY_TOLERANCE = 1e-9
SIGNIFICANT_DIGITS = 9
IGNORED_FILES = frozenset({"latency_stats.csv", "export_config.json"})
README_STAMP = "**Generated:**"
SCHEMAS = "schemas.json"
_DECIMAL = re.compile(r"^-?\d+(\.\d*)?([eE][-+]?\d+)?$")
_INTEGER = re.compile(r"^-?\d+$")


def main() -> None:
    """Print the differences between two exports; exit 1 when there are any."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    differences = compare_exports(args.left, args.right)
    print("\n".join(differences) if differences else "No differences")
    sys.exit(1 if differences else 0)


def compare_exports(expected: Path, actual: Path) -> list[str]:
    """One human-readable line per difference between two exports (export or canonical folders)."""
    with tempfile.TemporaryDirectory() as tmp:
        expected = _canonical(expected, Path(tmp) / "expected")
        actual = _canonical(actual, Path(tmp) / "actual")
        expected_files, actual_files = _files(expected), _files(actual)
        differences = [f"{name}: only in the expected export" for name in sorted(expected_files - actual_files)]
        differences += [f"{name}: only in the actual export" for name in sorted(actual_files - expected_files)]
        for name in sorted(expected_files & actual_files):
            differences += [f"{name}: {d}" for d in _compare_file(expected / name, actual / name)]
    return differences


def canonicalise_export(final_dir: Path, expected_dir: Path) -> None:
    """Replace `expected_dir` with a run-independent text form of the export in `final_dir`.

    Tables keyed by `aoi_id` are transposed: line 1 is `column` then the row keys (`aoi_id`, plus `feature_id` when
    present), and each later line is one column then its values, sorted by column name. A `.parquet` whose `.csv`
    twin exists drops each non-key column whose values match the twin's for every row key, and keeps the rest. Parquet
    dtypes and GeoParquet metadata go to `schemas.json`.
    """
    final_dir = _final(final_dir)
    shutil.rmtree(expected_dir, ignore_errors=True)
    expected_dir.mkdir(parents=True)
    names = _files(final_dir)
    schemas = {}
    for name in sorted(names):
        source = final_dir / name
        if source.suffix == ".csv":
            _write_table(_read_table(source), expected_dir / name)
        elif source.suffix == ".parquet":
            table = _read_table(source)
            schemas[name] = _schema(table)
            twin = source.with_name(source.stem.removesuffix("_features") + ".csv")
            if twin != source and twin.name in names:
                matching = _columns_matching(table, _read_table(final_dir / twin.name))
                table = table[[c for c in table.columns if c not in matching]]
            _write_table(table, expected_dir / f"{name}.csv")
        elif source.name == "README.md":
            (expected_dir / name).write_text(_readme_text(source), encoding="utf-8")
        elif source.suffix == ".json":
            parsed = json.loads(source.read_text(encoding="utf-8"))
            (expected_dir / name).write_text(json.dumps(parsed, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        else:
            shutil.copyfile(source, expected_dir / name)
    (expected_dir / SCHEMAS).write_text(json.dumps(schemas, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _canonical(folder: Path, scratch: Path) -> Path:
    if (folder / SCHEMAS).exists() and not (folder / "final").is_dir():
        return folder
    canonicalise_export(folder, scratch)
    return scratch


def _final(folder: Path) -> Path:
    return folder / "final" if (folder / "final").is_dir() else folder


def _files(folder: Path) -> set[str]:
    return {p.name for p in folder.iterdir() if p.is_file() and p.name not in IGNORED_FILES}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    table = gpd.read_parquet(path) if b"geo" in (pq.read_schema(path).metadata or {}) else pd.read_parquet(path)
    return table.reset_index() if any(table.index.names) else table


def _schema(table: pd.DataFrame) -> dict:
    schema = {"dtypes": {c: _dtype_name(table[c]) for c in table.columns}}
    if isinstance(table, gpd.GeoDataFrame):
        schema["geo"] = {"primary_column": table.geometry.name, "crs": table.crs.to_string() if table.crs else None}
    return schema


def _dtype_name(column: pd.Series) -> str:
    """The dtype; string-valued object columns are named like string columns (pandas versions name these apart)."""
    if column.dtype != object and not isinstance(column.dtype, pd.StringDtype):
        return str(column.dtype)
    inferred = pd.api.types.infer_dtype(column.astype(object), skipna=True)
    if isinstance(column.dtype, pd.StringDtype) or inferred in ("string", "empty"):
        return inferred
    return f"object:{inferred}"


def _key_columns(table: pd.DataFrame) -> list[str]:
    if "aoi_id" not in table.columns:
        return []
    return ["aoi_id", "feature_id"] if "feature_id" in table.columns else ["aoi_id"]


def _columns_matching(table: pd.DataFrame, twin: pd.DataFrame) -> set[str]:
    """The non-key columns `table` shares with `twin` whose canonical values are the same for every row key."""
    keys = _key_columns(table)
    shared = [c for c in table.columns if c in twin.columns and c not in keys]
    if not keys or _key_columns(twin) != keys:
        return set()
    left, right = _keyed_cells(table, keys, shared), _keyed_cells(twin, keys, shared)
    if left[keys[0]].keys() != right[keys[0]].keys():
        return set()
    return {c for c in shared if all(_cells_match(right[c][r], left[c][r], c) for r in left[c])}


def _keyed_cells(table: pd.DataFrame, keys: list[str], columns: list[str]) -> dict[str, dict[str, str]]:
    cells = {c: [_cell(v, c) for v in table[c]] for c in [*keys, *columns]}
    row_keys = _row_keys([[cells[k][i] for k in keys] for i in range(len(table))], cells)
    return {c: dict(zip(row_keys, values, strict=True)) for c, values in cells.items()}


def _write_table(table: pd.DataFrame, target: Path) -> None:
    cells = {c: [_cell(v, c) for v in table[c]] for c in table.columns}
    keys = _key_columns(table)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    if keys:
        row_keys = _row_keys([[cells[k][i] for k in keys] for i in range(len(table))], cells)
        order = sorted(range(len(table)), key=lambda i: row_keys[i])
        writer.writerow(["column", *(row_keys[i] for i in order)])
        for column in sorted(cells):
            writer.writerow([column, *(cells[column][i] for i in order)])
    else:
        writer.writerow(list(table.columns))
        rows = sorted(zip(*cells.values(), strict=True)) if cells else []
        writer.writerows(rows)
    target.write_text(out.getvalue(), encoding="utf-8")


def _row_keys(keys: list[list[str]], cells: dict[str, list[str]]) -> list[str]:
    """`aoi_id[/feature_id]`, with `#n` appended to repeats (ordered by the rows' other values)."""
    base = ["/".join(k) for k in keys]
    content = ["\x1f".join(column[i] for column in cells.values()) for i in range(len(base))]
    seen: dict[str, int] = {}
    row_keys = [""] * len(base)
    for i in sorted(range(len(base)), key=lambda i: (base[i], content[i])):
        seen[base[i]] = seen.get(base[i], 0) + 1
        row_keys[i] = base[i] if seen[base[i]] == 1 else f"{base[i]}#{seen[base[i]]}"
    return row_keys


def _cell(value: object, column: str) -> str:
    if isinstance(value, shapely.Geometry):
        return shapely.to_wkt(value, rounding_precision=SIGNIFICANT_DIGITS)
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return _float_text(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), default=str)
    text = str(value)
    if str(column).endswith("geometry") and text:
        return shapely.to_wkt(shapely.from_wkt(text), rounding_precision=SIGNIFICANT_DIGITS)
    if _DECIMAL.match(text) and not _INTEGER.match(text):
        return _float_text(float(text))
    return text


def _float_text(value: float) -> str:
    """Rounded to `SIGNIFICANT_DIGITS`, keeping a decimal marker so a float never reads as an integer."""
    text = f"{value:.{SIGNIFICANT_DIGITS}g}"
    return text if any(c in text for c in ".en") else f"{text}.0"


def _readme_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(line for line in lines if not line.startswith(README_STAMP))


def _compare_file(expected: Path, actual: Path) -> list[str]:
    if expected.suffix == ".csv":
        return _compare_tables(expected, actual)
    if expected.suffix == ".json":
        left, right = (json.loads(p.read_text(encoding="utf-8")) for p in (expected, actual))
        return _json_differences(left, right)
    return [] if expected.read_bytes() == actual.read_bytes() else ["text differs"]


def _json_differences(expected: object, actual: object, where: str = "") -> list[str]:
    """One line per differing leaf path (`a/b/0/c`), or per key or list item only one side has."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        left, right = expected, actual
    elif isinstance(expected, list) and isinstance(actual, list):
        left, right = dict(enumerate(expected)), dict(enumerate(actual))
    else:
        return [] if expected == actual else [f"{where or '/'}: {expected!r} expected, {actual!r} actual"]
    child = {k: f"{where}/{k}" if where else str(k) for k in left.keys() | right.keys()}
    differences = [f"{child[k]}: missing" for k in left if k not in right]
    differences += [f"{child[k]}: extra" for k in right if k not in left]
    for k in left.keys() & right.keys():
        differences += _json_differences(left[k], right[k], child[k])
    return sorted(differences)


def _compare_tables(expected: Path, actual: Path) -> list[str]:
    left, right = _parse_table(expected), _parse_table(actual)
    differences = []
    if missing := sorted(set(left) - set(right)):
        differences.append(f"missing columns {missing}")
    if extra := sorted(set(right) - set(left)):
        differences.append(f"extra columns {extra}")
    left_rows = next(iter(left.values()), {}).keys()
    right_rows = next(iter(right.values()), {}).keys()
    if missing := sorted(set(left_rows) - set(right_rows)):
        differences.append(f"missing rows {missing}")
    if extra := sorted(set(right_rows) - set(left_rows)):
        differences.append(f"extra rows {extra}")
    rows = sorted(set(left_rows) & set(right_rows))
    for column in sorted(set(left) & set(right)):
        mismatched = [r for r in rows if not _cells_match(left[column][r], right[column][r], column)]
        if mismatched:
            first = mismatched[0]
            differences.append(
                f"column {column}: {len(mismatched)} mismatches, first at {first} "
                f"({left[column][first]!r} expected, {right[column][first]!r} actual)"
            )
    return differences


def _parse_table(path: Path) -> dict[str, dict[str, str]]:
    """A canonical table as {column: {row key: value}} (non-transposed tables are keyed by their first column)."""
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    if not rows:
        return {}
    header, body = rows[0], rows[1:]
    if header and header[0] == "column":
        return {line[0]: dict(zip(header[1:], line[1:], strict=True)) for line in body}
    keys = _row_keys([[line[0]] for line in body], {str(i): [line[i] for line in body] for i in range(len(header))})
    return {c: {keys[r]: body[r][i] for r in range(len(body))} for i, c in enumerate(header)}


def _cells_match(expected: str, actual: str, column: str) -> bool:
    if expected == actual:
        return True
    if not expected or not actual or _INTEGER.match(expected) or _INTEGER.match(actual):
        return False
    if column.endswith("geometry"):
        try:
            return bool(shapely.equals_exact(shapely.from_wkt(expected), shapely.from_wkt(actual), GEOMETRY_TOLERANCE))
        except shapely.errors.GEOSException:
            return False
    if _DECIMAL.match(expected) and _DECIMAL.match(actual):
        return bool(np.isclose(float(expected), float(actual), rtol=RTOL, atol=ATOL))
    return False


if __name__ == "__main__":
    main()
