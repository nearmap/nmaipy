"""Compare two export folders by value, and write the canonical form of an export used as golden `expected/`.

Usage: python -m tests.golden.compare LEFT RIGHT
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

RTOL = 1e-6
ATOL = 1e-9
GEOMETRY_TOLERANCE = 1e-9
IGNORED_FILES = frozenset({"latency_stats.csv", "export_config.json"})
README_STAMP = "**Generated:**"


def main() -> None:
    """Print the differences between two export folders; exit 1 when there are any."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    differences = compare_exports(args.left, args.right)
    print("\n".join(differences) if differences else "No differences")
    sys.exit(1 if differences else 0)


def compare_exports(expected: Path, actual: Path) -> list[str]:
    """One human-readable line per difference between two export folders (their `final/` when present)."""
    expected, actual = _final(expected), _final(actual)
    expected_files, actual_files = _files(expected), _files(actual)
    differences = [f"{name}: only in {expected}" for name in sorted(expected_files - actual_files)]
    differences += [f"{name}: only in {actual}" for name in sorted(actual_files - expected_files)]
    for name in sorted(expected_files & actual_files):
        differences += [f"{name}: {d}" for d in _compare_file(expected / name, actual / name)]
    return differences


def canonicalise_export(final_dir: Path, expected_dir: Path) -> None:
    """Replace `expected_dir` with the files `compare_exports` reads from `final_dir`, in a run-independent form."""
    final_dir = _final(final_dir)
    shutil.rmtree(expected_dir, ignore_errors=True)
    expected_dir.mkdir(parents=True)
    for name in sorted(_files(final_dir)):
        source, target = final_dir / name, expected_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".csv":
            table = pd.read_csv(source, dtype=str, keep_default_na=False)
            _canonical_order(table).to_csv(target, index=False)
        elif source.suffix == ".parquet":
            _canonical_order(_read_table(source)).to_parquet(target, index=False)
        elif source.name == "README.md":
            target.write_text(_readme_text(source), encoding="utf-8")
        elif source.suffix == ".json":
            parsed = json.loads(source.read_text(encoding="utf-8"))
            target.write_text(json.dumps(parsed, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        else:
            shutil.copyfile(source, target)


def _final(folder: Path) -> Path:
    return folder / "final" if (folder / "final").is_dir() else folder


def _files(folder: Path) -> set[str]:
    return {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p.name not in IGNORED_FILES}


def _compare_file(expected: Path, actual: Path) -> list[str]:
    if expected.suffix in (".csv", ".parquet"):
        return _compare_tables(_read_table(expected), _read_table(actual))
    if expected.name == "README.md":
        return [] if _readme_text(expected) == _readme_text(actual) else ["text differs"]
    if expected.suffix == ".json":
        same = json.loads(expected.read_text(encoding="utf-8")) == json.loads(actual.read_text(encoding="utf-8"))
        return [] if same else ["JSON differs"]
    return [] if expected.read_bytes() == actual.read_bytes() else ["bytes differ"]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    table = gpd.read_parquet(path) if b"geo" in (pq.read_schema(path).metadata or {}) else pd.read_parquet(path)
    return table.reset_index() if any(table.index.names) else table


def _readme_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(line for line in lines if not line.startswith(README_STAMP))


def _key_columns(table: pd.DataFrame) -> list[str]:
    if "aoi_id" not in table.columns:
        return []
    return ["aoi_id", "feature_id"] if "feature_id" in table.columns else ["aoi_id"]


def _canonical_order(table: pd.DataFrame) -> pd.DataFrame:
    keys = _key_columns(table)
    others = sorted(c for c in table.columns if c not in keys)
    as_text = pd.DataFrame({c: table[c].map(str).to_numpy() for c in table.columns})
    order = as_text.sort_values(keys + others, kind="stable").index
    return table.iloc[order][keys + others].reset_index(drop=True)


def _compare_tables(expected: pd.DataFrame, actual: pd.DataFrame) -> list[str]:
    differences = []
    if missing := sorted(set(expected.columns) - set(actual.columns)):
        differences.append(f"missing columns {missing}")
    if extra := sorted(set(actual.columns) - set(expected.columns)):
        differences.append(f"extra columns {extra}")
    if len(expected) != len(actual):
        return differences + [f"row count {len(expected)} expected, {len(actual)} actual"]

    shared = set(expected.columns) & set(actual.columns)
    expected = _canonical_order(expected[[c for c in expected.columns if c in shared]])
    actual = _canonical_order(actual[[c for c in actual.columns if c in shared]])
    keys = _key_columns(expected)
    for column in expected.columns:
        matches = _cells_match(expected[column], actual[column])
        if not matches.all():
            first = int(np.flatnonzero(~matches)[0])
            where = expected.loc[first, keys].to_dict() if keys else f"row {first}"
            differences.append(f"column {column}: {int((~matches).sum())} mismatches, first at {where}")
    return differences


def _cells_match(expected: pd.Series, actual: pd.Series) -> np.ndarray:
    both_null = (expected.isna() & actual.isna()).to_numpy()
    if _is_geometry(expected) or _is_geometry(actual):
        return both_null | shapely.equals_exact(_geometries(expected), _geometries(actual), GEOMETRY_TOLERANCE)
    expected_num, actual_num = _numbers(expected), _numbers(actual)
    numeric = ~np.isnan(expected_num) & ~np.isnan(actual_num)
    close = numeric & np.isclose(expected_num, actual_num, rtol=RTOL, atol=ATOL)
    same_text = (expected.map(str) == actual.map(str)).to_numpy()
    one_null = (expected.isna() ^ actual.isna()).to_numpy()
    return both_null | close | (same_text & ~one_null)


def _is_geometry(column: pd.Series) -> bool:
    return isinstance(column, gpd.GeoSeries) or str(column.name).endswith("geometry")


def _geometries(column: pd.Series) -> np.ndarray:
    if isinstance(column, gpd.GeoSeries):
        return column.to_numpy()
    return shapely.from_wkt(column.where(column.notna(), None).to_numpy())


def _numbers(column: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(column):
        return np.full(len(column), np.nan)
    if pd.api.types.is_numeric_dtype(column):
        return column.to_numpy(dtype=float, na_value=np.nan)
    return np.array([_number(v) for v in column], dtype=float)


def _number(value: object) -> float:
    is_number = isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
    return float(value) if is_number else np.nan


if __name__ == "__main__":
    main()
