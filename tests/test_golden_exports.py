"""Golden exports: the full exporter run offline against a synthetic fake of the AI APIs, compared to stored outputs.

See tests/data/golden/README.md for the cases, how to re-baseline (`python -m tests.golden.regenerate`) and how to
refresh the fake when the live API changes (`python -m tests.golden.drift`).
"""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, mapping

from nmaipy import exporter
from nmaipy.damage_conflation_api import DamageConflationApi
from nmaipy.exporter import NearmapAIExporter
from nmaipy.feature_api import FeatureApi
from nmaipy.roof_age_api import RoofAgeApi

from .golden.compare import compare_exports
from .golden.fake_api import FIELD_PATHS
from .golden.regenerate import cache_paths
from .golden.replay import CASES, GOLDEN_DIR, emitted_paths, run_case, write_aois

REPO = Path(__file__).parents[1]
SIZE_CAP = 1024 * 1024
FORBIDDEN = re.compile(r"internal|sdSurveyId|taskId|webtile|bearer|apikey=(?!APIKEYREMOVED)", re.IGNORECASE)
EXPORT_CONFIG_KEYS = (
    "aoi_file",
    "packs",
    "classes",
    "include",
    "primary_decision",
    "aoi_grid_min_pct",
    "aoi_grid_inexact",
    "aoi_grid_cell_size",
    "regrid_on_skip",
    "processes",
    "threads",
    "chunk_size",
    "merge_read_workers",
    "include_parcel_geometry",
    "save_features",
    "tabular_file_format",
    "cache_dir",
    "no_cache",
    "overwrite_cache",
    "compress_cache",
    "country",
    "alpha",
    "beta",
    "prerelease",
    "only3d",
    "prefer3d",
    "since",
    "until",
    "url_root",
    "system_version_prefix",
    "system_version",
    "parcel_mode",
    "rapid",
    "order",
    "exclude_tiles_with_occlusion",
    "roof_age",
    "roof_age_dataset",
    "roof_age_resource_id",
    "class_level_files",
    "max_retries",
)
SAFE_TO_CHANGE = frozenset(
    {
        "cache_dir",
        "compress_cache",
        "log_level",
        "max_retries",
        "merge_read_workers",
        "no_cache",
        "overwrite_cache",
        "processes",
        "threads",
    }
)
COMPARED_AS_SETS = frozenset({"classes", "include", "packs"})
SQUARE = Polygon([(-74.0, 40.0), (-73.999, 40.0), (-73.999, 40.001), (-74.0, 40.001)])
ADDRESS = {"streetAddress": "1 Main St", "city": "Springfield", "state": "NJ", "zip": "07081"}
FEATURES_URL = "https://api.nearmap.com/ai/features/v4/bulk/features.json?apikey=dummy"
CACHE_PATHS = {
    "feature_polygon": "-74/40/f3d95bbe7e913c932a5c859c5ffa8c60.json",
    "feature_multipolygon": "-74/40/39e3f9dbc17cd3eb913380831ffebf57.json",
    "feature_compressed": "-74/40/f3d95bbe7e913c932a5c859c5ffa8c60.json.gz",
    "feature_address": "us/NJ/Springfield/07081/7a39b1da02b5dc33.json",
    "roof_age": "roofage/latest/none/none/10d4f48436b877450a335442d774c2ebadb2fab9283c9f8b93fad1c857497ea3.json",
    "roof_age_dated": "roofage/latest/2025-01-01/2020-01-01/10d4f48436b877450a335442d774c2ebadb2fab9283c9f8b93fad1c857497ea3.json",
    "roof_age_address": "roofage/latest/none/none/US/NJ/Springfield/07081/cc399247bbd1d063.json",
    "damage": "damageconflation/00000000-0000-0000-0000-000000000000/f80e0c14f54d72746adf61139bedbc8b2d9e40c0e686384b9c647e40c9da9f28.json",
}


@pytest.fixture(scope="module")
def golden_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Each case exported once offline; its `final/`, `markers/` and `cache/` are under the returned folder."""
    return {case: run_case(case, tmp_path_factory.mktemp(case)).parent for case in CASES}


@pytest.mark.parametrize("case", CASES)
def test_golden_export(case: str, golden_runs: dict[str, Path], tmp_path: Path) -> None:
    case_dir, run_dir = GOLDEN_DIR / case, golden_runs[case]
    differences = compare_exports(case_dir / "expected", run_dir / "final")
    assert differences == [], (
        "Export differs from the golden expected output (if the change is intended, re-baseline with "
        "`python -m tests.golden.regenerate`):\n" + "\n".join(differences)
    )
    assert cache_paths(run_dir) == (case_dir / "cache_paths.txt").read_text().splitlines(), "API cache layout changed"
    world_aois = write_aois(case, tmp_path / "aois.csv").read_text(encoding="utf-8")
    assert (case_dir / "aois.csv").read_text(encoding="utf-8") == world_aois, "aois.csv is stale: regenerate"


def test_fake_fields_are_allowlisted(golden_runs: dict[str, Path]) -> None:
    emitted: dict[str, set[str]] = {}
    for run_dir in golden_runs.values():
        for endpoint, paths in emitted_paths(run_dir / "markers").items():
            emitted.setdefault(endpoint, set()).update(paths)
    for endpoint, allowed in FIELD_PATHS.items():
        assert emitted.get(endpoint, set()) == allowed, (
            f"{endpoint}: emitted but not allowlisted {sorted(emitted.get(endpoint, set()) - allowed)}, "
            f"allowlisted but never emitted {sorted(allowed - emitted.get(endpoint, set()))}"
        )
    assert set(emitted) == set(FIELD_PATHS), f"unknown endpoints {set(emitted) - set(FIELD_PATHS)}"


def test_golden_data_is_small_text() -> None:
    listed = subprocess.run(
        ["git", "ls-files", "tests/data/golden", "tests/golden/vocabulary.json"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert any(p.startswith("tests/data/golden/") for p in listed), "no golden data is committed"
    total = 0
    for name in listed:
        data = (REPO / name).read_bytes()
        total += len(data)
        text = data.decode("utf-8")
        assert not (match := FORBIDDEN.search(text)), f"{name}: contains {match[0]!r}"
    assert total <= SIZE_CAP, f"golden data is {total} bytes, over {SIZE_CAP}"


def test_export_config_keys(tmp_path: Path) -> None:
    NearmapAIExporter(aoi_file=GOLDEN_DIR / "us" / "aois.csv", output_dir=tmp_path)
    config = json.loads((tmp_path / "final" / "export_config.json").read_text())
    assert tuple(config["parameters"]) == EXPORT_CONFIG_KEYS
    assert set(config["_metadata"]) == {"export_started_at", "nmaipy_version", "python_version", "platform"}
    assert exporter._CONFIG_KEYS_SAFE_TO_CHANGE == SAFE_TO_CHANGE
    assert exporter._CONFIG_KEYS_COMPARED_AS_SETS == COMPARED_AS_SETS


@pytest.mark.parametrize("kind", CACHE_PATHS)
def test_cache_key_layout(kind: str, tmp_path: Path) -> None:
    assert _cache_path(kind, tmp_path) == CACHE_PATHS[kind]


def test_compare_detects_changes(tmp_path: Path) -> None:
    expected = GOLDEN_DIR / "us" / "expected"
    actual = tmp_path / "actual"
    shutil.copytree(expected, actual)

    rollup = _rows(actual / "rollup.csv")
    line = next(i for i, row in enumerate(rollup) if row[0] == "building_count")
    rollup[line][1] = str(int(rollup[line][1]) + 1)
    as_float = next(i for i, row in enumerate(rollup) if row[0] == "roof_count")
    rollup[as_float][1] = f"{int(rollup[as_float][1])}.0"
    _write_rows(actual / "rollup.csv", rollup)
    building = _rows(actual / "building.csv")
    dropped_row = building[0][1]
    _write_rows(actual / "building.csv", [row[:1] + row[2:] for row in building])
    roof = _rows(actual / "roof.csv")
    roof_id = next(row for row in roof if row[0] == "primary_child_roof_age_feature_id")
    first_id = next(i for i, value in enumerate(roof_id) if i and value)
    roof_id[first_id] = str(int(roof_id[first_id]) + 1)
    _write_rows(actual / "roof.csv", [row for row in roof if row[0] != "area_sqft"])

    differences = compare_exports(expected, actual)
    assert any(d.startswith("rollup.csv: column building_count: 1 mismatches") for d in differences), differences
    assert any(d.startswith("rollup.csv: column roof_count: 1 mismatches") for d in differences), differences
    assert f"building.csv: missing rows [{dropped_row!r}]" in differences, differences
    assert "roof.csv: missing columns ['area_sqft']" in differences, differences
    assert any(
        d.startswith("roof.csv: column primary_child_roof_age_feature_id: 1 mismatches") for d in differences
    ), differences


def test_compare_detects_parquet_value_changes(golden_runs: dict[str, Path], tmp_path: Path) -> None:
    final = tmp_path / "final"
    shutil.copytree(golden_runs["us"] / "final", final)
    roofs = gpd.read_parquet(final / "roof_features.parquet")
    roofs.loc[roofs.index[0], "area_sqft"] += 1.0
    roofs.to_parquet(final / "roof_features.parquet")

    differences = compare_exports(GOLDEN_DIR / "us" / "expected", final)
    assert "roof_features.parquet.csv: extra columns ['area_sqft']" in differences, differences


@pytest.mark.live_api
def test_drift() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tests.golden.drift"], cwd=REPO, capture_output=True, text=True, check=False
    )
    output = f"{result.stdout}\n{result.stderr}"
    if key := os.environ.get("API_KEY"):
        output = output.replace(key, "APIKEYREMOVED")
    assert result.returncode == 0, f"the live API drifted from the golden fake:\n{output}"


def _rows(path: Path) -> list[list[str]]:
    return list(csv.reader(path.open(encoding="utf-8", newline="")))


def _write_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def _cache_path(kind: str, cache_dir: Path) -> str:
    polygon = {"aoi": mapping(SQUARE), "packs": "building"}
    multipolygon = {"aoi": {"type": "MultiPolygon", "coordinates": [mapping(SQUARE)["coordinates"]]}}
    if kind.startswith("feature"):
        api = FeatureApi(api_key="dummy", cache_dir=cache_dir, compress_cache=kind == "feature_compressed")
        if kind == "feature_address":
            address = "&".join(f"{k}={v}" for k, v in [("country", "us"), *ADDRESS.items()] if k != "streetAddress")
            path = api._post_request_cache_path(f"{FEATURES_URL}&{address}", {"packs": "building"})
        else:
            path = api._post_request_cache_path(
                FEATURES_URL, multipolygon if kind == "feature_multipolygon" else polygon
            )
    elif kind.startswith("roof_age"):
        api = RoofAgeApi(api_key="dummy", cache_dir=cache_dir)
        key = api._build_cache_key(address=ADDRESS) if kind == "roof_age_address" else api._build_cache_key(aoi=SQUARE)
        dates = dict(until_as_of_date="2025-01-01", since_as_of_date="2020-01-01") if kind == "roof_age_dated" else {}
        path = api._get_cache_path(key, **dates)
    else:
        api = DamageConflationApi(event_id="00000000-0000-0000-0000-000000000000", api_key="dummy", cache_dir=cache_dir)
        path = api._get_cache_path(api._build_cache_key(aoi=SQUARE))
    return str(Path(path).relative_to(cache_dir))
