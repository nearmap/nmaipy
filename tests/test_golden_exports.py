"""Golden exports: the full exporter run offline on recorded API responses, compared to stored outputs.

See tests/data/golden/README.md for the cases, the recording policy and how to re-baseline or re-record.
"""

import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import Polygon, mapping

from nmaipy import exporter
from nmaipy.damage_conflation_api import DamageConflationApi
from nmaipy.exporter import NearmapAIExporter
from nmaipy.feature_api import FeatureApi
from nmaipy.roof_age_api import RoofAgeApi

from .golden.compare import canonicalise_export, compare_exports
from .golden.replay import (
    executor_with,
    install_recorder,
    install_replay,
    load_exchanges,
    recording,
    serving,
    write_exchanges,
)

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"
CLASS_DESCRIPTIONS = Path(exporter.__file__).parent / "class_descriptions.json"

COMMON = dict(
    processes=1,
    threads=1,
    chunk_size=100,
    max_retries=0,
    save_features=True,
    class_level_files=True,
    include_parcel_geometry=True,
)
PUBLIC_PACKS = frozenset(
    {"building", "building_char", "roof_char", "roof_cond", "surfaces", "vegetation", "pool", "solar"}
)
CASES: dict[str, dict] = {
    "us_parcels": dict(country="us", packs=sorted(PUBLIC_PACKS), roof_age=True),
    "au_parcels": dict(country="au", packs=["building", "roof_char"]),
    "au_gridded": dict(country="au", packs=["building"], since="2025-03-06", until="2025-03-06"),
}

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


@pytest.fixture
def offline_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No credentials, a dead proxy, a fixed hash seed for workers, and a guard on the bundled class catalogue."""
    _go_offline(monkeypatch)
    with _class_descriptions_unchanged("a golden run rewrote nmaipy/class_descriptions.json"):
        yield


@pytest.mark.parametrize("case", CASES)
def test_golden_export(case: str, tmp_path: Path, offline_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    case_dir = GOLDEN_DIR / case
    final = _run_replayed(case, tmp_path, monkeypatch)
    differences = compare_exports(case_dir / "expected", final)
    assert differences == [], (
        "Export differs from the golden expected output (if the change is intended, re-baseline with "
        "test_regenerate_golden_expected):\n" + "\n".join(differences)
    )
    expected_paths = (case_dir / "cache_paths.txt").read_text().splitlines()
    assert _cache_paths(tmp_path) == expected_paths, "API cache layout changed"


def test_recordings_are_public_safe() -> None:
    bundled_classes = set(json.loads(CLASS_DESCRIPTIONS.read_text()))
    for case in CASES:
        exchanges = load_exchanges(GOLDEN_DIR / case).values()
        assert exchanges, f"{case} has no recordings"
        packs = [
            p for e in exchanges if e["request"]["url"].endswith("/packs.json") for p in e["response"]["body"]["packs"]
        ]
        assert {p["code"] for p in packs} <= PUBLIC_PACKS, f"{case}: non-public pack in packs.json"
        pack_classes = {c["id"] for p in packs for c in p["featureClasses"]}
        for path in sorted((GOLDEN_DIR / case / "exchanges").glob("*.json")):
            text = path.read_text(encoding="utf-8")
            request, response = json.loads(text).values()
            assert "?" not in request["url"], f"{path}: query string in url"
            assert "apikey" not in text.lower(), f"{path}: api key"
            assert "bearer" not in text.lower(), f"{path}: bearer token"
            query = dict(request["query"])
            assert not {k.lower() for k in query} & {"alpha", "beta", "prerelease"}, f"{path}: non-public flag"
            assert set(query.get("packs", "").split(",")) - {""} <= PUBLIC_PACKS, f"{path}: non-public pack"
            body = response["body"] if isinstance(response["body"], dict) else {}
            catalogue = {c["id"] for c in body.get("classes", [])}
            features = {f["classId"] for f in body.get("features", []) if "classId" in f}
            unexpected = (catalogue | features) - (pack_classes & bundled_classes)
            assert not unexpected, f"{path}: classes outside the recorded packs or class_descriptions.json {unexpected}"


def test_export_config_keys(tmp_path: Path) -> None:
    NearmapAIExporter(aoi_file=GOLDEN_DIR / "us_parcels" / "aois.csv", output_dir=tmp_path)
    config = json.loads((tmp_path / "final" / "export_config.json").read_text())
    assert tuple(config["parameters"]) == EXPORT_CONFIG_KEYS
    assert set(config["_metadata"]) == {"export_started_at", "nmaipy_version", "python_version", "platform"}
    assert exporter._CONFIG_KEYS_SAFE_TO_CHANGE == SAFE_TO_CHANGE
    assert exporter._CONFIG_KEYS_COMPARED_AS_SETS == COMPARED_AS_SETS


@pytest.mark.parametrize("kind", CACHE_PATHS)
def test_cache_key_layout(kind: str, tmp_path: Path) -> None:
    assert _cache_path(kind, tmp_path) == CACHE_PATHS[kind]


def test_compare_detects_changes(tmp_path: Path) -> None:
    expected = GOLDEN_DIR / "us_parcels" / "expected"
    actual = tmp_path / "actual"
    shutil.copytree(expected, actual)
    rollup = pd.read_csv(actual / "rollup.csv", dtype=str, keep_default_na=False)
    rollup.loc[0, "building_count"] = str(int(rollup.loc[0, "building_count"]) + 1)
    rollup.to_csv(actual / "rollup.csv", index=False)
    building = pd.read_csv(actual / "building.csv", dtype=str, keep_default_na=False)
    building.iloc[1:].to_csv(actual / "building.csv", index=False)
    roof = pd.read_csv(actual / "roof.csv", dtype=str, keep_default_na=False)
    roof.drop(columns="area_sqft").to_csv(actual / "roof.csv", index=False)

    differences = compare_exports(expected, actual)
    assert any(d.startswith("rollup.csv: column building_count") for d in differences), differences
    assert any(d.startswith("building.csv: row count") for d in differences), differences
    assert any(d.startswith("roof.csv: missing columns ['area_sqft']") for d in differences), differences


@pytest.mark.skip("Comment out this line if you wish to regen the test data")
def test_regenerate_golden_expected(tmp_path: Path, offline_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    for case in CASES:
        _regenerate(case, tmp_path / case, monkeypatch)


@pytest.mark.skip("Comment out this line if you wish to regen the test data")
@pytest.mark.live_api
def test_record_golden_exchanges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    guard = "the live catalogue differs from nmaipy/class_descriptions.json; refresh it in a separate change first"
    for case in CASES:
        case_dir, record_dir, live_dir = GOLDEN_DIR / case, tmp_path / case / "record", tmp_path / case / "live"
        record_dir.mkdir(parents=True)
        with _class_descriptions_unchanged(guard), monkeypatch.context() as m:
            m.setattr("nmaipy.base_exporter.ProcessPoolExecutor", executor_with(install_recorder, str(record_dir)))
            with recording(record_dir):
                _export(case, live_dir)
        write_exchanges(record_dir, case_dir, CASES[case]["packs"])
        with _class_descriptions_unchanged(guard), monkeypatch.context() as m:
            _go_offline(m)
            _regenerate(case, tmp_path / case / "replay", m)
        differences = compare_exports(case_dir / "expected", live_dir)
        assert differences == [], f"{case}: replay differs from the live export:\n" + "\n".join(differences)


def _go_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)


@contextmanager
def _class_descriptions_unchanged(message: str) -> Iterator[None]:
    original = CLASS_DESCRIPTIONS.read_bytes()
    try:
        yield
    finally:
        changed = CLASS_DESCRIPTIONS.read_bytes() != original
        if changed:
            CLASS_DESCRIPTIONS.write_bytes(original)
    if changed:
        pytest.fail(message)


def _export(case: str, out_dir: Path) -> None:
    aoi_file = GOLDEN_DIR / case / "aois.csv"
    NearmapAIExporter(aoi_file=aoi_file, output_dir=out_dir, cache_dir=out_dir / "cache", **COMMON, **CASES[case]).run()


def _run_replayed(case: str, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    case_dir, marker_dir = GOLDEN_DIR / case, out_dir / "markers"
    marker_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "nmaipy.base_exporter.ProcessPoolExecutor", executor_with(install_replay, str(case_dir), str(marker_dir))
    )
    with serving(case_dir, marker_dir):
        try:
            _export(case, out_dir)
        finally:
            _assert_replay_complete(marker_dir)
    used = {line for p in marker_dir.glob("*.used") for line in p.read_text().splitlines()}
    unused = sorted(set(load_exchanges(case_dir)) - used)
    assert not unused, "recorded requests the export never made:\n" + "\n".join(unused)
    return out_dir / "final"


def _assert_replay_complete(marker_dir: Path) -> None:
    installed = {int(p.stem) for p in marker_dir.glob("*.installed")} - {os.getpid()}
    assert installed, "replay not installed in the workers"
    misses = sorted({line for p in marker_dir.glob("*.misses") for line in p.read_text().splitlines()})
    assert not misses, "requests with no golden recording:\n" + "\n".join(misses)


def _regenerate(case: str, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case_dir = GOLDEN_DIR / case
    canonicalise_export(_run_replayed(case, out_dir, monkeypatch), case_dir / "expected")
    (case_dir / "cache_paths.txt").write_text("".join(f"{p}\n" for p in _cache_paths(out_dir)))


def _cache_paths(out_dir: Path) -> list[str]:
    root = out_dir / "cache" / "cache"
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


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
