"""
Tests for the Damage Conflation exporter.

Mocks at the process_chunk / API-class level (mocks don't cross the multiprocessing
boundary) and feeds REAL parsed fixture data — never hand-crafted mock data.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from nmaipy import parcels, storage
from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS
from nmaipy.damage_conflation_api import DamageConflationApi
from nmaipy.damage_conflation_exporter import DamageConflationExporter

EVENT_ID = "2f510853-5d55-50f4-9102-2c02de08190e"


@pytest.fixture
def milton_response(data_directory: Path):
    with open(data_directory / "test_damage_conflation_milton_response.json", "r") as f:
        return json.load(f)


@pytest.fixture
def chunk_inputs(milton_response):
    """A features_gdf (two AOIs) + matching aoi_gdf + metadata, like get_damage_bulk returns."""
    api = DamageConflationApi(event_id=EVENT_ID, api_key="t")
    f1 = api._parse_response({**milton_response, "features": milton_response["features"][:6]}, "p1")
    f2 = api._parse_response({**milton_response, "features": milton_response["features"][6:]}, "p2")
    features_gdf = gpd.GeoDataFrame(pd.concat([f1, f2], ignore_index=True), crs=API_CRS)

    aoi_gdf = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1), box(0, 0, 1, 1)],
        crs=API_CRS,
        index=pd.Index(["p1", "p2"], name=AOI_ID_COLUMN_NAME),
    )
    metadata_df = pd.DataFrame(
        {"event_uuid": [milton_response["eventUuid"], milton_response["eventUuid"]]},
        index=pd.Index(["p1", "p2"], name=AOI_ID_COLUMN_NAME),
    )
    return aoi_gdf, features_gdf, metadata_df


def _make_exporter(tmp_path, **kwargs):
    return DamageConflationExporter(
        aoi_file=str(tmp_path / "aois.csv"),  # not read for direct process_chunk tests
        output_dir=str(tmp_path / "out"),
        event_id=EVENT_ID,
        api_key="test_key",
        processes=1,
        **kwargs,
    )


def test_exporter_initialization(tmp_path):
    exporter = _make_exporter(tmp_path, output_format="both", rollup=True)
    assert exporter.event_id == EVENT_ID
    assert exporter.rollup is True
    # config written at init
    assert (Path(exporter.final_path) / "damage_conflation_export_config.json").exists()


def test_exporter_requires_event_id(tmp_path):
    with pytest.raises(ValueError, match="event_id is required"):
        DamageConflationExporter(aoi_file="x", output_dir=str(tmp_path / "o"), event_id="", api_key="t")


def test_process_chunk_writes_features_and_metadata(tmp_path, chunk_inputs):
    aoi_gdf, features_gdf, metadata_df = chunk_inputs
    exporter = _make_exporter(tmp_path, output_format="both")
    Path(exporter.chunk_path).mkdir(parents=True, exist_ok=True)

    with patch("nmaipy.damage_conflation_exporter.DamageConflationApi") as mock_cls:
        mock_api = Mock()
        mock_cls.return_value = mock_api
        mock_api.get_damage_bulk.return_value = (features_gdf.copy(), metadata_df.copy(), pd.DataFrame())
        mock_api.get_latency_stats.return_value = None

        exporter.process_chunk("0000", aoi_gdf)

    feats = Path(exporter.chunk_path) / "damage_features_0000.parquet"
    meta = Path(exporter.chunk_path) / "metadata_0000.parquet"
    assert feats.exists() and meta.exists()
    result = gpd.read_parquet(feats)
    assert len(result) == len(features_gdf)
    for col in ("feature_id", "damage_event_rating", "event_uuid"):
        assert col in result.columns
    # no rollup file when rollup is off
    assert not (Path(exporter.chunk_path) / "damage_rollup_0000.parquet").exists()


def test_process_chunk_does_not_write_rollup(tmp_path, chunk_inputs):
    """Rollup is derived at combine time (_run_inner), not per chunk — so even with
    rollup=True, process_chunk writes no per-chunk rollup file. This is what keeps the
    rollup correct when --rollup is enabled on a resumed run with cached chunks."""
    aoi_gdf, features_gdf, metadata_df = chunk_inputs
    exporter = _make_exporter(tmp_path, output_format="both", rollup=True)
    Path(exporter.chunk_path).mkdir(parents=True, exist_ok=True)

    with patch("nmaipy.damage_conflation_exporter.DamageConflationApi") as mock_cls:
        mock_api = Mock()
        mock_cls.return_value = mock_api
        mock_api.get_damage_bulk.return_value = (features_gdf.copy(), metadata_df.copy(), pd.DataFrame())
        mock_api.get_latency_stats.return_value = None

        exporter.process_chunk("0000", aoi_gdf)

    assert not (Path(exporter.chunk_path) / "damage_rollup_0000.parquet").exists()
    # features + metadata still written
    assert (Path(exporter.chunk_path) / "damage_features_0000.parquet").exists()
    assert (Path(exporter.chunk_path) / "metadata_0000.parquet").exists()


def test_process_chunk_writes_errors(tmp_path):
    exporter = _make_exporter(tmp_path)
    Path(exporter.chunk_path).mkdir(parents=True, exist_ok=True)
    aoi_gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=API_CRS, index=pd.Index(["x"], name=AOI_ID_COLUMN_NAME))
    empty = gpd.GeoDataFrame(columns=[AOI_ID_COLUMN_NAME, "geometry"], crs=API_CRS)
    errors_df = pd.DataFrame(
        [{"status_code": 403, "message": "no access"}], index=pd.Index(["x"], name=AOI_ID_COLUMN_NAME)
    )

    with patch("nmaipy.damage_conflation_exporter.DamageConflationApi") as mock_cls:
        mock_api = Mock()
        mock_cls.return_value = mock_api
        mock_api.get_damage_bulk.return_value = (empty, pd.DataFrame(), errors_df)
        mock_api.get_latency_stats.return_value = None
        exporter.process_chunk("0000", aoi_gdf)

    assert (Path(exporter.chunk_path) / "damage_errors_0000.parquet").exists()


def test_save_outputs_drops_wrong_unit_area_and_writes_files(tmp_path, chunk_inputs):
    aoi_gdf, features_gdf, metadata_df = chunk_inputs
    exporter = _make_exporter(tmp_path, output_format="both", rollup=True, country="us")
    rollup_df = parcels.conflation_rollup(
        aoi_gdf, features_gdf, country="us", successful_aoi_ids=set(metadata_df.index)
    )

    exporter._save_outputs(features_gdf.copy(), rollup_df, metadata_df, pd.DataFrame(), exporter.final_path)

    final = Path(exporter.final_path)
    assert (final / "damage_buildings.parquet").exists()
    assert (final / "damage_buildings.csv").exists()
    assert (final / "damage_rollup.parquet").exists()
    assert (final / "damage_rollup.csv").exists()

    # US -> imperial: area_sqm dropped, area_sqft kept.
    buildings = gpd.read_parquet(final / "damage_buildings.parquet")
    assert "area_sqft" in buildings.columns
    assert "area_sqm" not in buildings.columns

    csv = pd.read_csv(final / "damage_buildings.csv")
    assert "damage_event_rating" in csv.columns
    assert "geometry" in csv.columns
    # Previously silently dropped vs. the parquet — now carried through to the CSV too.
    for col in ("hilbert_id", "damage_event_latest_capture_date", "geomatched_address"):
        assert col in csv.columns
    # The verbose JSON classRatios columns stay parquet-only by design.
    assert "damage_event_class_ratios" not in csv.columns


def test_combine_reads_present_chunks_and_preserves_index(tmp_path, chunk_inputs):
    """Combine concatenates the per-chunk files that are present and round-trips the aoi_id index."""
    aoi_gdf, features_gdf, metadata_df = chunk_inputs
    exporter = _make_exporter(tmp_path, output_format="both")
    Path(exporter.chunk_path).mkdir(parents=True, exist_ok=True)
    storage.write_parquet(metadata_df, str(Path(exporter.chunk_path) / "metadata_0000.parquet"))
    storage.write_parquet(features_gdf, str(Path(exporter.chunk_path) / "damage_features_0000.parquet"))

    md = exporter.combine_chunk_files("metadata", num_chunks=1)
    feats = exporter.combine_chunk_files("damage_features", num_chunks=1, geo=True)

    # aoi_id index must survive the round-trip — successful_aoi_ids = set(metadata_df.index)
    # depends on it, and a lost index would null out the whole rollup.
    assert set(md.index) == set(metadata_df.index)
    assert len(feats) == len(features_gdf)


def test_combine_recovers_from_stale_listing(tmp_path, chunk_inputs):
    """Deterministically reproduce the reported S3 bug: a stale listing makes file_exists
    return False for present chunks until the cache is invalidated. The combine must
    invalidate BEFORE its existence sweep, so it still reads the chunks. Without the fix
    (no invalidate), file_exists stays False and the combine reads 0 records — this test fails."""
    _, features_gdf, metadata_df = chunk_inputs
    exporter = _make_exporter(tmp_path)
    Path(exporter.chunk_path).mkdir(parents=True, exist_ok=True)
    storage.write_parquet(metadata_df, str(Path(exporter.chunk_path) / "metadata_0000.parquet"))

    real_exists = storage.file_exists
    state = {"fresh": False}  # listing starts stale (mirrors s3fs after an earlier empty ls)

    def fake_exists(path):
        return real_exists(path) if state["fresh"] else False

    def fake_invalidate(path):
        state["fresh"] = True

    with (
        patch("nmaipy.storage.file_exists", side_effect=fake_exists),
        patch("nmaipy.storage.invalidate_cache", side_effect=fake_invalidate),
    ):
        md = exporter.combine_chunk_files("metadata", num_chunks=1)

    assert set(md.index) == set(metadata_df.index)  # recovered only because invalidate ran first


def test_save_outputs_writes_consolidated_errors(tmp_path, chunk_inputs):
    """Per-chunk errors are consolidated into final/damage_errors.csv (status_code + message)."""
    _, features_gdf, metadata_df = chunk_inputs
    exporter = _make_exporter(tmp_path, output_format="both")
    errors_df = pd.DataFrame(
        [{"status_code": 429, "message": "rate limited"}],
        index=pd.Index(["bad"], name=AOI_ID_COLUMN_NAME),
    )
    exporter._save_outputs(features_gdf.copy(), pd.DataFrame(), metadata_df, errors_df, exporter.final_path)

    errs = Path(exporter.final_path) / "damage_errors.csv"
    assert errs.exists()
    out = pd.read_csv(errs)
    assert "status_code" in out.columns and "message" in out.columns
    assert (out["status_code"] == 429).any()


def test_save_outputs_skips_rollup_when_disabled(tmp_path, chunk_inputs):
    aoi_gdf, features_gdf, metadata_df = chunk_inputs
    exporter = _make_exporter(tmp_path, output_format="both", rollup=False)
    exporter._save_outputs(features_gdf.copy(), pd.DataFrame(), metadata_df, pd.DataFrame(), exporter.final_path)
    final = Path(exporter.final_path)
    assert (final / "damage_buildings.parquet").exists()
    assert not (final / "damage_rollup.parquet").exists()


def test_run_end_to_end_consolidates_and_rolls_up(tmp_path, milton_response):
    """End-to-end DamageConflationExporter.run(): read AOI file -> split -> (parallel
    processing) -> combine chunks -> rollup -> save. This is the whole consolidation ->
    rollup -> save path that previously failed silently (all-FALSE rollup). Chunks are
    pre-seeded (as the workers would have written them) and run_parallel is mocked, so no
    processes spawn and no API calls are made — the real code under test is _run_inner's
    combine + conflation_rollup + _save_outputs."""
    # AOI file with two property-sized AOIs (ids p1/p2) matching the seeded chunk data.
    aoi_csv = tmp_path / "aois.csv"
    pd.DataFrame(
        {"aoi_id": ["p1", "p2"], "geometry": [box(0, 0, 0.001, 0.001).wkt, box(0, 0, 0.001, 0.001).wkt]}
    ).to_csv(aoi_csv, index=False)

    exporter = DamageConflationExporter(
        aoi_file=str(aoi_csv),
        output_dir=str(tmp_path / "out"),
        event_id=EVENT_ID,
        api_key="test_key",
        output_format="both",
        rollup=True,
        processes=1,
    )

    # Seed one chunk exactly as get_damage_bulk would have written it.
    api = DamageConflationApi(event_id=EVENT_ID, api_key="t")
    f1 = api._parse_response({**milton_response, "features": milton_response["features"][:6]}, "p1")
    f2 = api._parse_response({**milton_response, "features": milton_response["features"][6:]}, "p2")
    features = gpd.GeoDataFrame(pd.concat([f1, f2], ignore_index=True), crs=API_CRS)
    metadata = pd.DataFrame(
        {"event_uuid": [milton_response["eventUuid"]] * 2},
        index=pd.Index(["p1", "p2"], name=AOI_ID_COLUMN_NAME),
    )
    Path(exporter.chunk_path).mkdir(parents=True, exist_ok=True)
    storage.write_parquet(features, str(Path(exporter.chunk_path) / "damage_features_0000.parquet"))
    storage.write_parquet(metadata, str(Path(exporter.chunk_path) / "metadata_0000.parquet"))

    with (
        patch.object(exporter, "run_parallel", return_value=[]),
        patch("nmaipy.damage_conflation_exporter.combine_chunk_latency_stats", return_value=[]),
    ):
        exporter.run()

    final = Path(exporter.final_path)
    # All declared outputs land in final/.
    for name in ("damage_buildings.parquet", "damage_buildings.csv", "damage_rollup.parquet", "damage_rollup.csv"):
        assert (final / name).exists(), f"missing {name}"

    # Per-building output has every building.
    buildings = gpd.read_parquet(final / "damage_buildings.parquet")
    assert len(buildings) == len(features)

    # Rollup is correct — NOT all-FALSE (the regression this branch exists to prevent).
    rollup = pd.read_csv(final / "damage_rollup.csv")
    assert set(rollup["aoi_id"]) == {"p1", "p2"}
    assert rollup["query_succeeded"].all()
    assert rollup["n_buildings"].sum() == len(features)
    assert rollup["primary_damage_event_rating"].notna().all()
