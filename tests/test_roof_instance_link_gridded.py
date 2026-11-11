"""Gridded AOIs link Roof Age instances by the roof outline clipped to the AOI."""

import logging
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, box

from nmaipy import parcels
from nmaipy.constants import AOI_ID_COLUMN_NAME, API_CRS, ROOF_ID, ROOF_INSTANCE_CLASS_ID
import nmaipy.exporter as exporter_mod
from nmaipy.exporter import AOIExporter, _clip_roof_geoms_to_aoi, _link_roof_instances

_AOI = box(0, 0, 10, 10)
# A row roof crossing the AOI boundary; Roof Age's lot line is 0.05 inside it, so the neighbour's lot overlaps the AOI.
_ROW_ROOF = box(8, 1, 14, 2)
_OUR_LOT = box(8, 1, 9.95, 2)
_NEIGHBOURS_LOT = box(9.95, 1, 14, 2)
_SHED = box(1, 5, 2, 6)


def _gdf(rows):
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=API_CRS).set_index("aoi_id")


def _roofs(*rows):
    return _gdf([{"aoi_id": aoi_id, "feature_id": fid, "geometry": geom} for aoi_id, fid, geom in rows])


def _instances(*rows):
    return _gdf([{"aoi_id": aoi_id, "feature_id": fid, "kind": "roof", "geometry": geom} for aoi_id, fid, geom in rows])


def _row_instances(aoi_id):
    return [(aoi_id, f"{aoi_id}-ours", _OUR_LOT), (aoi_id, f"{aoi_id}-neighbour", _NEIGHBOURS_LOT)]


def _by_feature_id(roofs_linked):
    return roofs_linked.reset_index().set_index("feature_id")


def test_whole_outline_links_the_larger_neighbouring_lot():
    """The bug: against the whole row outline, the neighbour's larger lot wins (IoU 0.675 against 0.325)."""
    _, roofs_linked = parcels.link_roof_instances_to_roofs(
        _instances(*_row_instances("a")), _roofs(("a", "row", _ROW_ROOF))
    )
    row = _by_feature_id(roofs_linked).loc["row"]
    assert row["primary_child_roof_age_feature_id"] == "a-neighbour"
    assert row["primary_child_roof_age_iou"] == pytest.approx(0.675, abs=1e-3)


def test_gridded_link_compares_the_aoi_clipped_outline():
    """On a gridded AOI the row links this lot's instance by its in-AOI outline, and keeps its whole outline."""
    instances_linked, roofs_linked = _link_roof_instances(
        _instances(*_row_instances("a")), _roofs(("a", "row", _ROW_ROOF)), {"a": _AOI}
    )
    instances = instances_linked.reset_index().set_index("feature_id")
    assert instances.loc["a-ours", "parent_id"] == "row"
    assert instances.loc["a-ours", "parent_iou"] == pytest.approx(0.975, abs=1e-3)
    # The neighbour's lot overlaps the clipped roof by 0.05 of its 4.05: IoU 0.05 / 6.0.
    assert instances.loc["a-neighbour", "parent_iou"] == pytest.approx(0.05 / 6.0, abs=1e-4)
    assert instances.loc["a-neighbour", "geometry"].equals(_NEIGHBOURS_LOT), "Instances are never clipped"
    row = _by_feature_id(roofs_linked).loc["row"]
    assert row["primary_child_roof_age_feature_id"] == "a-ours"
    assert row["primary_child_roof_age_iou"] == pytest.approx(0.975, abs=1e-3)
    assert row["geometry"].equals(_ROW_ROOF), "The linked roof must keep its whole outline"


def test_only_gridded_aois_are_clipped_and_every_roof_keeps_its_outline():
    """AOIs outside the map link by the whole outline; every roof comes back whole, in input order."""
    roofs = _roofs(("a", "a-row", _ROW_ROOF), ("b", "b-row", _ROW_ROOF), ("a", "a-shed", _SHED))
    instances = _instances(*_row_instances("a"), *_row_instances("b"), ("a", "a-shed-instance", _SHED))
    _, roofs_linked = _link_roof_instances(instances, roofs, {"a": _AOI})

    assert list(roofs_linked["feature_id"]) == ["a-row", "b-row", "a-shed"], "Roofs must keep their input order"
    assert all(
        linked.equals(original) for linked, original in zip(roofs_linked.geometry, roofs.geometry, strict=True)
    ), "Every linked roof must keep its whole outline"
    by_fid = _by_feature_id(roofs_linked)
    assert by_fid.loc["a-row", "primary_child_roof_age_feature_id"] == "a-ours"
    assert by_fid.loc["b-row", "primary_child_roof_age_feature_id"] == "b-neighbour"
    assert by_fid.loc["a-shed", "primary_child_roof_age_feature_id"] == "a-shed-instance"


def test_an_invalid_roof_keeps_its_outline_without_unclipping_the_rest_of_its_aoi(caplog):
    """One invalid roof fails GEOS's vectorized clip; only that roof keeps its whole outline.

    A roof with no geometry in the same AOI passes through as None and isn't counted as a failure.
    """
    bowtie = Polygon([(8, 4), (12, 6), (12, 4), (8, 6), (8, 4)])
    roofs = _roofs(("a", "row", _ROW_ROOF), ("a", "bowtie", bowtie), ("a", "no-geometry", None))

    nmaipy_logger = logging.getLogger("nmaipy")
    nmaipy_logger.addHandler(caplog.handler)  # propagate=False, so attach caplog's handler directly
    try:
        with caplog.at_level(logging.WARNING, logger="nmaipy"):
            clipped = _clip_roof_geoms_to_aoi(roofs, {"a": _AOI}, "Roof Age link")
    finally:
        nmaipy_logger.removeHandler(caplog.handler)

    assert clipped.iloc[0].equals(box(8, 1, 10, 2)), "The valid roof must still be clipped to its AOI"
    assert clipped.iloc[1].equals(bowtie), "The invalid roof must keep its whole outline"
    assert clipped.iloc[2] is None, "A roof with no geometry must pass through untouched"
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1, f"Expected one warning, got {messages}"
    assert messages[0].startswith("Roof Age link: could not clip 1 of 3 roof(s) to AOI a"), messages[0]
    assert messages[0].endswith("Feature IDs: ['bowtie']"), messages[0]


@pytest.mark.parametrize(
    ("roof_rows", "reorder"),
    [
        ([("a", "a-row", _ROW_ROOF), ("a", "a-shed", _SHED)], lambda df: df.iloc[::-1]),
        # One roof spanning two AOIs: the feature_id sequence is identical after the swap.
        ([("b", "row", _ROW_ROOF), ("a", "row", _ROW_ROOF)], lambda df: df.sort_index(kind="stable")),
        ([("a", "a-row", _ROW_ROOF), ("a", "a-shed", _SHED)], lambda df: df.iloc[:1]),
    ],
    ids=["reversed", "same-feature-in-two-aois", "row-dropped"],
)
def test_link_refuses_to_restore_outlines_onto_reordered_roofs(monkeypatch, roof_rows, reorder):
    """Whole outlines go back by position, so a link that moved the roofs must fail, not swap outlines."""
    real_link = parcels.link_roof_instances_to_roofs

    def reordering_link(roof_age_gdf, roofs_gdf):
        instances_linked, roofs_linked = real_link(roof_age_gdf, roofs_gdf)
        return instances_linked, reorder(roofs_linked)

    monkeypatch.setattr(parcels, "link_roof_instances_to_roofs", reordering_link)
    instances = _instances(*_row_instances("a"), *_row_instances("b"))
    with pytest.raises(RuntimeError, match="reordered or changed the roofs"):
        _link_roof_instances(instances, _roofs(*roof_rows), {"a": _AOI, "b": _AOI})


class TestProcessChunkWiring:
    """process_chunk must hand the Roof Age link a clip map holding exactly its gridded AOIs.

    Drives process_chunk with mocked Feature API and Roof Age responses: AOI p1 was gridded, p2 was not. Both hold
    the same row roof across their boundary, with Roof Age's lot line just inside it.
    """

    # The unit-square geometry above, placed in lon/lat: 1 unit = 1e-5 degrees, each AOI at its own origin.
    _ORIGINS: ClassVar[dict[str, tuple[float, float]]] = {"p1": (-75.0, 40.0), "p2": (-75.01, 40.0)}

    @classmethod
    def _place(cls, geom, aoi_id):
        x0, y0 = cls._ORIGINS[aoi_id]
        return affine_transform(geom, [1e-5, 0, 0, 1e-5, x0, y0])

    @pytest.fixture
    def exporter(self, tmp_path):
        ex = AOIExporter(
            aoi_file="data/examples/sydney_parcels.geojson",  # not read: process_chunk is driven directly
            output_dir=str(tmp_path),
            country="us",
            packs=["building"],
            roof_age=True,
            chunk_size=2,
            no_cache=True,
        )
        Path(ex.chunk_path).mkdir(parents=True, exist_ok=True)
        return ex

    def _aoi_gdf(self):
        return gpd.GeoDataFrame(
            geometry=[self._place(_AOI, a) for a in ("p1", "p2")],
            crs=API_CRS,
            index=pd.Index(["p1", "p2"], name=AOI_ID_COLUMN_NAME),
        )

    def _feature_api_response(self):
        aoi_ids = ["p1", "p2"]
        features = gpd.GeoDataFrame(
            {
                "class_id": [ROOF_ID] * 2,
                "feature_id": [f"{a}-row" for a in aoi_ids],
                "parent_id": [None] * 2,
                "confidence": [0.9] * 2,
                "fidelity": [0.9] * 2,
                "area_sqm": [100.0] * 2,
                "clipped_area_sqm": [100.0] * 2,
                "unclipped_area_sqm": [100.0] * 2,
                "area_sqft": [1076.4] * 2,
                "clipped_area_sqft": [1076.4] * 2,
                "unclipped_area_sqft": [1076.4] * 2,
                "survey_date": ["2024-01-01"] * 2,
                "mesh_date": ["2024-01-01"] * 2,
                "description": ["Roof"] * 2,
                "attributes": [[]] * 2,
                "belongs_to_parcel": [True] * 2,
            },
            geometry=[self._place(_ROW_ROOF, a) for a in aoi_ids],
            crs=API_CRS,
            index=pd.Index(aoi_ids, name=AOI_ID_COLUMN_NAME),
        )
        metadata = pd.DataFrame(
            {
                "system_version": ["gen6-x"] * 2,
                "survey_date": ["2024-01-01"] * 2,
                "gridded": [True, np.nan],
            },
            index=pd.Index(aoi_ids, name=AOI_ID_COLUMN_NAME),
        )
        errors = pd.DataFrame([])
        errors.index.name = AOI_ID_COLUMN_NAME
        return features, metadata, errors

    def _roof_age_response(self):
        rows = [
            (a, f"{a}-{lot}", self._place(geom, a))
            for a in ("p1", "p2")
            for lot, geom in (("ours", _OUR_LOT), ("neighbour", _NEIGHBOURS_LOT))
        ]
        instances = gpd.GeoDataFrame(
            {
                AOI_ID_COLUMN_NAME: [a for a, _, _ in rows],
                "feature_id": [fid for _, fid, _ in rows],
                "class_id": [ROOF_INSTANCE_CLASS_ID] * len(rows),
                "kind": ["roof"] * len(rows),
                "area_sqm": [50.0] * len(rows),
            },
            geometry=[g for _, _, g in rows],
            crs=API_CRS,
        )
        # A real response has one metadata row per answered AOI, and no error rows.
        metadata = pd.DataFrame({"resource_id": ["r"] * 2}, index=pd.Index(["p1", "p2"], name=AOI_ID_COLUMN_NAME))
        return instances, metadata, pd.DataFrame()

    def test_only_gridded_aois_are_clipped_and_the_chunk_writes_their_links(self, exporter, monkeypatch):
        clips = set()
        real_clip = exporter_mod._clip_roof_geoms_to_aoi

        def recording_clip(parent_gdf, aoi_geometries, purpose):
            clips.add((purpose, tuple(sorted(aoi_geometries))))
            assert aoi_geometries["p1"].equals(self._place(_AOI, "p1")), "The clip map must hold p1's AOI polygon"
            return real_clip(parent_gdf, aoi_geometries, purpose)

        monkeypatch.setattr(exporter_mod, "_clip_roof_geoms_to_aoi", recording_clip)
        classes_df = pd.DataFrame({"description": ["Roof"]}, index=pd.Index([ROOF_ID], name="id"))
        with (
            patch("nmaipy.exporter.FeatureApi") as feature_api,
            patch("nmaipy.exporter.RoofAgeApi") as roof_age_api,
            patch("nmaipy.exporter.collect_latency_stats_from_apis", return_value=None),
        ):
            feature_api.return_value.get_features_gdf_bulk.return_value = self._feature_api_response()
            roof_age_api.return_value.get_roof_age_bulk.return_value = self._roof_age_response()
            exporter.process_chunk(chunk_id="0000", aoi_gdf=self._aoi_gdf(), classes_df=classes_df)

        chunk_path = Path(exporter.chunk_path)
        assert (chunk_path / "rollup_0000.parquet").exists(), "The chunk must complete"
        # Both users of the clip map must get it, holding exactly the gridded AOI.
        assert clips == {("Roof Age link", ("p1",)), ("Roof component recalculation", ("p1",))}, clips

        roofs = pd.read_parquet(chunk_path / "roof_0000.parquet").set_index("feature_id")
        primary = roofs["primary_child_roof_age_feature_id"]
        assert primary["p1-row"] == "p1-ours", "The gridded AOI's roof must link by its AOI-clipped outline"
        assert primary["p2-row"] == "p2-neighbour", "The non-gridded AOI's roof must link by its whole outline"
        # Per-class chunk files hold geometry as plain WKB.
        roof_features = pd.read_parquet(chunk_path / "roof_features_0000.parquet").set_index("feature_id")
        for aoi_id in ("p1", "p2"):
            written = shapely.from_wkb(roof_features.loc[f"{aoi_id}-row", "geometry"])
            assert written.equals_exact(self._place(_ROW_ROOF, aoi_id), tolerance=1e-12), (
                f"{aoi_id}'s roof must be written with its whole outline"
            )
