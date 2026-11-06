"""Gridded AOIs link Roof Age instances by the roof outline clipped to the AOI."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from nmaipy import parcels
from nmaipy.constants import API_CRS
from nmaipy.exporter import _link_roof_instances

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
    _, roofs_linked = _link_roof_instances(
        _instances(*_row_instances("a")), _roofs(("a", "row", _ROW_ROOF)), {"a": _AOI}
    )
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
        linked.equals(original) for linked, original in zip(roofs_linked.geometry, roofs.geometry)
    ), "Every linked roof must keep its whole outline"
    by_fid = _by_feature_id(roofs_linked)
    assert by_fid.loc["a-row", "primary_child_roof_age_feature_id"] == "a-ours"
    assert by_fid.loc["b-row", "primary_child_roof_age_feature_id"] == "b-neighbour"
    assert by_fid.loc["a-shed", "primary_child_roof_age_feature_id"] == "a-shed-instance"
